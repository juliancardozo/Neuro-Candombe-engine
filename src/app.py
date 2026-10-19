import logging
import os
import sys
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# `uvicorn src.app:app` imports this as a package; the modules in src/ import
# each other by plain name, so put src/ on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from candombe_rules import CLAVE, REPIQUE, empty_fixed, fix_from_piece, render_piece  # noqa: E402
from generate import generate_piece, load_model  # noqa: E402
from representation import (  # noqa: E402
    ARTICULATIONS,
    INSTRUMENTS,
    SECTIONS,
    STEPS_PER_BAR,
    Piece,
    articulation_id,
    default_structure,
    offset_to_bin,
    piece_to_events,
    velocity_to_class,
)
from toques import ALIASES, LEVELS, TOQUES, dynamics_curve, get_toque, tempo_curve  # noqa: E402

logger = logging.getLogger("neuro-candombe")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_VERSION = 3
MODEL_PATH = os.environ.get("MODEL_PATH", "models/candombe_v3.pt")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:  # fall back to rules, but say why
        logger.warning("Could not load %s (%s); using the rule engine", MODEL_PATH, exc)
        model = None

MAX_BARS = 32
MAX_REPIQUES = 4


def _parse_structure(structure: Optional[str], bars: int) -> list:
    if not structure:
        return default_structure(bars)
    parts = [part.strip() for part in structure.split(",") if part.strip()]
    unknown = [part for part in parts if part not in SECTIONS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown sections {unknown}; use {list(SECTIONS)}")
    # Repeat or trim the plan to cover every bar.
    return [parts[i % len(parts)] for i in range(bars)] if parts else default_structure(bars)


def _palmas_events(step_duration: float, structure: list) -> list:
    events = []
    for bar_index, section in enumerate(structure):
        for step_in_bar, _, level in CLAVE.events:
            step = bar_index * STEPS_PER_BAR + step_in_bar
            events.append(
                {
                    "time": step * step_duration,
                    "instrument": "clave",
                    "velocity": LEVELS[level],
                    "offset_ms": 0.0,
                    "step": step,
                    "section": section,
                    "articulation": "palmas",
                    "hit_type": "palmas",
                    "voice": 0,
                }
            )
    return events


class GenerationParams(BaseModel):
    style: str = "ansina"
    tempo: Optional[int] = None  # defaults to the toque's typical tempo
    bars: int = 16
    subir: bool = False
    temperature: float = 0.9
    density: float = 0.55
    top_p: float = 0.92
    structure: Optional[str] = None
    lock_chico: bool = True
    strict_rules: bool = True
    palmas: bool = False
    guidance: float = 1.0
    timing_temperature: float = 0.5
    repiques: int = 1
    random_seed: Optional[int] = None


class RegenerateRequest(GenerationParams):
    """Regenerate part of a previous result; everything not selected stays fixed."""

    events: List[dict] = Field(default_factory=list)
    keep_instruments: List[str] = Field(default_factory=list)
    regenerate_bars: Optional[List[int]] = None


def _clamp_params(params: GenerationParams) -> GenerationParams:
    params = params.model_copy()
    style = (params.style or "ansina").lower()
    if ALIASES.get(style, style) not in TOQUES:
        raise HTTPException(status_code=422, detail=f"Unknown toque {params.style}; use {list(TOQUES)}")
    params.style = ALIASES.get(style, style)
    params.tempo = max(50, min(220, params.tempo or get_toque(params.style).tempo))
    params.bars = max(4, min(MAX_BARS, params.bars))
    params.repiques = max(1, min(MAX_REPIQUES, params.repiques))
    params.timing_temperature = max(0.1, min(1.5, params.timing_temperature))
    params.temperature = max(0.3, min(1.8, params.temperature))
    params.density = max(0.05, min(0.95, params.density))
    params.top_p = max(0.5, min(1.0, params.top_p))
    params.guidance = max(0.0, min(3.0, params.guidance))
    if params.random_seed is None:
        # Pick the seed here so the response can report it and the result can be repeated.
        params.random_seed = int(np.random.SeedSequence().entropy % (2**31))
    return params


def _fixed_from_events(events: List[dict], steps: int, step_duration: float, keep: np.ndarray) -> dict:
    """Fixed cells for the `keep` mask (steps, 3) from a previous result's events."""
    fixed = empty_fixed(steps)
    fixed["vel"][keep] = 0  # kept cells are silent unless an event says otherwise
    for event in events:
        instrument = event.get("instrument")
        if instrument not in INSTRUMENTS or int(event.get("voice", 0)) != 0:
            continue
        inst = INSTRUMENTS.index(instrument)
        step = int(event.get("step", round(float(event.get("time", 0.0)) / step_duration)))
        if not 0 <= step < steps or not keep[step, inst]:
            continue
        fixed["vel"][step, inst] = max(1, int(velocity_to_class(np.array([int(event.get("velocity", 0))]))[0]))
        offset_frac = float(event.get("offset_ms", 0.0)) / (step_duration * 1000.0)
        fixed["off"][step, inst] = int(offset_to_bin(np.array([offset_frac]))[0])
        art_name = event.get("articulation") or event.get("hit_type") or ""
        fixed["art"][step, inst] = articulation_id(instrument, art_name)
    return fixed


def _support_repique(piece: Piece, sections: list, other: Piece) -> Piece:
    """A second repiquero for the rule engine: own phrases in the base, madera
    while the lead gives the call or the closing signal [S13]."""
    variant = Piece(piece.vel.copy(), piece.off.copy(), piece.art.copy(), piece.section.copy(), piece.tempo, piece.style)
    madera = articulation_id("repique", "madera")
    for bar, section in enumerate(sections):
        cells = slice(bar * STEPS_PER_BAR, (bar + 1) * STEPS_PER_BAR)
        if section in ("base", "llamada_piano"):
            variant.vel[cells, REPIQUE] = other.vel[cells, REPIQUE]
            variant.off[cells, REPIQUE] = other.off[cells, REPIQUE]
            variant.art[cells, REPIQUE] = other.art[cells, REPIQUE]
        elif section in ("llamada_repique", "cierre"):
            variant.vel[cells, REPIQUE] = 0
            variant.off[cells, REPIQUE] = -1
            variant.art[cells, REPIQUE] = -1
            for step, _, _ in CLAVE.events:
                variant.vel[bar * STEPS_PER_BAR + step, REPIQUE] = 2
                variant.off[bar * STEPS_PER_BAR + step, REPIQUE] = int(offset_to_bin(np.array([0.0]))[0])
                variant.art[bar * STEPS_PER_BAR + step, REPIQUE] = madera
    return variant


def _compose(params: GenerationParams, keep: Optional[np.ndarray] = None, previous: Optional[List[dict]] = None) -> dict:
    params = _clamp_params(params)
    toque = get_toque(params.style)
    bars = params.bars
    steps = bars * STEPS_PER_BAR
    step_duration = 60.0 / params.tempo / 4  # 16th-note grid at the target tempo
    sections = _parse_structure(params.structure, bars)
    tempos = tempo_curve(toque, bars, params.tempo, params.subir, np.random.default_rng(params.random_seed))
    dynamics = dynamics_curve(sections, params.subir)

    kept = _fixed_from_events(previous or [], steps, step_duration, keep) if keep is not None else None

    def model_piece(fixed: Optional[dict], seed: int, lead: bool = True) -> Piece:
        return generate_piece(
            model,
            bars=bars,
            tempo=params.tempo,
            density=params.density,
            structure=sections,
            style=toque.key,
            fixed=fixed,
            chico_locked=params.lock_chico,
            strict_rules=params.strict_rules,
            tempos=tempos,
            lead_repique=lead,
            temperature=params.temperature,
            top_p=params.top_p,
            guidance=params.guidance,
            timing_temperature=params.timing_temperature,
            seed=seed,
        )

    if model is not None:
        piece = model_piece(kept, params.random_seed)
        engine = "model"
    else:
        rng = np.random.default_rng(params.random_seed)
        piece = render_piece(rng, sections, density=params.density, tempo=params.tempo, style=toque.key, tempos=tempos)
        if kept is not None:
            piece.vel[keep] = kept["vel"][keep]
            piece.off[keep] = np.where(kept["vel"][keep] > 0, kept["off"][keep], -1)
            piece.art[keep] = np.where(kept["vel"][keep] > 0, kept["art"][keep], -1)
        engine = "rules"

    # Every extra repiquero improvises its own line over the same chico and piano,
    # and plays madera while the lead repique gives the signals.
    extra_repiques = []
    shared = np.repeat([section in ("intro", "final") for section in sections], STEPS_PER_BAR)
    for voice in range(1, params.repiques):
        voice_seed = params.random_seed + 7919 * voice
        if model is not None:
            fixed = fix_from_piece(empty_fixed(steps), piece, ["chico", "piano"])
            for key in fixed:
                fixed[key][shared, REPIQUE] = getattr(piece, key)[shared, REPIQUE]
            extra_repiques.append(model_piece(fixed, voice_seed, lead=False))
        else:
            other = render_piece(np.random.default_rng(voice_seed), sections, density=params.density, tempo=params.tempo, style=toque.key, tempos=tempos)
            extra_repiques.append(_support_repique(piece, sections, other))

    events = piece_to_events(piece, step_duration, sections)
    for event in events:
        event["voice"] = 0
    for voice, variant in enumerate(extra_repiques, start=1):
        for event in piece_to_events(variant, step_duration, sections):
            if event["instrument"] == "repique":
                event["voice"] = voice
                events.append(event)
    if params.palmas:
        events.extend(_palmas_events(step_duration, sections))
    events.sort(key=lambda event: (float(event["time"]), event["instrument"]))
    return {
        "version": API_VERSION,
        "engine": engine,
        "style": toque.key,
        "toque": {"label": toque.label, "barrio": toque.barrio},
        "tempo": params.tempo,
        "tempo_curve": tempos,
        "dynamics": dynamics,
        "bars": bars,
        "step_duration": step_duration,
        "model_loaded": model is not None,
        "structure": sections,
        "random_seed": params.random_seed,
        "repiques": params.repiques,
        "articulations": {name: list(values) for name, values in ARTICULATIONS.items()},
        "events": events,
    }


@app.get("/generate")
def generate(
    style: str = "ansina",
    tempo: Optional[int] = None,
    bars: int = 16,
    subir: bool = False,
    temperature: float = 0.9,
    density: float = 0.55,
    top_p: float = 0.92,
    structure: Optional[str] = None,
    lock_chico: bool = True,
    strict_rules: bool = True,
    palmas: bool = False,
    guidance: float = 1.0,
    timing_temperature: float = 0.5,
    repiques: int = 1,
    random_seed: Optional[int] = None,
):
    return _compose(GenerationParams(**locals()))


@app.post("/regenerate")
def regenerate(request: RegenerateRequest):
    unknown = set(request.keep_instruments) - set(INSTRUMENTS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown instruments {sorted(unknown)}; use {list(INSTRUMENTS)}")
    bars = max(4, min(MAX_BARS, request.bars))
    keep = np.zeros((bars * STEPS_PER_BAR, len(INSTRUMENTS)), dtype=bool)
    for name in request.keep_instruments:
        keep[:, INSTRUMENTS.index(name)] = True
    if request.regenerate_bars is not None:
        for bar in range(bars):
            if bar not in request.regenerate_bars:
                keep[bar * STEPS_PER_BAR : (bar + 1) * STEPS_PER_BAR] = True
    if keep.all():
        raise HTTPException(status_code=422, detail="Nothing to regenerate: every drum and bar is kept")
    return _compose(request, keep=keep, previous=request.events)


@app.get("/toques")
def toques():
    """The toques madre with their documented character, for the UI."""
    return [
        {
            "key": t.key,
            "label": t.label,
            "barrio": t.barrio,
            "description": t.description,
            "tempo": t.tempo,
            "tempo_range": list(t.tempo_range),
            "chico": "de tres" if t.chico_de_tres_always else "de dos (de tres a tempo lento)",
        }
        for t in TOQUES.values()
    ]


@app.get("/health")
def health():
    return {"ok": True, "version": API_VERSION, "model_loaded": model is not None, "model_path": MODEL_PATH}


# Serve web/ directory at root after API routes so `/generate` is not shadowed.
app.mount("/", StaticFiles(directory="web", html=True), name="web")
