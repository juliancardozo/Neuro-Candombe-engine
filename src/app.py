import copy
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

from candombe_rules import CLAVE_STEPS, CLAVE_VELOCITIES, empty_fixed, fix_from_piece, render_piece  # noqa: E402
from candombe_rules import lock_chico as lock_chico_cells  # noqa: E402
from generate import generate_piece, load_model  # noqa: E402
from representation import (  # noqa: E402
    ARTICULATIONS,
    INSTRUMENTS,
    SECTIONS,
    Piece,
    STEPS_PER_BAR,
    STYLES,
    articulation_id,
    default_structure,
    offset_to_bin,
    piece_to_events,
    velocity_to_class,
)

logger = logging.getLogger("neuro-candombe")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_VERSION = 2
MODEL_PATH = os.environ.get("MODEL_PATH", "models/candombe_v2.pt")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:  # fall back to rules, but say why
        logger.warning("Could not load %s (%s); using the rule engine", MODEL_PATH, exc)
        model = None

MAX_BARS = 32
MAX_REPIQUES = 4

# The `seed` control picks how the piece starts.
SEED_DENSITY_SHIFT = {"clave": 0.0, "base": -0.1, "energia": 0.15, "energy": 0.15}


def _parse_structure(structure: Optional[str], bars: int) -> list:
    if not structure:
        return default_structure(bars)
    parts = [part.strip() for part in structure.split(",") if part.strip()]
    unknown = [part for part in parts if part not in SECTIONS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown sections {unknown}; use {list(SECTIONS)}")
    # Repeat or trim the plan to cover every bar.
    return [parts[i % len(parts)] for i in range(bars)] if parts else default_structure(bars)


def _clave_intro(fixed: dict, structure: list) -> dict:
    """`seed=clave`: the repique opens the intro bars playing the clave on the madera."""
    for bar_index, section in enumerate(structure):
        if section != "intro":
            continue
        start = bar_index * STEPS_PER_BAR
        fixed["vel"][start : start + STEPS_PER_BAR, 1] = 0
        for step_in_bar, velocity in zip(CLAVE_STEPS, CLAVE_VELOCITIES):
            fixed["vel"][start + step_in_bar, 1] = 3 if velocity >= 65 else 2
            fixed["art"][start + step_in_bar, 1] = articulation_id("repique", "madera")
    return fixed


def _entrance(fixed: dict, structure: list) -> dict:
    """Like a real llamada, chicos and pianos wait while the repiques mark the intro."""
    for bar_index, section in enumerate(structure):
        if section == "intro":
            start = bar_index * STEPS_PER_BAR
            fixed["vel"][start : start + STEPS_PER_BAR, 0] = 0
            fixed["vel"][start : start + STEPS_PER_BAR, 2] = 0
    return fixed


def _build_clave_events(step_duration: float, structure: list) -> list:
    events = []
    for bar_index, section in enumerate(structure):
        for step_in_bar, velocity in zip(CLAVE_STEPS, CLAVE_VELOCITIES):
            step = bar_index * STEPS_PER_BAR + step_in_bar
            events.append(
                {
                    "time": step * step_duration,
                    "instrument": "clave",
                    "velocity": velocity,
                    "offset_ms": 0.0,
                    "step": step,
                    "section": section,
                    "articulation": "palmas",
                    "hit_type": "palmas",
                }
            )
    return events


class GenerationParams(BaseModel):
    tempo: int = 90
    bars: int = 4
    seed: str = "clave"
    temperature: float = 0.9
    density: float = 0.55
    top_p: float = 0.92
    structure: Optional[str] = None
    style: Optional[str] = None
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
    params.tempo = max(50, min(220, params.tempo))
    params.bars = max(4, min(MAX_BARS, params.bars))
    params.repiques = max(1, min(MAX_REPIQUES, params.repiques))
    params.timing_temperature = max(0.1, min(1.5, params.timing_temperature))
    params.temperature = max(0.3, min(1.8, params.temperature))
    params.density = max(0.05, min(0.95, params.density))
    params.top_p = max(0.5, min(1.0, params.top_p))
    params.guidance = max(0.0, min(3.0, params.guidance))
    if params.style is not None and params.style not in STYLES:
        raise HTTPException(status_code=422, detail=f"Unknown style {params.style}; use {list(STYLES)}")
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


def _compose(params: GenerationParams, keep: Optional[np.ndarray] = None, previous: Optional[List[dict]] = None) -> dict:
    params = _clamp_params(params)
    bars = params.bars
    steps = bars * STEPS_PER_BAR
    seed_kind = (params.seed or "clave").lower()
    density = float(np.clip(params.density + SEED_DENSITY_SHIFT.get(seed_kind, 0.0), 0.05, 0.95))
    step_duration = 60.0 / params.tempo / 4  # 16th-note grid
    sections = _parse_structure(params.structure, bars)

    kept = _fixed_from_events(previous or [], steps, step_duration, keep) if keep is not None else None

    def model_piece(fixed: dict, seed: int) -> Piece:
        return generate_piece(
            model,
            bars=bars,
            tempo=params.tempo,
            density=density,
            structure=sections,
            style=params.style,
            fixed=fixed,
            chico_locked=False,  # applied by the caller so kept cells can override it
            strict_rules=params.strict_rules,
            temperature=params.temperature,
            top_p=params.top_p,
            guidance=params.guidance,
            timing_temperature=params.timing_temperature,
            seed=seed,
        )

    intro = np.repeat([section == "intro" for section in sections], STEPS_PER_BAR)
    if model is not None:
        fixed = empty_fixed(steps)
        if seed_kind == "clave":
            fixed = _clave_intro(fixed, sections)
        if params.lock_chico:
            fixed = lock_chico_cells(fixed)
        fixed = _entrance(fixed, sections)
        if kept is not None:
            for key in fixed:
                fixed[key][keep] = kept[key][keep]
        piece = model_piece(fixed, params.random_seed)
        engine = "model"
    else:
        rng = np.random.default_rng(params.random_seed)
        piece = render_piece(rng, sections, density=density, tempo=params.tempo, style=params.style)
        for inst in (0, 2):
            piece.vel[intro, inst] = 0
            piece.off[intro, inst] = -1
            piece.art[intro, inst] = -1
        if kept is not None:
            piece.vel[keep] = kept["vel"][keep]
            piece.off[keep] = np.where(kept["vel"][keep] > 0, kept["off"][keep], -1)
            piece.art[keep] = np.where(kept["vel"][keep] > 0, kept["art"][keep], -1)
        engine = "rules"

    # Every extra repiquero improvises a different line over the same chico and
    # piano; in the intro they all mark the same clave.
    extra_repiques = []
    for voice in range(1, params.repiques):
        voice_seed = params.random_seed + 7919 * voice
        if model is not None:
            fixed = fix_from_piece(empty_fixed(steps), piece, ["chico", "piano"])
            for key in fixed:
                fixed[key][intro, 1] = getattr(piece, key)[intro, 1]
            extra_repiques.append(model_piece(fixed, voice_seed))
        else:
            other = render_piece(np.random.default_rng(voice_seed), sections, density=density, tempo=params.tempo)
            variant = copy.deepcopy(piece)
            free = ~intro
            variant.vel[free, 1] = other.vel[free, 1]
            variant.off[free, 1] = other.off[free, 1]
            variant.art[free, 1] = other.art[free, 1]
            extra_repiques.append(variant)

    events = piece_to_events(piece, step_duration, sections)
    for event in events:
        event["voice"] = 0
    for voice, variant in enumerate(extra_repiques, start=1):
        for event in piece_to_events(variant, step_duration, sections):
            if event["instrument"] == "repique":
                event["voice"] = voice
                events.append(event)
    if params.palmas:
        events.extend(_build_clave_events(step_duration, sections))
    events.sort(key=lambda event: (float(event["time"]), event["instrument"]))
    return {
        "version": API_VERSION,
        "engine": engine,
        "tempo": params.tempo,
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
    tempo: int = 90,
    bars: int = 4,
    seed: str = "clave",
    temperature: float = 0.9,
    density: float = 0.55,
    top_p: float = 0.92,
    structure: Optional[str] = None,
    style: Optional[str] = None,
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


@app.get("/health")
def health():
    return {"ok": True, "version": API_VERSION, "model_loaded": model is not None, "model_path": MODEL_PATH}


# Serve web/ directory at root after API routes so `/generate` is not shadowed.
app.mount("/", StaticFiles(directory="web", html=True), name="web")
