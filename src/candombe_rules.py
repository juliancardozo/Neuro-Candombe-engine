"""Candombe rules built on the documented toques (toques.py).

- render_piece: a rule-based llamada in a given toque (Cuareim, Ansina or
  Gaboto). It feeds the synthetic corpus and is the API fallback.
- Constraint helpers for the model's sampler: they only fill cells the caller
  left free, so explicitly fixed cells always win.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from build_repique_locked_dataset import REPIQUE_TEMPLATES, clamp
from representation import (
    INSTRUMENTS,
    N_ART,
    N_INSTR,
    N_VEL,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
    Piece,
    articulation_id,
    default_structure,
    make_piece,
    velocity_to_class,
)
from toques import (
    CLAVE,
    LEVELS,
    PIANO_REPICADOS,
    REPIQUE_BASICO,
    REPIQUE_CIERRE,
    REPIQUE_LLAMADA,
    REPIQUE_REPICADO,
    TIMING_SPREAD,
    Pattern,
    Toque,
    chico_pattern,
    get_toque,
    groove_offsets,
    tempo_curve,
)

CHICO, REPIQUE, PIANO = range(N_INSTR)

# Bars where the base keeps running underneath.
GROOVE_SECTIONS = ("base", "llamada_repique", "llamada_piano", "cierre")


def _team_phrases() -> List[Pattern]:
    """The project's earlier repique templates, kept as extra improvised phrases."""
    phrases = []
    for index, template in enumerate(REPIQUE_TEMPLATES):
        events = []
        for step, hit in enumerate(template):
            if hit:
                level = "accent" if hit["velocity"] >= 100 else "normal" if hit["velocity"] >= 70 else "light"
                events.append((step, "palo" if hit["velocity"] >= 60 else "mano", level))
        phrases.append(Pattern(f"Frase {index + 1}", "plantillas del proyecto", tuple(events)))
    return phrases


REPIQUE_PHRASES = tuple(_team_phrases())
FINAL_HIT = {
    CHICO: Pattern("Final", "S13 ex.4.7", ((0, "palo", "accent"),)),
    REPIQUE: Pattern("Final", "S13 ex.4.7", ((0, "rimshot", "accent"),)),
    PIANO: Pattern("Final", "S13 ex.4.7", ((0, "masa", "accent"),)),
}


def level_class(level: str) -> int:
    return int(velocity_to_class(np.array([LEVELS[level]]))[0])


def cierre_index(structure: Sequence[str], bar: int) -> int:
    """0 for the first bar of the closing signal, 1 for the second."""
    previous = bar - 1
    return 1 if previous >= 0 and structure[previous] == "cierre" else 0


# --- rule-based llamada ---------------------------------------------------------------------

def random_structure(rng: np.random.Generator, bars: int) -> List[str]:
    """The llamada form with a few extra calls sprinkled in, for corpus variety."""
    structure = default_structure(bars)
    for bar, section in enumerate(structure):
        if section == "base" and structure[bar - 1] == "base":
            roll = rng.random()
            if roll < 0.08:
                structure[bar] = "llamada_repique"
            elif roll < 0.16:
                structure[bar] = "llamada_piano"
    return structure


def _repique_pattern(rng: np.random.Generator, section: str, structure, bar: int, density: float) -> Pattern:
    if section == "intro":
        return CLAVE
    if section == "llamada_repique":
        return REPIQUE_LLAMADA
    if section == "cierre":
        return REPIQUE_CIERRE[cierre_index(structure, bar)]
    if section == "final":
        return FINAL_HIT[REPIQUE]
    # Base: the repique alternates its primary pattern, madera cycles and phrases [J22].
    options = [(REPIQUE_BASICO, 0.45 - 0.2 * density), (CLAVE, 0.12), (REPIQUE_REPICADO, 0.1 + 0.2 * density)]
    options += [(phrase, (0.33 + 0.1 * density) / len(REPIQUE_PHRASES)) for phrase in REPIQUE_PHRASES]
    weights = np.array([w for _, w in options])
    return options[int(rng.choice(len(options), p=weights / weights.sum()))][0]


def _choose_base(rng: np.random.Generator, toque: Toque) -> Pattern:
    weights = np.array([p.weight for p in toque.piano_bases])
    return toque.piano_bases[int(rng.choice(len(weights), p=weights / weights.sum()))]


def render_piece(
    rng: np.random.Generator,
    structure: Sequence[str],
    density: float = 0.5,
    tempo: Optional[float] = None,
    style: Optional[str] = None,
    subir: bool = False,
    tempos: Optional[Sequence[float]] = None,
    profile=None,  # kept for callers of the old signature; timing comes from toques
) -> Piece:
    """`tempos` (one per bar) overrides the tempo curve derived from `subir`."""
    toque = get_toque(style)
    tempo = float(tempo or toque.tempo)
    bars = len(structure)
    curve = list(tempos) if tempos is not None else tempo_curve(toque, bars, tempo, subir, rng)
    steps = bars * STEPS_PER_BAR
    velocities = np.zeros((steps, N_INSTR), dtype=np.int32)
    articulation = np.full((steps, N_INSTR), -1, dtype=np.int64)
    offsets = np.full((steps, N_INSTR), np.nan)
    density = float(np.clip(density, 0.0, 1.0))

    def put(bar: int, inst: int, pattern: Pattern, scale: float = 1.0) -> None:
        start = bar * STEPS_PER_BAR
        groove = groove_offsets(toque, INSTRUMENTS[inst], curve[bar])
        for step, art, level in pattern.events:
            velocity = clamp(int(rng.normal(LEVELS[level] * scale, 5)), 1, 127)
            velocities[start + step, inst] = velocity
            articulation[start + step, inst] = articulation_id(INSTRUMENTS[inst], art)
            mean = 0.0 if art == "madera" or pattern is FINAL_HIT[inst] else groove[step % STEPS_PER_BEAT]
            offsets[start + step, inst] = float(np.clip(rng.normal(mean, TIMING_SPREAD[INSTRUMENTS[inst]]), -0.4, 0.4))

    base = _choose_base(rng, toque)
    for bar, section in enumerate(structure):
        # Under the repique's closing signal the cuerda plays softer [S13].
        support = 0.75 if section == "cierre" else 1.0
        if section == "intro":
            for inst in range(N_INSTR):
                put(bar, inst, CLAVE)
            continue
        if section == "final":
            for inst in range(N_INSTR):
                put(bar, inst, FINAL_HIT[inst])
            continue

        put(bar, CHICO, chico_pattern(toque, curve[bar]), support)
        put(bar, REPIQUE, _repique_pattern(rng, section, structure, bar, density))

        if rng.random() < toque.piano_change:
            base = _choose_base(rng, toque)
        repicado = section == "llamada_piano" or (section == "base" and rng.random() < toque.piano_repicado * (0.5 + density))
        if repicado:
            put(bar, PIANO, PIANO_REPICADOS[int(rng.integers(0, len(PIANO_REPICADOS)))], support)
        else:
            put(bar, PIANO, base, support)

        # Players add ghost notes to the basic forms [J22]; more with density.
        start = bar * STEPS_PER_BAR
        for step in range(STEPS_PER_BAR):
            for inst, art, chance in ((REPIQUE, "mano", 0.18), (PIANO, "mano", 0.08)):
                if velocities[start + step, inst] == 0 and rng.random() < chance * density:
                    velocities[start + step, inst] = clamp(int(rng.normal(LEVELS["ghost"], 4)), 1, 127)
                    articulation[start + step, inst] = articulation_id(INSTRUMENTS[inst], art)
                    groove = groove_offsets(toque, INSTRUMENTS[inst], curve[bar])
                    offsets[start + step, inst] = float(rng.normal(groove[step % 4], TIMING_SPREAD[INSTRUMENTS[inst]]))

    return make_piece(velocities, offsets, articulation, list(structure), tempo=tempo, style=toque.key, source="rules")


# --- constraints for the model sampler ------------------------------------------------------

def empty_fixed(steps: int) -> Dict[str, np.ndarray]:
    """Fixed cells: -1 means free, otherwise the value the sampler must keep."""
    return {key: np.full((steps, N_INSTR), -1, dtype=np.int64) for key in ("vel", "off", "art")}


def _fill_bar(fixed: Dict[str, np.ndarray], bar: int, inst: int, pattern: Optional[Pattern], scale: float = 1.0) -> None:
    """Fix one instrument for a whole bar to `pattern` (None = silence), free cells only."""
    start = bar * STEPS_PER_BAR
    if start >= fixed["vel"].shape[0]:
        return
    events = {step: (art, level) for step, art, level in (pattern.events if pattern else ())}
    for step in range(STEPS_PER_BAR):
        cell = start + step
        if cell >= fixed["vel"].shape[0] or fixed["vel"][cell, inst] >= 0:
            continue
        if step in events:
            art, level = events[step]
            fixed["vel"][cell, inst] = level_class(level) if scale >= 1.0 else max(1, level_class(level) - 1)
            fixed["art"][cell, inst] = articulation_id(INSTRUMENTS[inst], art)
        else:
            fixed["vel"][cell, inst] = 0


def lock_chico(
    fixed: Dict[str, np.ndarray],
    structure: Optional[Sequence[str]] = None,
    style: Optional[str] = None,
    tempos: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Fix the chico: its pattern follows the toque and each bar's tempo; its
    microtiming stays free for the model."""
    toque = get_toque(style)
    bars = fixed["vel"].shape[0] // STEPS_PER_BAR
    structure = list(structure) if structure is not None else ["base"] * bars
    for bar in range(bars):
        section = structure[bar] if bar < len(structure) else "base"
        if section in ("intro", "final"):
            continue  # set by apply_form
        tempo = tempos[bar] if tempos is not None else toque.tempo
        _fill_bar(fixed, bar, CHICO, chico_pattern(toque, tempo), 0.75 if section == "cierre" else 1.0)
    return fixed


def apply_form(fixed: Dict[str, np.ndarray], structure: Sequence[str], lead_repique: bool = True) -> Dict[str, np.ndarray]:
    """The parts of a llamada that are signals, not improvisation: everyone on the
    madera in the intro, the repique's call and closing signal, the final hit.
    A non-lead repique plays madera while the lead gives the signals [S13]."""
    for bar, section in enumerate(structure):
        if section == "intro":
            for inst in range(N_INSTR):
                _fill_bar(fixed, bar, inst, CLAVE)
        elif section == "final":
            for inst in range(N_INSTR):
                _fill_bar(fixed, bar, inst, FINAL_HIT[inst])
        elif section == "llamada_repique":
            _fill_bar(fixed, bar, REPIQUE, REPIQUE_LLAMADA if lead_repique else CLAVE)
        elif section == "cierre":
            _fill_bar(fixed, bar, REPIQUE, REPIQUE_CIERRE[cierre_index(structure, bar)] if lead_repique else CLAVE, 1.0 if lead_repique else 0.5)
    return fixed


def fix_from_piece(fixed: Dict[str, np.ndarray], piece: Piece, instruments: Sequence[str]) -> Dict[str, np.ndarray]:
    """Fix whole tracks from a given piece, e.g. generate repique over a given piano."""
    steps = min(fixed["vel"].shape[0], piece.steps)
    for name in instruments:
        inst = INSTRUMENTS.index(name)
        fixed["vel"][:steps, inst] = piece.vel[:steps, inst]
        fixed["off"][:steps, inst] = piece.off[:steps, inst]
        fixed["art"][:steps, inst] = piece.art[:steps, inst]
    return fixed


def allowed_masks(steps: int, structure: Sequence[str], style: Optional[str] = None):
    """Invariants of the toque as allowed classes for the sampler.

    Velocity mask (steps, 3, N_VEL) and articulation mask (steps, 3, max_art):
      - in groove bars the piano plays its masa on the downbeat and the accented
        open stroke on the 4th sixteenth, "one of the most important features of
        the candombe rhythm" [J22];
      - in Gaboto the masa always falls on the 2nd sixteenth of beat two [S13].
    Piano repicado bars (llamada_piano) are left free.
    """
    toque = get_toque(style)
    vel = np.ones((steps, N_INSTR, N_VEL), dtype=bool)
    art = np.ones((steps, N_INSTR, max(N_ART)), dtype=bool)
    for inst in range(N_INSTR):
        art[:, inst, N_ART[inst]:] = False
    masa = articulation_id("piano", "masa")
    abierto = articulation_id("piano", "abierto")

    def require(step: int, articulation_index: int, min_class: int) -> None:
        if step >= steps:
            return
        vel[step, PIANO, :min_class] = False
        art[step, PIANO, :] = False
        art[step, PIANO, articulation_index] = True

    for bar, section in enumerate(structure):
        if section not in ("base", "llamada_repique", "cierre"):
            continue
        start = bar * STEPS_PER_BAR
        require(start, masa, 2)
        require(start + 3, abierto, 3)
        if toque.key == "gaboto":
            require(start + 5, masa, 2)
    return vel, art
