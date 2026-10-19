"""Candombe rules: canonical patterns, a rule-based renderer and generation constraints.

The renderer produces section-aware pieces with articulations and microtiming.
It feeds the synthetic corpus and is the API fallback when no checkpoint is
loaded. The constraint helpers express rules as fixed cells / allowed classes
for the model's sampler instead of overwriting its output afterwards.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from build_chico_locked_dataset import PIANO_SUPPORT, REPIQUE_SUPPORT
from build_piano_grounded_dataset import PIANO_VARIATIONS, REPIQUE_REFERENCE
from build_repique_locked_dataset import CHICO_TEMPLATE, PIANO_TEMPLATES, REPIQUE_TEMPLATES, clamp, jitter
from microtiming import DEFAULT_PROFILE, sample_offset
from representation import (
    INSTRUMENTS,
    N_INSTR,
    N_VEL,
    STEPS_PER_BAR,
    Piece,
    articulation_id,
    make_piece,
    velocity_to_class,
)

CHICO, REPIQUE, PIANO = range(N_INSTR)

# Chico cadence per beat: rest, mano, palo, palo.
CHICO_BEAT = (None, "mano", "palo", "palo")
CHICO_ARTICULATION = tuple(CHICO_BEAT[step % 4] for step in range(STEPS_PER_BAR))

CLAVE_STEPS = (0, 3, 6, 10, 12)
CLAVE_VELOCITIES = (80, 66, 76, 66, 90)
REPIQUE_CALL = ((11, 104), (13, 96), (14, 112), (15, 92))
PIANO_CALL = ((1, 70, "stick_open"), (9, 74, "stick_open"), (14, 60, "stick_closed"))
PIANO_CLOSE = ((9, 76, "stick_open"), (13, 80, "stick_open"), (15, 62, "stick_closed"))

PIANO_POOL = tuple(PIANO_TEMPLATES) + tuple(PIANO_VARIATIONS) + tuple(PIANO_SUPPORT)
REPIQUE_POOL = tuple(REPIQUE_TEMPLATES) + tuple(REPIQUE_REFERENCE) + tuple(REPIQUE_SUPPORT)


def piano_articulation(step_in_bar: int, velocity: int) -> str:
    if step_in_bar in (0, 8):
        return "bass_open"
    if step_in_bar in (3, 4, 11, 12):
        return "bass_muted"
    return "stick_open" if velocity >= 55 else "stick_closed"


def repique_articulation(velocity: int) -> str:
    return "mano" if velocity < 60 else "palo"


def random_structure(rng: np.random.Generator, bars: int) -> List[str]:
    """Plausible section plan: intro, base with calls, and a closing bar."""
    structure = ["base"] * bars
    intro_bars = int(rng.integers(1, 3)) if bars >= 4 else 1
    for bar_index in range(min(intro_bars, bars)):
        structure[bar_index] = "intro"
    structure[-1] = "cierre"
    for bar_index in range(intro_bars, bars - 1):
        roll = rng.random()
        if roll < 0.22:
            structure[bar_index] = "llamada_repique"
        elif roll < 0.38:
            structure[bar_index] = "llamada_piano"
    return structure


def render_piece(
    rng: np.random.Generator,
    structure: Sequence[str],
    density: float = 0.5,
    profile: Optional[Dict[str, list]] = None,
    tempo: Optional[float] = None,
    style: Optional[str] = None,
) -> Piece:
    profile = profile or DEFAULT_PROFILE
    bars = len(structure)
    steps = bars * STEPS_PER_BAR
    velocities = np.zeros((steps, N_INSTR), dtype=np.int32)
    articulation = np.full((steps, N_INSTR), -1, dtype=np.int64)
    density = float(np.clip(density, 0.0, 1.0))

    def put(step: int, inst: int, velocity: int, art_name: str) -> None:
        velocities[step, inst] = clamp(int(velocity), 1, 127)
        articulation[step, inst] = articulation_id(INSTRUMENTS[inst], art_name)

    piano_template = PIANO_POOL[int(rng.integers(0, len(PIANO_POOL)))]
    for bar_index, section in enumerate(structure):
        offset = bar_index * STEPS_PER_BAR

        for step_in_bar, hit in enumerate(CHICO_TEMPLATE):
            if hit:
                put(offset + step_in_bar, CHICO, jitter(hit["velocity"], 3, rng), CHICO_ARTICULATION[step_in_bar])

        # Piano keeps a template for a few bars before varying, like a real player.
        if rng.random() < 0.35:
            piano_template = PIANO_POOL[int(rng.integers(0, len(PIANO_POOL)))]
        for step_in_bar, hit in enumerate(piano_template):
            if not hit:
                continue
            if hit["velocity"] <= 60 and rng.random() > 0.35 + 0.6 * density:
                continue
            velocity = jitter(hit["velocity"], 6, rng)
            put(offset + step_in_bar, PIANO, velocity, piano_articulation(step_in_bar, velocity))

        if section == "intro":
            for step_in_bar, velocity in zip(CLAVE_STEPS, CLAVE_VELOCITIES):
                put(offset + step_in_bar, REPIQUE, jitter(velocity, 6, rng), "madera")
            continue

        repique_template = REPIQUE_POOL[int(rng.integers(0, len(REPIQUE_POOL)))]
        last_step = 11 if section in ("llamada_repique", "cierre") else STEPS_PER_BAR
        for step_in_bar, hit in enumerate(repique_template[:last_step]):
            if not hit:
                continue
            if hit["velocity"] <= 62 and rng.random() > 0.3 + 0.7 * density:
                continue
            if section == "llamada_piano" and hit["velocity"] < 90 and rng.random() < 0.5:
                continue  # repique gives room while the piano calls
            velocity = jitter(hit["velocity"], 8, rng)
            put(offset + step_in_bar, REPIQUE, velocity, repique_articulation(velocity))

        # Extra ghost strokes on free off-beats when the density knob is high.
        for step_in_bar in range(1, last_step, 2):
            step = offset + step_in_bar
            if velocities[step, REPIQUE] == 0 and rng.random() < 0.25 * max(0.0, density - 0.4):
                put(step, REPIQUE, jitter(48, 6, rng), "mano")

        if section in ("llamada_repique", "cierre"):
            for step_in_bar, velocity in REPIQUE_CALL:
                put(offset + step_in_bar, REPIQUE, jitter(velocity, 6, rng), "palo")
        if section == "llamada_piano":
            for step_in_bar, velocity, art_name in PIANO_CALL:
                put(offset + step_in_bar, PIANO, jitter(velocity, 6, rng), art_name)
        if section == "cierre":
            for step_in_bar, velocity, art_name in PIANO_CLOSE:
                put(offset + step_in_bar, PIANO, jitter(velocity, 6, rng), art_name)

    offsets = np.full((steps, N_INSTR), np.nan, dtype=np.float64)
    for step in range(steps):
        for inst in range(N_INSTR):
            if velocities[step, inst] > 0:
                offsets[step, inst] = sample_offset(profile, INSTRUMENTS[inst], step, rng)

    return make_piece(velocities, offsets, articulation, list(structure), tempo=tempo, style=style, source="rules")


# --- constraints for the model sampler -------------------------------------------------

def empty_fixed(steps: int) -> Dict[str, np.ndarray]:
    """Fixed cells: -1 means free, otherwise the value the sampler must keep."""
    return {key: np.full((steps, N_INSTR), -1, dtype=np.int64) for key in ("vel", "off", "art")}


def lock_chico(fixed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fix the chico cadence (hits and articulation); its microtiming stays free."""
    steps = fixed["vel"].shape[0]
    for step in range(steps):
        step_in_bar = step % STEPS_PER_BAR
        hit = CHICO_TEMPLATE[step_in_bar]
        if hit:
            fixed["vel"][step, CHICO] = int(velocity_to_class(np.array([hit["velocity"]]))[0])
            fixed["art"][step, CHICO] = articulation_id("chico", CHICO_ARTICULATION[step_in_bar])
        else:
            fixed["vel"][step, CHICO] = 0
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


def allowed_velocity_mask(steps: int, structure: Sequence[str]) -> np.ndarray:
    """Hard musical rules as an allowed-class mask (steps, 3, N_VEL).

    Kept deliberately small: everything else is the model's decision.
      - piano must sound on beat 1 of every bar (it anchors the cuerda);
      - in llamada_repique / cierre bars the repique must play the call
        accents (no silence or ghost strokes there); how it fills the rest
        of the bar is up to the model.
    """
    allowed = np.ones((steps, N_INSTR, N_VEL), dtype=bool)
    for bar_index, section in enumerate(structure):
        offset = bar_index * STEPS_PER_BAR
        if offset >= steps:
            break
        allowed[offset, PIANO, 0] = False
        if section in ("llamada_repique", "cierre"):
            for step_in_bar, _ in REPIQUE_CALL:
                if offset + step_in_bar < steps:
                    allowed[offset + step_in_bar, REPIQUE, :2] = False
    return allowed
