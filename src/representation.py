"""Shared symbolic representation for candombe pieces.

A piece is a 16th-note grid of shape (steps, 3) for chico, repique and piano.
Each cell carries:
  - a velocity class: 0 = silence, 1..VEL_BUCKETS = hit intensity,
  - a microtiming offset, as a fraction of one 16th step, quantized into bins,
  - an articulation id from that instrument's vocabulary.
Offsets and articulations may be unknown (-1), e.g. for MIDI or audio sources
that do not provide them; the training loss ignores unknown labels.

On disk a piece is an .npz. The legacy format only had `data` (steps, 3) with
MIDI velocities 0..127, so old files still load; the extra keys are optional.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

INSTRUMENTS = ("chico", "repique", "piano")
N_INSTR = len(INSTRUMENTS)
STEPS_PER_BAR = 16
STEPS_PER_BEAT = 4

VEL_BUCKETS = 4
N_VEL = VEL_BUCKETS + 1  # + silence
_VEL_BINS = np.linspace(1, 127, VEL_BUCKETS + 1)
VEL_CENTERS = (22, 52, 82, 112)

# Microtiming as a fraction of a 16th step, so it is tempo independent.
# 0.03 of a step is ~4 ms at 110 BPM, fine enough for the chico's few-ms shifts.
OFFSET_BIN_WIDTH = 0.03
N_OFF = 25
OFFSET_CENTERS = (np.arange(N_OFF) - N_OFF // 2) * OFFSET_BIN_WIDTH  # -0.36 .. 0.36

# Strokes as named by players and in the literature (see toques.py):
# piano "abierto" = open stick, "apagado" = muted stick, "masa" = heavy hand
# pressed into the head, "mano" = open hand; "madera" = stick on the shell.
ARTICULATIONS = {
    "chico": ("mano", "palo", "madera"),
    "repique": ("palo", "mano", "madera", "rimshot"),
    "piano": ("abierto", "apagado", "masa", "mano", "madera"),
}
N_ART = tuple(len(ARTICULATIONS[name]) for name in INSTRUMENTS)

SECTIONS = ("base", "intro", "llamada_repique", "llamada_piano", "cierre", "final")
SECTION_UNKNOWN = len(SECTIONS)

STYLES = ("cuareim", "ansina", "gaboto")  # the toques madre, see toques.py
STYLE_UNKNOWN = len(STYLES)

_TEMPO_EDGES = (80, 95, 110, 125, 140)
N_TEMPO = len(_TEMPO_EDGES) + 1
TEMPO_UNKNOWN = N_TEMPO

# Density = fraction of repique + piano cells with a hit. The chico is excluded
# because its cadence is constant and would dominate the measure.
_DENSITY_EDGES = (0.18, 0.26, 0.34, 0.42, 0.50)
N_DENSITY = len(_DENSITY_EDGES) + 1
DENSITY_UNKNOWN = N_DENSITY

UNKNOWN = -1


def velocity_to_class(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.int64)
    classes = np.zeros_like(velocity)
    hit = velocity > 0
    if np.any(hit):
        classes[hit] = np.clip(np.digitize(velocity[hit], _VEL_BINS, right=True), 1, VEL_BUCKETS)
    return classes


def class_to_velocity(vel_class: int) -> int:
    return 0 if vel_class <= 0 else VEL_CENTERS[int(vel_class) - 1]


def offset_to_bin(offset_frac: np.ndarray) -> np.ndarray:
    """Fractional offsets (NaN = unknown) to bins (-1 = unknown)."""
    offset_frac = np.asarray(offset_frac, dtype=np.float64)
    bins = np.full(offset_frac.shape, UNKNOWN, dtype=np.int64)
    known = np.isfinite(offset_frac)
    bins[known] = np.clip(np.round(offset_frac[known] / OFFSET_BIN_WIDTH) + N_OFF // 2, 0, N_OFF - 1)
    return bins


def bin_to_offset(offset_bin: int) -> float:
    return 0.0 if offset_bin < 0 else float(OFFSET_CENTERS[int(offset_bin)])


def articulation_id(instrument: str, name: str) -> int:
    names = ARTICULATIONS[instrument]
    return names.index(name) if name in names else UNKNOWN


def tempo_bucket(tempo: Optional[float]) -> int:
    if tempo is None or not np.isfinite(tempo) or tempo <= 0:
        return TEMPO_UNKNOWN
    return int(np.digitize(tempo, _TEMPO_EDGES))


def hit_rate(vel: np.ndarray) -> float:
    """Density measure of a velocity-class grid (see _DENSITY_EDGES)."""
    if vel.shape[0] == 0:
        return 0.0
    return float((vel[:, 1:] > 0).mean())


def density_bucket(rate: Optional[float]) -> int:
    if rate is None or not np.isfinite(rate):
        return DENSITY_UNKNOWN
    return int(np.digitize(rate, _DENSITY_EDGES))


def density_control_to_rate(density: float) -> float:
    """Map the UI/API `density` knob (0..1) to a target hit rate."""
    return 0.12 + 0.45 * float(np.clip(density, 0.0, 1.0))


def section_id(name: Optional[str]) -> int:
    return SECTIONS.index(name) if name in SECTIONS else SECTION_UNKNOWN


def style_id(name: Optional[str]) -> int:
    name = {"cordon": "gaboto", "cordón": "gaboto"}.get(name, name)
    return STYLES.index(name) if name in STYLES else STYLE_UNKNOWN


def default_structure(total_bars: int) -> list:
    """Section per bar when none is requested, shaped like a llamada:
    every drum marks the clave on the madera, a repique call starts the toque,
    the base runs with a repique call closing every 8-bar phrase (and a piano
    call in longer pieces), the repique gives the closing signal over two bars
    and everyone stops together on the last downbeat."""
    if total_bars < 4:
        return ["base"] * total_bars
    out = ["base"] * total_bars
    out[0] = "intro"
    out[1] = "llamada_repique"
    for phrase_end in range(9, total_bars - 3, 8):
        out[phrase_end] = "llamada_repique"
    if total_bars >= 12:
        middle = total_bars // 2 + 2
        if out[middle] == "base":
            out[middle] = "llamada_piano"
    if total_bars >= 6:
        out[-3] = "cierre"
    out[-2] = "cierre"
    out[-1] = "final"
    return out


@dataclass
class Piece:
    vel: np.ndarray  # (S, 3) int, 0..VEL_BUCKETS
    off: np.ndarray  # (S, 3) int, offset bin or -1
    art: np.ndarray  # (S, 3) int, articulation id or -1
    section: np.ndarray  # (S,) int section id
    tempo: Optional[float] = None
    style: Optional[str] = None
    source: str = field(default="")

    @property
    def steps(self) -> int:
        return int(self.vel.shape[0])

    @property
    def bars(self) -> int:
        return self.steps // STEPS_PER_BAR


def make_piece(
    velocities: np.ndarray,
    offsets_frac: Optional[np.ndarray] = None,
    articulation: Optional[np.ndarray] = None,
    sections: Optional[Sequence[str]] = None,
    tempo: Optional[float] = None,
    style: Optional[str] = None,
    source: str = "",
) -> Piece:
    velocities = np.asarray(velocities)
    steps = velocities.shape[0]
    vel = velocity_to_class(velocities)
    off = offset_to_bin(offsets_frac) if offsets_frac is not None else np.full((steps, N_INSTR), UNKNOWN, dtype=np.int64)
    art = np.asarray(articulation, dtype=np.int64) if articulation is not None else np.full((steps, N_INSTR), UNKNOWN, dtype=np.int64)
    off = np.where(vel > 0, off, UNKNOWN)
    art = np.where(vel > 0, art, UNKNOWN)

    section = np.full(steps, SECTION_UNKNOWN, dtype=np.int64)
    if sections is not None:
        for bar_index, name in enumerate(sections):
            section[bar_index * STEPS_PER_BAR : (bar_index + 1) * STEPS_PER_BAR] = section_id(str(name))
    return Piece(vel=vel, off=off, art=art, section=section, tempo=tempo, style=style, source=source)


def save_piece(
    path: str,
    velocities: np.ndarray,
    offsets_frac: Optional[np.ndarray] = None,
    articulation: Optional[np.ndarray] = None,
    sections: Optional[Sequence[str]] = None,
    tempo: Optional[float] = None,
    style: Optional[str] = None,
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"data": np.asarray(velocities, dtype=np.uint8)}
    if offsets_frac is not None:
        payload["offsets_frac"] = np.asarray(offsets_frac, dtype=np.float32)
    if articulation is not None:
        payload["articulation"] = np.asarray(articulation, dtype=np.int8)
    if sections is not None:
        payload["sections"] = np.asarray(list(sections), dtype="U32")
    if tempo is not None:
        payload["tempo"] = np.float32(tempo)
    if style is not None:
        payload["style"] = np.asarray(style, dtype="U32")
    np.savez_compressed(path, **payload)


def load_piece(path: str) -> Piece:
    with np.load(path, allow_pickle=False) as npz:
        velocities = npz["data"]
        offsets = npz["offsets_frac"] if "offsets_frac" in npz else None
        articulation = npz["articulation"] if "articulation" in npz else None
        sections = [str(s) for s in npz["sections"]] if "sections" in npz else None
        tempo = float(npz["tempo"]) if "tempo" in npz else None
        style = str(npz["style"]) if "style" in npz else None
    # Drop a trailing partial bar so windows always align to bar lines.
    steps = (velocities.shape[0] // STEPS_PER_BAR) * STEPS_PER_BAR
    velocities = velocities[:steps]
    offsets = None if offsets is None else offsets[:steps]
    articulation = None if articulation is None else articulation[:steps]
    return make_piece(velocities, offsets, articulation, sections, tempo, style, source=path)


def piece_to_events(piece: Piece, step_duration: float, sections: Optional[Sequence[str]] = None) -> list:
    """Events for the web player. `offset_ms` carries the microtiming."""
    events = []
    for step in range(piece.steps):
        bar_index = step // STEPS_PER_BAR
        section_name = sections[bar_index] if sections is not None and bar_index < len(sections) else None
        if section_name is None:
            sid = int(piece.section[step])
            section_name = SECTIONS[sid] if sid < len(SECTIONS) else "base"
        for inst_idx, instrument in enumerate(INSTRUMENTS):
            vel_class = int(piece.vel[step, inst_idx])
            if vel_class <= 0:
                continue
            art = int(piece.art[step, inst_idx])
            articulation = ARTICULATIONS[instrument][art] if art >= 0 else ARTICULATIONS[instrument][0]
            offset_frac = bin_to_offset(int(piece.off[step, inst_idx]))
            events.append(
                {
                    "time": step * step_duration,
                    "instrument": instrument,
                    "velocity": class_to_velocity(vel_class),
                    "offset_ms": round(offset_frac * step_duration * 1000.0, 2),
                    "step": step,
                    "section": section_name,
                    "articulation": articulation,
                    "hit_type": articulation,
                }
            )
    return events
