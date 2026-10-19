"""The toques madre of Montevideo candombe, as documented in the literature.

Every pattern below is a transcription from a published source, written on a
16-step bar (4 beats x 4 sixteenths, step 0 = downbeat). Each event is
(step, articulation, level). Sources:

  [J22] Jure & Rocamora (2022), "Microtiming in the rhythmic structure of
        Candombe drumming" (luisjure.org/publications/2022-11_Micro).
        Figs. 4, 5, 7 and the chico microtiming measurements (sec. 3-4).
  [J18] Jure & Rocamora (2018), "Subir la llamada: negotiating tempo and
        dynamics in Uruguayan Candombe drumming" (tempos, fig. 3).
  [J16] Jure & Rocamora (2016), "Microtiming in the rhythmic structure of
        Candombe drumming patterns" (repique timing).
  [N15] Nunes, Rocamora, Jure & Biscainho (2015), ISMIR (fig. 2, piano
        base as binary pattern).
  [S13] Sutton (2013), "The Candombe Drumming of Uruguay", PhD dissertation,
        University of Miami. Musical examples 4.4-4.7 and 5.1-5.4, including
        Luis Ferreira's base piano transcriptions ("Los tambores del
        candombe", 1997) and a Gaboto pattern by Aquiles Pintos.

Notation was read from the published scores: notes below the line are hand
strokes and above are stick strokes; "masa" is the piano's heavy hand stroke
pressed into the head (often together with a muted stick), "apagado" a muted
stick stroke, "abierto" an open stick stroke, "madera" the stick on the shell.
Transcriptions are basic forms: players add ghost notes and variants [J22].
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

STEPS = 16

# Dynamic levels -> MIDI-like velocity.
LEVELS = {"ghost": 30, "light": 55, "normal": 82, "accent": 112}

Event = Tuple[int, str, str]  # (step, articulation, level)


@dataclass(frozen=True)
class Pattern:
    name: str
    source: str
    events: Tuple[Event, ...]
    weight: float = 1.0


def _p(name: str, source: str, events, weight: float = 1.0) -> Pattern:
    return Pattern(name, source, tuple((s, a, lvl) for s, a, lvl in events), weight)


# --- shared patterns -------------------------------------------------------------------

# Madera / clave: 3+3+4+2+4, played by every drum to open the llamada [N15, J22, S13].
CLAVE = _p("Madera (clave)", "N15 fig.2, J22 fig.4", [
    (0, "madera", "accent"), (3, "madera", "normal"), (6, "madera", "normal"),
    (10, "madera", "normal"), (12, "madera", "accent"),
])

# Chico: no stroke on the beat, accented hand on the 2nd sixteenth [N15, J22 fig.5].
CHICO_DE_DOS = _p("Chico de dos (liso)", "J22 fig.5", [
    (b + 1, "mano", "accent") for b in range(0, 16, 4)
] + [(b + k, "palo", "normal") for b in range(0, 16, 4) for k in (2, 3)])
# Chico de tres (repicado): a third stick stroke on the beat; common at slow
# tempos and in Cuareim [J22 fig.5, S13 ex.4.3 and ch.5].
CHICO_DE_TRES = _p("Chico de tres (repicado)", "J22 fig.5", [
    (b, "palo", "normal") for b in range(0, 16, 4)
] + list(CHICO_DE_DOS.events))

# Repique básico: stick 0,3,5,8,11,13 / hand 1,4,7,9,12,15 [J22 fig.4, N15 fig.2].
REPIQUE_BASICO = _p("Repique básico (corrido)", "J22 fig.4", [
    (0, "palo", "normal"), (1, "mano", "normal"), (3, "palo", "normal"),
    (4, "mano", "normal"), (5, "palo", "normal"), (7, "mano", "normal"),
    (8, "palo", "normal"), (9, "mano", "normal"), (11, "palo", "normal"),
    (12, "mano", "normal"), (13, "palo", "normal"), (15, "mano", "normal"),
])

# Repique repicado: phrases start with a hand slap on the 2nd sixteenth and the
# hand slaps are always stronger than the stick [S13 ex.4.6].
REPIQUE_LLAMADA = _p("Llamada de repique (introducción)", "S13 ex.4.6", [
    (1, "mano", "accent"), (2, "palo", "normal"), (3, "palo", "normal"), (4, "mano", "accent"),
    (9, "mano", "accent"), (10, "palo", "normal"), (11, "palo", "normal"), (12, "mano", "accent"),
])
REPIQUE_REPICADO = _p("Repicado de repique", "S13 ex.4.6", [
    (1, "mano", "accent"), (2, "palo", "normal"), (3, "palo", "normal"),
    (4, "mano", "accent"), (5, "palo", "normal"), (7, "mano", "accent"),
    (8, "palo", "normal"), (9, "mano", "accent"), (11, "palo", "normal"),
    (12, "mano", "accent"), (13, "palo", "normal"), (15, "mano", "accent"),
])

# Closing signal of the repique, with rim-shots; the cuerda plays softer so it is
# heard, and everyone stops on the downbeat after it [S13 ex.4.7].
REPIQUE_CIERRE = (
    _p("Cierre 1", "S13 ex.4.7", [(0, "rimshot", "accent"), (6, "rimshot", "accent"), (12, "rimshot", "accent")]),
    _p("Cierre 2", "S13 ex.4.7", [
        (1, "mano", "normal"), (3, "rimshot", "accent"), (4, "mano", "normal"), (5, "rimshot", "accent"),
        (7, "mano", "normal"), (8, "rimshot", "accent"), (9, "mano", "normal"), (11, "rimshot", "accent"),
        (12, "mano", "normal"), (14, "rimshot", "accent"),
    ]),
)

# Piano repicado: ornamented repique-like cycles, 1-2 bars, answered among
# pianos and with the repiques [J22 fig.7, S13 ex.4.5].
PIANO_REPICADOS = (
    _p("Repicado de piano (Ansina) 1", "J22 fig.7", [
        (0, "masa", "normal"), (2, "abierto", "accent"), (4, "mano", "normal"), (5, "abierto", "normal"),
        (7, "abierto", "normal"), (8, "mano", "normal"), (9, "abierto", "normal"), (11, "abierto", "normal"),
        (12, "abierto", "normal"), (13, "mano", "normal"), (14, "abierto", "normal"),
    ]),
    _p("Repicado de piano (Ansina) 2", "J22 fig.7", [
        (0, "abierto", "normal"), (1, "mano", "normal"), (3, "abierto", "normal"), (4, "mano", "normal"),
        (5, "abierto", "normal"), (7, "abierto", "normal"), (8, "abierto", "normal"), (9, "mano", "normal"),
        (11, "abierto", "normal"), (12, "abierto", "normal"), (13, "mano", "normal"), (14, "abierto", "normal"),
    ]),
    _p("Repicado de piano común", "S13 ex.4.5 (Ferreira)", [
        (1, "mano", "accent"), (2, "abierto", "normal"), (3, "abierto", "normal"), (4, "mano", "normal"),
        (5, "abierto", "normal"), (7, "mano", "normal"), (8, "abierto", "normal"), (9, "mano", "normal"),
        (11, "abierto", "normal"), (12, "masa", "normal"), (14, "abierto", "accent"),
    ]),
)


# --- the three toques madre -------------------------------------------------------------

@dataclass(frozen=True)
class Toque:
    key: str
    label: str
    barrio: str
    description: str
    tempo: int  # typical tempo once the llamada is up
    tempo_range: Tuple[int, int]
    tempo_wobble: float  # bar-to-bar tempo fluctuation (fraction) once up to speed
    chico_de_tres_always: bool
    piano_bases: Tuple[Pattern, ...]
    piano_change: float  # probability that a piano changes base pattern on a new bar
    piano_repicado: float  # probability of a piano repicado cycle in a base bar
    microtiming_p1: float  # chico P1 location (fraction of beat) [J22]


TOQUES: Dict[str, Toque] = {
    "cuareim": Toque(
        key="cuareim",
        label="Cuareim",
        barrio="Barrio Sur",
        description=(
            "El más pausado y asentado. El piano es el más simple y casi no varía; "
            "el chico toca de tres y llevan el acento el chico y el repique."
        ),
        tempo=124,
        tempo_range=(100, 140),
        tempo_wobble=0.004,
        chico_de_tres_always=True,
        piano_bases=(
            _p("Cuareim moderno 1", "S13 ex.5.2-1", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "abierto", "light"), (5, "masa", "light"),
                (8, "masa", "normal"), (11, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=3),
            _p("Cuareim moderno 2", "S13 ex.5.2-2", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "masa", "light"), (5, "apagado", "normal"),
                (8, "masa", "normal"), (11, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=2),
            _p("Cuareim (Fernando 'Lobo' Núñez)", "S13 ex.5.1 (Ferreira)", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (5, "masa", "light"),
                (8, "masa", "normal"), (11, "abierto", "normal"), (12, "masa", "normal"), (14, "apagado", "light"),
            ], weight=2),
            _p("Cuareim antiguo", "S13 ex.5.2-3", [
                (0, "masa", "normal"), (3, "abierto", "normal"), (4, "masa", "normal"),
                (8, "masa", "normal"), (11, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=0.5),
        ),
        piano_change=0.08,
        piano_repicado=0.03,
        microtiming_p1=0.49,
    ),
    "ansina": Toque(
        key="ansina",
        label="Ansina",
        barrio="Palermo",
        description=(
            "Rápido y agresivo. Los pianos dialogan entre sí y con los repiques, "
            "con más golpes en el tercer tiempo; el chico toca de dos."
        ),
        tempo=132,
        tempo_range=(110, 145),
        tempo_wobble=0.006,
        chico_de_tres_always=False,
        piano_bases=(
            _p("Ansina 1 (el más común desde los 60)", "S13 ex.5.3-1", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "abierto", "light"), (5, "masa", "light"),
                (8, "abierto", "normal"), (9, "mano", "normal"), (10, "abierto", "normal"), (11, "abierto", "normal"),
                (12, "masa", "normal"), (14, "abierto", "accent"),
            ], weight=3),
            _p("Ansina 2", "S13 ex.5.3-2", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "masa", "light"), (5, "apagado", "normal"),
                (8, "masa", "normal"), (10, "abierto", "normal"), (11, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=2),
            _p("Ansina 3", "S13 ex.5.3-3", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "abierto", "light"), (5, "masa", "light"),
                (8, "abierto", "normal"), (9, "mano", "normal"), (10, "abierto", "accent"), (12, "masa", "normal"),
            ], weight=2),
            _p("Ansina (Ferreira)", "S13 ex.5.1 (Ferreira)", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "masa", "light"), (5, "abierto", "accent"),
                (8, "masa", "normal"), (9, "abierto", "accent"), (10, "mano", "normal"), (11, "abierto", "normal"),
                (12, "masa", "normal"),
            ], weight=1.5),
            _p("Ansina base A", "J22 fig.7", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "masa", "light"), (5, "apagado", "normal"),
                (8, "masa", "normal"), (10, "mano", "ghost"), (11, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=1.5),
            _p("Ansina base B", "J22 fig.7", [
                (0, "masa", "normal"), (2, "abierto", "normal"), (3, "abierto", "accent"), (4, "masa", "light"),
                (5, "abierto", "accent"), (7, "abierto", "normal"), (8, "mano", "normal"), (9, "abierto", "normal"),
                (10, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=1.5),
            _p("Ansina de los 50", "S13 ex.5.3-4", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "masa", "light"), (5, "apagado", "normal"),
                (8, "masa", "normal"), (9, "abierto", "accent"), (12, "masa", "normal"),
            ], weight=0.5),
        ),
        piano_change=0.25,
        piano_repicado=0.1,
        microtiming_p1=0.465,
    ),
    "gaboto": Toque(
        key="gaboto",
        label="Gaboto",
        barrio="Cordón",
        description=(
            "Más rápido que Ansina pero menos estable. Piano muy presente que alterna "
            "figuras de Cuareim y de Ansina; la masa cae siempre en la segunda "
            "semicorchea del segundo tiempo."
        ),
        tempo=136,
        tempo_range=(115, 150),
        tempo_wobble=0.012,
        chico_de_tres_always=False,
        piano_bases=(
            _p("Gaboto (Aquiles Pintos) 1", "S13 ex.5.4", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "abierto", "light"), (5, "masa", "normal"),
                (8, "masa", "normal"), (11, "abierto", "normal"), (12, "masa", "normal"),
            ], weight=2),
            _p("Gaboto (Aquiles Pintos) 2", "S13 ex.5.4", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "abierto", "light"), (5, "masa", "normal"),
                (8, "masa", "normal"), (10, "abierto", "accent"), (12, "masa", "normal"),
            ], weight=2),
            _p("Gaboto (Aquiles Pintos) 3", "S13 ex.5.4", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (4, "abierto", "light"), (5, "masa", "normal"),
                (8, "abierto", "normal"), (9, "mano", "normal"), (10, "abierto", "accent"), (12, "masa", "normal"),
            ], weight=2),
            _p("Gaboto (Ferreira)", "S13 ex.5.1 (Ferreira)", [
                (0, "masa", "normal"), (3, "abierto", "accent"), (5, "masa", "normal"),
                (8, "masa", "normal"), (9, "masa", "ghost"), (10, "masa", "ghost"), (11, "abierto", "normal"),
                (12, "masa", "normal"), (14, "abierto", "accent"),
            ], weight=1.5),
        ),
        piano_change=0.6,
        piano_repicado=0.12,
        microtiming_p1=0.465,
    ),
}

ALIASES = {"cordon": "gaboto", "cordón": "gaboto", "sur": "cuareim", "palermo": "ansina"}
DEFAULT_TOQUE = "ansina"


def get_toque(key: str) -> Toque:
    key = (key or DEFAULT_TOQUE).lower()
    return TOQUES[ALIASES.get(key, key)]


# --- tempo, dynamics and chico pattern along a llamada ------------------------------------

def tempo_curve(toque: Toque, bars: int, target: float, subir: bool, rng=None) -> List[float]:
    """Tempo per bar. With `subir`, the llamada starts ~12% slower and speeds up to
    the target over the first third, then holds with small fluctuations [J18]."""
    import numpy as np

    rng = rng or np.random.default_rng(0)
    curve = []
    start = target * 0.88 if subir else target
    ramp_bars = max(1, bars // 3)
    for bar in range(bars):
        base = start + (target - start) * min(1.0, bar / ramp_bars)
        wobble = rng.normal(0.0, toque.tempo_wobble) if bar >= ramp_bars or not subir else 0.0
        curve.append(round(float(base * (1.0 + wobble)), 2))
    return curve


def dynamics_curve(structure: List[str], subir: bool) -> List[float]:
    """Loudness per bar: the energy rises with the tempo [J18]; the cuerda plays
    softer under the repique's closing signal so it can be heard [S13]."""
    bars = len(structure)
    ramp_bars = max(1, bars // 3)
    out = []
    for bar, section in enumerate(structure):
        level = 0.82 + 0.18 * min(1.0, bar / ramp_bars) if subir else 1.0
        if section == "cierre":
            level *= 0.75
        out.append(round(level, 3))
    return out


def chico_pattern(toque: Toque, tempo: float) -> Pattern:
    """Cuareim chicos play de tres; elsewhere de tres only at slow tempos and
    de dos once the llamada is up to speed [J22, S13]."""
    return CHICO_DE_TRES if toque.chico_de_tres_always or tempo < 115 else CHICO_DE_DOS


# --- microtiming ----------------------------------------------------------------------------

def chico_p2(tempo: float) -> float:
    """Location (fraction of beat) of the last chico stroke before the beat.

    Measured 0.73 at ~90 BPM down to 0.69 at ~135 BPM: the faster, the closer to a
    triplet; 0.70-0.71 most frequent [J22 sec.3-4]."""
    t = min(1.0, max(0.0, (tempo - 90.0) / 45.0))
    return 0.73 - 0.04 * t


def groove_offsets(toque: Toque, instrument: str, tempo: float) -> Tuple[float, float, float, float]:
    """Mean offset of each sixteenth of the beat, as a fraction of a sixteenth.

    The chico values are the measurements of [J22]: the hand (M) on or very near
    the 2nd sixteenth, P1 and especially P2 early. The other drums lock into the
    chico [J22 fig.8]: the piano's accented open stroke coincides with the chico's
    P2, and the last stroke of the repique básico is near a ternary division [J16]."""
    p2 = (chico_p2(tempo) - 0.75) / 0.25
    p1 = (toque.microtiming_p1 - 0.50) / 0.25
    m = -0.02 if toque.key != "cuareim" else 0.0
    if instrument == "repique":
        # Repique básico leans to a triplet: 2nd sixteenth slightly late, last one early.
        return (0.0, 0.06, p1 * 0.5, min(p2, -0.24))
    if instrument == "piano":
        return (0.0, m, p1 * 0.5, p2)
    return (0.0, m, p1, p2)


TIMING_SPREAD = {"chico": 0.035, "repique": 0.06, "piano": 0.045}  # std, fraction of a sixteenth
