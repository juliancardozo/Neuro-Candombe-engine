"""The documented toques: patterns are well formed and the renderer follows them."""

import numpy as np
import pytest

from candombe_rules import render_piece
from evaluate import style_metrics
from representation import ARTICULATIONS, STEPS_PER_BAR, default_structure
from toques import (
    CHICO_DE_DOS,
    CHICO_DE_TRES,
    CLAVE,
    PIANO_REPICADOS,
    REPIQUE_BASICO,
    REPIQUE_CIERRE,
    REPIQUE_LLAMADA,
    REPIQUE_REPICADO,
    TOQUES,
    chico_p2,
    dynamics_curve,
    get_toque,
    groove_offsets,
    tempo_curve,
)


def all_patterns():
    yield "chico", CHICO_DE_DOS
    yield "chico", CHICO_DE_TRES
    yield "repique", CLAVE
    for pattern in (REPIQUE_BASICO, REPIQUE_LLAMADA, REPIQUE_REPICADO, *REPIQUE_CIERRE):
        yield "repique", pattern
    for pattern in PIANO_REPICADOS:
        yield "piano", pattern
    for toque in TOQUES.values():
        for pattern in toque.piano_bases:
            yield "piano", pattern


@pytest.mark.parametrize("instrument,pattern", list(all_patterns()), ids=lambda x: getattr(x, "name", x))
def test_patterns_are_well_formed(instrument, pattern):
    steps = [step for step, _, _ in pattern.events]
    assert all(0 <= step < STEPS_PER_BAR for step in steps)
    assert len(steps) == len(set(steps)), "one stroke per sixteenth"
    assert all(art in ARTICULATIONS[instrument] for _, art, _ in pattern.events)
    assert pattern.source


def test_documented_invariants():
    # Chico: never on the beat in "de dos", accented hand on the 2nd sixteenth.
    assert {s for s, _, _ in CHICO_DE_DOS.events} == {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15}
    assert all(a == "mano" and lvl == "accent" for s, a, lvl in CHICO_DE_DOS.events if s % 4 == 1)
    # Clave 3+3+4+2+4.
    assert [s for s, _, _ in CLAVE.events] == [0, 3, 6, 10, 12]
    for toque in TOQUES.values():
        for base in toque.piano_bases:
            events = {s: (a, lvl) for s, a, lvl in base.events}
            assert events[0][0] == "masa"
            assert events[3][0] == "abierto"  # the accent anticipating beat two
            if toque.key == "gaboto":
                assert events[5][0] == "masa"
    # Cuareim plays chico de tres; the others only when slow.
    assert get_toque("cuareim").chico_de_tres_always and not get_toque("ansina").chico_de_tres_always
    assert get_toque("cordon").key == "gaboto"
    assert get_toque("cuareim").tempo < get_toque("ansina").tempo < get_toque("gaboto").tempo


def test_measured_microtiming():
    assert chico_p2(90) == pytest.approx(0.73)
    assert chico_p2(135) == pytest.approx(0.69)
    ansina = groove_offsets(get_toque("ansina"), "chico", 132)
    # Last stroke before the beat at ~0.70 of the beat -> 0.2 of a sixteenth early.
    assert -0.24 < ansina[3] < -0.17
    assert ansina[2] < 0  # P1 early too
    assert groove_offsets(get_toque("ansina"), "piano", 132)[3] == pytest.approx(ansina[3])


def test_subir_la_llamada_curves():
    toque = get_toque("ansina")
    curve = tempo_curve(toque, 12, 132, subir=True)
    assert curve[0] == pytest.approx(132 * 0.88)
    assert abs(np.mean(curve[6:]) - 132) < 2
    structure = default_structure(12)
    dynamics = dynamics_curve(structure, subir=True)
    assert dynamics[0] < dynamics[5]
    assert dynamics[structure.index("cierre")] < dynamics[5]


@pytest.mark.parametrize("style", list(TOQUES))
def test_rules_follow_their_toque(style):
    pieces = [render_piece(np.random.default_rng(i), default_structure(16), style=style) for i in range(12)]
    metrics = style_metrics(pieces)
    assert metrics["chico_conformity"] == 1.0
    assert metrics["piano_style_match"] > 0.8
    assert metrics["piano_accent"] > 0.85
    assert metrics["call_adherence"] > 0.9
    assert all(piece.style == style for piece in pieces)
