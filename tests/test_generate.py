import numpy as np
import pytest

from candombe_rules import empty_fixed, fix_from_piece, render_piece
from generate import generate_piece
from representation import ARTICULATIONS, N_ART, N_OFF, STEPS_PER_BAR, default_structure
from toques import CHICO_DE_DOS, CHICO_DE_TRES, CLAVE, REPIQUE_CIERRE, REPIQUE_LLAMADA

CHICO, REPIQUE, PIANO = range(3)


def onsets(pattern):
    grid = np.zeros(STEPS_PER_BAR, dtype=bool)
    grid[[step for step, _, _ in pattern.events]] = True
    return grid


def bar(piece, index, inst):
    return piece.vel[index * STEPS_PER_BAR : (index + 1) * STEPS_PER_BAR, inst] > 0


def art_names(piece, index, inst):
    names = ARTICULATIONS[("chico", "repique", "piano")[inst]]
    cells = slice(index * STEPS_PER_BAR, (index + 1) * STEPS_PER_BAR)
    return {names[a] for v, a in zip(piece.vel[cells, inst], piece.art[cells, inst]) if v > 0}


@pytest.mark.parametrize("style", ["cuareim", "ansina", "gaboto"])
def test_llamada_form_and_toque_invariants(tiny_model, style):
    structure = default_structure(8)  # intro, llamada, base x3, cierre x2, final
    piece = generate_piece(tiny_model, bars=8, structure=structure, style=style, tempo=130, seed=3)

    for inst in range(3):  # everyone opens on the madera, and closes together
        np.testing.assert_array_equal(bar(piece, 0, inst), onsets(CLAVE))
        assert art_names(piece, 0, inst) == {"madera"}
        np.testing.assert_array_equal(bar(piece, 7, inst), np.eye(STEPS_PER_BAR, dtype=bool)[0])
    np.testing.assert_array_equal(bar(piece, 1, REPIQUE), onsets(REPIQUE_LLAMADA))
    np.testing.assert_array_equal(bar(piece, 5, REPIQUE), onsets(REPIQUE_CIERRE[0]))
    np.testing.assert_array_equal(bar(piece, 6, REPIQUE), onsets(REPIQUE_CIERRE[1]))

    chico = onsets(CHICO_DE_TRES if style == "cuareim" else CHICO_DE_DOS)
    for index in range(1, 7):
        np.testing.assert_array_equal(bar(piece, index, CHICO), chico)
    for index in (2, 3, 4):
        start = index * STEPS_PER_BAR
        assert piece.art[start, PIANO] == ARTICULATIONS["piano"].index("masa")
        assert piece.art[start + 3, PIANO] == ARTICULATIONS["piano"].index("abierto")
        assert piece.vel[start + 3, PIANO] >= 3
        if style == "gaboto":
            assert piece.art[start + 5, PIANO] == ARTICULATIONS["piano"].index("masa")


def test_slow_bars_switch_chico_de_tres(tiny_model):
    tempos = [100.0] * 4 + [132.0] * 4
    piece = generate_piece(tiny_model, bars=8, structure=["base"] * 8, style="ansina", tempos=tempos, seed=1)
    np.testing.assert_array_equal(bar(piece, 1, CHICO), onsets(CHICO_DE_TRES))
    np.testing.assert_array_equal(bar(piece, 6, CHICO), onsets(CHICO_DE_DOS))


def test_outputs_are_valid_labels(tiny_model):
    piece = generate_piece(tiny_model, bars=4, seed=1, strict_rules=False, chico_locked=False)
    hits = piece.vel > 0
    assert ((piece.off[hits] >= 0) & (piece.off[hits] < N_OFF)).all()
    for inst in range(3):
        arts = piece.art[hits[:, inst], inst]
        assert ((arts >= 0) & (arts < N_ART[inst])).all()
    assert (piece.off[~hits] == -1).all()


def test_fixed_tracks_win_over_the_form(tiny_model, rng):
    reference = render_piece(rng, default_structure(8), style="ansina")
    fixed = fix_from_piece(empty_fixed(reference.steps), reference, ["piano"])
    piece = generate_piece(tiny_model, bars=8, fixed=fixed, style="ansina", seed=5)
    np.testing.assert_array_equal(piece.vel[:, PIANO], reference.vel[:, PIANO])
    np.testing.assert_array_equal(piece.off[:, PIANO], reference.off[:, PIANO])


def test_long_pieces_use_sliding_windows(tiny_model):
    piece = generate_piece(tiny_model, bars=10, structure=["base"] * 10, style="ansina", tempo=132, seed=2)  # window is 4 bars
    assert piece.vel.shape == (10 * STEPS_PER_BAR, 3)
    assert (piece.vel[:, CHICO] > 0).sum() == 10 * 12


def test_seed_makes_generation_reproducible(tiny_model):
    a = generate_piece(tiny_model, bars=4, seed=11)
    b = generate_piece(tiny_model, bars=4, seed=11)
    np.testing.assert_array_equal(a.vel, b.vel)
    np.testing.assert_array_equal(a.off, b.off)
