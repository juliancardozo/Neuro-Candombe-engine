import numpy as np

from candombe_rules import CHICO_TEMPLATE, REPIQUE_CALL, empty_fixed, fix_from_piece, render_piece
from generate import generate_piece
from representation import N_ART, N_OFF, STEPS_PER_BAR, default_structure

CHICO, REPIQUE, PIANO = range(3)


def test_chico_lock_and_rules_hold(tiny_model):
    structure = ["intro", "llamada_repique", "llamada_piano", "cierre"]
    piece = generate_piece(tiny_model, bars=4, structure=structure, seed=3)
    chico_hits = np.array([1 if hit else 0 for hit in CHICO_TEMPLATE] * 4)
    np.testing.assert_array_equal(piece.vel[:, CHICO] > 0, chico_hits.astype(bool))
    for bar, section in enumerate(structure):
        assert piece.vel[bar * STEPS_PER_BAR, PIANO] > 0
        if section in ("llamada_repique", "cierre"):
            for step_in_bar, _ in REPIQUE_CALL:
                assert piece.vel[bar * STEPS_PER_BAR + step_in_bar, REPIQUE] >= 2


def test_outputs_are_valid_labels(tiny_model):
    piece = generate_piece(tiny_model, bars=4, seed=1, strict_rules=False, chico_locked=False)
    hits = piece.vel > 0
    assert ((piece.off[hits] >= 0) & (piece.off[hits] < N_OFF)).all()
    for inst in range(3):
        arts = piece.art[hits[:, inst], inst]
        assert ((arts >= 0) & (arts < N_ART[inst])).all()
    assert (piece.off[~hits] == -1).all()


def test_fixed_tracks_are_preserved_for_infilling(tiny_model, rng):
    reference = render_piece(rng, default_structure(4))
    fixed = fix_from_piece(empty_fixed(reference.steps), reference, ["piano"])
    piece = generate_piece(tiny_model, bars=4, fixed=fixed, chico_locked=True, seed=5)
    np.testing.assert_array_equal(piece.vel[:, PIANO], reference.vel[:, PIANO])
    np.testing.assert_array_equal(piece.off[:, PIANO], reference.off[:, PIANO])


def test_long_pieces_use_sliding_windows(tiny_model):
    piece = generate_piece(tiny_model, bars=10, seed=2)  # model window is 4 bars
    assert piece.vel.shape == (10 * STEPS_PER_BAR, 3)
    assert (piece.vel[:, CHICO] > 0).sum() == 10 * 12


def test_seed_makes_generation_reproducible(tiny_model):
    a = generate_piece(tiny_model, bars=4, seed=11)
    b = generate_piece(tiny_model, bars=4, seed=11)
    np.testing.assert_array_equal(a.vel, b.vel)
    np.testing.assert_array_equal(a.off, b.off)
