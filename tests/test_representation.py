import numpy as np

from candombe_rules import random_structure, render_piece
from microtiming import DEFAULT_PROFILE, estimate_profile
from representation import (
    N_OFF,
    OFFSET_BIN_WIDTH,
    VEL_CENTERS,
    bin_to_offset,
    class_to_velocity,
    load_piece,
    offset_to_bin,
    save_piece,
    velocity_to_class,
)


def test_velocity_classes_round_trip():
    assert velocity_to_class(np.array([0]))[0] == 0
    for vel_class, center in enumerate(VEL_CENTERS, start=1):
        assert velocity_to_class(np.array([center]))[0] == vel_class
        assert class_to_velocity(vel_class) == center


def test_offset_bins_clip_and_unknown():
    bins = offset_to_bin(np.array([0.0, OFFSET_BIN_WIDTH, -10.0, 10.0, np.nan]))
    assert bins.tolist() == [N_OFF // 2, N_OFF // 2 + 1, 0, N_OFF - 1, -1]
    assert bin_to_offset(N_OFF // 2) == 0.0


def test_legacy_npz_still_loads(tmp_path):
    path = tmp_path / "legacy.npz"
    data = np.zeros((70, 3), dtype=np.uint8)  # 4 bars + a partial bar
    data[0, 2] = 120
    np.savez_compressed(path, data=data)
    piece = load_piece(str(path))
    assert piece.steps == 64
    assert piece.vel[0, 2] == 4
    assert (piece.off == -1).all() and (piece.art == -1).all()


def test_save_load_keeps_microtiming_articulation_sections(tmp_path, rng):
    structure = random_structure(rng, 8)
    piece = render_piece(rng, structure, tempo=104.0)
    velocities = np.zeros(piece.vel.shape, dtype=np.int32)
    for value, center in enumerate(VEL_CENTERS, start=1):
        velocities[piece.vel == value] = center
    offsets = np.where(piece.off >= 0, np.vectorize(bin_to_offset)(piece.off), np.nan)
    path = tmp_path / "piece.npz"
    save_piece(str(path), velocities, offsets, piece.art, structure, tempo=104.0, style="ansina")

    loaded = load_piece(str(path))
    np.testing.assert_array_equal(loaded.vel, piece.vel)
    np.testing.assert_array_equal(loaded.off, piece.off)
    np.testing.assert_array_equal(loaded.art, piece.art)
    np.testing.assert_array_equal(loaded.section, piece.section)
    assert loaded.tempo == 104.0 and loaded.style == "ansina"


def test_rules_have_labels_on_every_hit(rng):
    piece = render_piece(rng, random_structure(rng, 12))
    hits = piece.vel > 0
    assert (piece.off[hits] >= 0).all()
    assert (piece.art[hits] >= 0).all()
    assert (piece.off[~hits] == -1).all()


def test_estimated_profile_recovers_rule_microtiming(rng):
    pieces = [render_piece(rng, random_structure(rng, 16)) for _ in range(40)]
    profile = estimate_profile(pieces)
    for instrument, cells in profile.items():
        for (mean, _), (true_mean, true_std) in zip(cells, DEFAULT_PROFILE[instrument]):
            # Quantization to offset bins limits how close the estimate can be.
            assert abs(mean - true_mean) < OFFSET_BIN_WIDTH / 2 + 2 * true_std
