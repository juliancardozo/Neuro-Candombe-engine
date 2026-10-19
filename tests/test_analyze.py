"""Audio analysis on synthetic recordings with known grid, phase and microtiming."""

import numpy as np
import pytest
import soundfile as sf

from analyze_reference import SR, analyze_reference, estimate_phase, snap, subdivide
from candombe_rules import render_piece
from representation import STEPS_PER_BAR, bin_to_offset

TEMPO = 110.0
STEP = 60.0 / TEMPO / 4
PRE_ROLL_STEPS = 8  # half a bar of silence before the first downbeat


def _burst(kind: str, rng) -> np.ndarray:
    t = np.arange(int(0.12 * SR)) / SR
    if kind == "chico":
        noise = rng.standard_normal(t.size)
        noise = np.diff(noise, prepend=0)  # crude high-pass
        return 0.5 * noise * np.exp(-t * 90)
    if kind == "repique":
        return 0.7 * np.sin(2 * np.pi * 720 * t) * np.exp(-t * 40)
    return 0.9 * np.sin(2 * np.pi * 95 * t) * np.exp(-t * 18)


def _render(piece, rng):
    total = (piece.steps + PRE_ROLL_STEPS + 16) * STEP
    audio = np.zeros(int(total * SR))
    kinds = ("chico", "repique", "piano")
    for step in range(piece.steps):
        for inst, kind in enumerate(kinds):
            vel = piece.vel[step, inst]
            if vel == 0:
                continue
            onset = (step + PRE_ROLL_STEPS + bin_to_offset(int(piece.off[step, inst]))) * STEP
            start = int(onset * SR)
            burst = _burst(kind, rng) * (0.4 + 0.15 * vel)
            audio[start : start + burst.size] += burst[: audio.size - start]
    return audio / np.max(np.abs(audio))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    rng = np.random.default_rng(4)
    structure = ["intro", "base", "base", "llamada_repique", "intro", "base", "base", "cierre"]
    piece = render_piece(rng, structure)
    audio = _render(piece, rng)
    path = tmp_path_factory.mktemp("audio") / "candombe.wav"
    sf.write(str(path), audio, SR)
    return piece, str(path)


def test_subdivide_and_snap():
    beats = np.array([0.0, 0.5, 1.0, 1.6])
    grid = subdivide(beats)
    assert grid[:5].tolist() == [0.0, 0.125, 0.25, 0.375, 0.5]
    idx, off = snap(np.array([0.13, 1.0 + 0.15 * 1.01]), grid)
    assert idx.tolist() == [1, 9]
    assert abs(off[0] - 0.04) < 1e-6


def test_beats_file_gives_exact_bar_alignment(recording, tmp_path):
    piece, path = recording
    first_downbeat = PRE_ROLL_STEPS * STEP
    beat = 4 * STEP
    rows = []
    # Two beats before the first downbeat, labelled as beats 3 and 4 of bar 0.
    for k in range(-2, piece.steps // 4):
        rows.append(f"{first_downbeat + k * beat:.6f},{(k // 4) + 1}.{(k % 4) + 1}")
    beats_csv = tmp_path / "beats.csv"
    beats_csv.write_text("\n".join(rows))

    summary, _, arrays = analyze_reference(audio_path=path, beats_file=str(beats_csv))
    assert summary["phase_source"] == "annotations"
    assert abs(summary["tempo"] - TEMPO) < 0.5
    got = arrays["velocities"][: piece.steps] > 0
    expected = piece.vel[: got.shape[0]] > 0
    # Onset detection on synthetic bursts is not perfect, but the grid must line up.
    for inst in range(3):
        agreement = np.mean(got[:, inst] == expected[:, inst])
        assert agreement > 0.85, (inst, agreement)

    chico_hits = np.flatnonzero(got[:, 0])
    assert np.mean(chico_hits % 4 != 0) > 0.95
    # Recovered chico microtiming close to what was rendered (fraction of a step).
    matched = [s for s in chico_hits if piece.vel[s, 0] > 0]
    error = [abs(arrays["offsets_frac"][s, 0] - bin_to_offset(int(piece.off[s, 0]))) for s in matched]
    assert np.median(error) < 0.06


def test_phase_estimation_finds_the_downbeat(recording):
    piece, _ = recording
    # Absolute grid indices with the known 8-step pre-roll, as the analyzer sees them.
    steps = {name: np.flatnonzero(piece.vel[:, i] > 0) + PRE_ROLL_STEPS for i, name in enumerate(("chico", "repique", "piano"))}
    weights = {name: np.ones(len(steps[name])) for name in steps}
    shift, confidence = estimate_phase(steps, weights)
    assert shift == PRE_ROLL_STEPS % STEPS_PER_BAR
    assert confidence > 0.2
