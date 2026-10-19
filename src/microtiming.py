"""Microtiming profiles: mean/std offset per instrument and 16th position in the beat.

Offsets are fractions of a 16th step (see representation.py).

DEFAULT_PROFILE is the documented groove of the Ansina toque at its typical
tempo (Jure & Rocamora 2022, see toques.groove_offsets); the rule renderer uses
the per-toque, per-tempo version directly. To measure a profile from recordings:

    python src/microtiming.py --data-glob "data/processed/candombe_dataset/*.npz" --out data/analysis/microtiming_profile.json
"""

import argparse
import glob
import json
from typing import Dict, Iterable, Optional

import numpy as np

from representation import INSTRUMENTS, STEPS_PER_BEAT, Piece, bin_to_offset, load_piece
from toques import TIMING_SPREAD, get_toque, groove_offsets

# instrument -> [(mean, std)] for 16th positions 0..3 inside the beat
DEFAULT_PROFILE: Dict[str, list] = {
    name: [(round(mean, 4), TIMING_SPREAD[name]) for mean in groove_offsets(get_toque("ansina"), name, get_toque("ansina").tempo)]
    for name in INSTRUMENTS
}

MIN_HITS_PER_CELL = 8


def estimate_profile(pieces: Iterable[Piece], fallback: Optional[Dict[str, list]] = None) -> Dict[str, list]:
    fallback = fallback or DEFAULT_PROFILE
    samples = {inst: [[] for _ in range(STEPS_PER_BEAT)] for inst in INSTRUMENTS}
    for piece in pieces:
        for inst_idx, instrument in enumerate(INSTRUMENTS):
            known = np.flatnonzero((piece.vel[:, inst_idx] > 0) & (piece.off[:, inst_idx] >= 0))
            for step in known:
                samples[instrument][step % STEPS_PER_BEAT].append(bin_to_offset(int(piece.off[step, inst_idx])))

    profile = {}
    for instrument in INSTRUMENTS:
        cells = []
        for position in range(STEPS_PER_BEAT):
            values = samples[instrument][position]
            if len(values) >= MIN_HITS_PER_CELL:
                cells.append((round(float(np.mean(values)), 4), round(float(max(np.std(values), 0.004)), 4)))
            else:
                cells.append(tuple(fallback[instrument][position]))
        profile[instrument] = cells
    return profile


def load_profile(path: Optional[str]) -> Dict[str, list]:
    if not path:
        return DEFAULT_PROFILE
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {instrument: [tuple(cell) for cell in raw[instrument]] for instrument in INSTRUMENTS}


def sample_offset(profile: Dict[str, list], instrument: str, step: int, rng: np.random.Generator) -> float:
    mean, std = profile[instrument][step % STEPS_PER_BEAT]
    return float(np.clip(rng.normal(mean, std), -0.4, 0.4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a microtiming profile from pieces with known offsets.")
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--out", default="data/analysis/microtiming_profile.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise FileNotFoundError(f"No files for: {args.data_glob}")
    profile = estimate_profile(load_piece(path) for path in paths)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    for instrument, cells in profile.items():
        print(instrument, " ".join(f"{mean:+.3f}±{std:.3f}" for mean, std in cells))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
