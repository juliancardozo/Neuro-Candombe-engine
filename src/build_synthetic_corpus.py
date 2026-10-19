"""Synthetic training corpus from the rule renderer (candombe_rules.render_piece).

Unlike the older build_*_dataset.py scripts, every piece carries sections,
articulations, microtiming, tempo and a varied density, so the model can learn
the conditioning. Synthetic data only teaches what the rules know: mix it with
real recordings (import_candombe_dataset.py) and weight those higher in train.py.
"""

import argparse
from pathlib import Path

import numpy as np

from candombe_rules import random_structure, render_piece
from microtiming import load_profile
from representation import VEL_CENTERS, bin_to_offset, save_piece


def build_corpus(out_dir: Path, count: int, bars_choices, seed: int, profile_path: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    profile = load_profile(profile_path or None)
    for index in range(count):
        bars = int(rng.choice(bars_choices))
        structure = random_structure(rng, bars)
        tempo = float(rng.uniform(80, 140))
        density = float(rng.beta(2.2, 2.2))
        piece = render_piece(rng, structure, density=density, profile=profile, tempo=tempo)

        velocities = np.zeros(piece.vel.shape, dtype=np.int32)
        for value, center in enumerate(VEL_CENTERS, start=1):
            velocities[piece.vel == value] = center
        offsets = np.vectorize(bin_to_offset, otypes=[float])(piece.off)
        offsets[piece.off < 0] = np.nan
        save_piece(
            str(out_dir / f"synthetic_{index:05d}.npz"),
            velocities,
            offsets_frac=offsets,
            articulation=piece.art,
            sections=structure,
            tempo=tempo,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/synthetic"))
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--bars", type=str, default="8,12,16", help="Comma-separated bar counts to sample from.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--microtiming-profile", type=str, default="", help="JSON from microtiming.py (defaults to the provisional profile).")
    args = parser.parse_args()
    bars_choices = [int(b) for b in args.bars.split(",") if b.strip()]
    build_corpus(args.out_dir, args.count, bars_choices, args.seed, args.microtiming_profile)
    print(f"Saved {args.count} pieces to {args.out_dir}")


if __name__ == "__main__":
    main()
