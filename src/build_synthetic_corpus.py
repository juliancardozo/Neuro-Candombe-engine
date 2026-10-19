"""Synthetic training corpus from the rule renderer (candombe_rules.render_piece).

Pieces are balanced across the three toques madre (toques.py), with tempos in
each toque's range, half of them "subiendo" (starting slower), and every piece
labelled with its style, sections, articulations and microtiming so the model
learns the conditioning. Synthetic data only teaches what the documented
patterns say: mix it with real recordings (import_candombe_dataset.py) and
weight those higher in train.py.
"""

import argparse
from pathlib import Path

import numpy as np

from candombe_rules import random_structure, render_piece
from representation import VEL_CENTERS, bin_to_offset, save_piece
from toques import TOQUES


def build_corpus(out_dir: Path, count: int, bars_choices, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    styles = list(TOQUES)
    for index in range(count):
        toque = TOQUES[styles[index % len(styles)]]
        bars = int(rng.choice(bars_choices))
        structure = random_structure(rng, bars)
        tempo = float(rng.uniform(*toque.tempo_range))
        density = float(rng.beta(2.2, 2.2))
        piece = render_piece(rng, structure, density=density, tempo=tempo, style=toque.key, subir=bool(rng.random() < 0.5))

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
            style=toque.key,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/synthetic"))
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--bars", type=str, default="8,12,16", help="Comma-separated bar counts to sample from.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    bars_choices = [int(b) for b in args.bars.split(",") if b.strip()]
    build_corpus(args.out_dir, args.count, bars_choices, args.seed)
    print(f"Saved {args.count} pieces to {args.out_dir}")


if __name__ == "__main__":
    main()
