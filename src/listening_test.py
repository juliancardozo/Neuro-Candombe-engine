"""Blind listening test: anonymous, shuffled clips from each system.

Build the test (share clips/ and ratings.csv with the listeners, keep key.json):
    python src/listening_test.py build --checkpoint models/candombe_v2.pt \
        --reference-glob "data/processed/candombe_dataset/*.npz" --out-dir data/listening_test

Score it once ratings.csv is filled in (1-5 per criterion):
    python src/listening_test.py score --out-dir data/listening_test
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from candombe_rules import random_structure, render_piece
from render_audio import SampleBank, render_to_file
from representation import SECTIONS, load_piece, piece_to_events

CRITERIA = ("groove", "autenticidad", "interes")


def _payload(piece, tempo: float) -> dict:
    step_duration = 60.0 / tempo / 4
    sections = [SECTIONS[s] if s < len(SECTIONS) else "base" for s in piece.section[::16]]
    return {"step_duration": step_duration, "bars": piece.bars, "events": piece_to_events(piece, step_duration, sections)}


def build(args) -> None:
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bank = SampleBank()

    model = None
    if args.checkpoint:
        from generate import generate_piece, load_model  # torch only when a model is used

        model = load_model(args.checkpoint)

    pieces = []
    for i in range(args.n):
        structure = random_structure(rng, args.bars)
        tempo = float(rng.uniform(90, 125))
        density = float(rng.uniform(0.35, 0.75))
        pieces.append(("rules", render_piece(np.random.default_rng(args.seed + i), structure, density, tempo=tempo), tempo))
        if model is not None:
            pieces.append(("model", generate_piece(model, args.bars, tempo, density, structure, seed=args.seed + i), tempo))
    if args.reference_glob:
        paths = sorted(glob.glob(args.reference_glob))
        for path in rng.choice(paths, size=min(args.n, len(paths)), replace=False):
            piece = load_piece(str(path))
            start_bar = int(rng.integers(0, max(1, piece.bars - args.bars + 1)))
            lo, hi = start_bar * 16, (start_bar + args.bars) * 16
            piece.vel, piece.off, piece.art, piece.section = piece.vel[lo:hi], piece.off[lo:hi], piece.art[lo:hi], piece.section[lo:hi]
            pieces.append(("reference", piece, piece.tempo or 110.0))

    order = rng.permutation(len(pieces))
    key = {}
    for clip_number, index in enumerate(order, start=1):
        system, piece, tempo = pieces[index]
        clip = f"clip_{clip_number:03d}.wav"
        render_to_file(_payload(piece, tempo), str(clips_dir / clip), bank)
        key[clip] = system

    (out_dir / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with open(out_dir / "ratings.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", *CRITERIA, "comentario"])
        for clip in sorted(key):
            writer.writerow([clip, "", "", "", ""])
    print(f"{len(key)} clips in {clips_dir}. Share clips/ and ratings.csv; keep key.json private.")


def score(args) -> None:
    out_dir = Path(args.out_dir)
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    ratings = {}
    for path in [Path(p) for p in args.ratings] if args.ratings else [out_dir / "ratings.csv"]:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                system = key.get(row["clip"])
                if system is None:
                    continue
                for criterion in CRITERIA:
                    if row.get(criterion, "").strip():
                        ratings.setdefault(system, {}).setdefault(criterion, []).append(float(row[criterion]))

    print(f"{'system':12s}" + "".join(f"{c:>16s}" for c in CRITERIA))
    for system in sorted(ratings):
        cells = []
        for criterion in CRITERIA:
            values = ratings[system].get(criterion, [])
            cells.append(f"{np.mean(values):.2f}±{np.std(values):.2f} (n={len(values)})" if values else "-")
        print(f"{system:12s}" + "".join(f"{cell:>16s}" for cell in cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--checkpoint", default="")
    b.add_argument("--reference-glob", default="")
    b.add_argument("--n", type=int, default=6, help="Clips per system.")
    b.add_argument("--bars", type=int, default=8)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out-dir", default="data/listening_test")
    s = sub.add_parser("score")
    s.add_argument("--out-dir", default="data/listening_test")
    s.add_argument("--ratings", nargs="*", help="One ratings CSV per listener (default: out-dir/ratings.csv).")
    args = parser.parse_args()
    build(args) if args.command == "build" else score(args)


if __name__ == "__main__":
    main()
