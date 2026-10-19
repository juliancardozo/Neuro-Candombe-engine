"""Objective evaluation: compare generated pieces against a reference corpus.

    python src/evaluate.py --checkpoint models/candombe_v2.pt \
        --reference-glob "data/processed/candombe_dataset/*.npz" --n 64 --bars 8

Reports, per system (model / rules / reference):
  style rules   chico_conformity, piano_style_match (bars that are a
                documented base of the piece's toque, toques.py),
                piano_accent (masa on 1 + open accent on the 4th sixteenth),
                call_adherence
  distributions JS divergence to the reference for onset position, velocity
                class and articulation, per instrument
  microtiming   mean |diff| of per-position mean and std offsets vs reference
  creativity    diversity (pairwise distance between pieces) and copy_rate
                (bars copied verbatim from the reference / training corpus)
Lower distances are closer to the reference; a copy_rate near 1 means the
system memorizes instead of composing.
"""

import argparse
import glob
import json
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

from candombe_rules import random_structure, render_piece
from representation import (
    ARTICULATIONS,
    INSTRUMENTS,
    N_VEL,
    SECTIONS,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
    Piece,
    bin_to_offset,
    load_piece,
)

from toques import CHICO_DE_DOS, CHICO_DE_TRES, REPIQUE_LLAMADA, TOQUES, get_toque

CHICO, REPIQUE, PIANO = range(3)


def _onsets(pattern) -> np.ndarray:
    grid = np.zeros(STEPS_PER_BAR, dtype=bool)
    grid[[step for step, _, _ in pattern.events]] = True
    return grid


CHICO_ONSETS = (_onsets(CHICO_DE_DOS), _onsets(CHICO_DE_TRES))
LLAMADA_ONSETS = _onsets(REPIQUE_LLAMADA)
PIANO_BASES = {key: [_onsets(p) for p in toque.piano_bases] for key, toque in TOQUES.items()}


def _hist(values: np.ndarray, bins: int) -> np.ndarray:
    counts = np.bincount(values.astype(int), minlength=bins).astype(float)
    return counts / counts.sum() if counts.sum() else np.full(bins, 1.0 / bins)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-9
    q = q + 1e-9
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def distributions(pieces: Sequence[Piece]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for inst, name in enumerate(INSTRUMENTS):
        positions, vels, arts = [], [], []
        for piece in pieces:
            hits = np.flatnonzero(piece.vel[:, inst] > 0)
            positions.append(hits % STEPS_PER_BAR)
            vels.append(piece.vel[hits, inst])
            arts.append(piece.art[hits, inst][piece.art[hits, inst] >= 0])
        out[name] = {
            "position": _hist(np.concatenate(positions) if positions else np.array([]), STEPS_PER_BAR),
            "velocity": _hist(np.concatenate(vels) if vels else np.array([]), N_VEL),
            "articulation": _hist(np.concatenate(arts) if arts else np.array([]), len(ARTICULATIONS[name])),
        }
    return out


def microtiming_stats(pieces: Sequence[Piece]) -> Dict[str, np.ndarray]:
    """Per instrument: (4, 2) mean/std offset per 16th position in the beat.

    Positions with too few hits are NaN (e.g. the chico never plays the first 16th).
    """
    out: Dict[str, np.ndarray] = {}
    for inst, name in enumerate(INSTRUMENTS):
        cells: List[List[float]] = [[] for _ in range(STEPS_PER_BEAT)]
        for piece in pieces:
            for step in np.flatnonzero((piece.vel[:, inst] > 0) & (piece.off[:, inst] >= 0)):
                cells[step % STEPS_PER_BEAT].append(bin_to_offset(int(piece.off[step, inst])))
        out[name] = np.array([[np.mean(c), np.std(c)] if len(c) >= 4 else [np.nan, np.nan] for c in cells])
    return out


def style_metrics(pieces: Sequence[Piece]) -> Dict[str, float]:
    chico_match, style_match, accent, call = [], [], [], []
    hits_per_bar = {name: [] for name in INSTRUMENTS}
    offbeat = []
    for piece in pieces:
        bases = PIANO_BASES[get_toque(piece.style).key] if piece.style else [b for v in PIANO_BASES.values() for b in v]
        for bar in range(piece.bars):
            lo = bar * STEPS_PER_BAR
            block = piece.vel[lo : lo + STEPS_PER_BAR] > 0
            sid = int(piece.section[lo])
            section = SECTIONS[sid] if sid < len(SECTIONS) else "base"
            for inst, name in enumerate(INSTRUMENTS):
                hits_per_bar[name].append(int(block[:, inst].sum()))
            if section in ("intro", "final"):
                continue
            chico_match.append(max(float(np.mean(block[:, CHICO] == ref)) for ref in CHICO_ONSETS))
            rep_hits = np.flatnonzero(block[:, REPIQUE])
            if rep_hits.size:
                offbeat.append(float(np.mean(rep_hits % STEPS_PER_BEAT != 0)))
            if section == "llamada_repique":
                call.append(float(np.mean(block[:, REPIQUE] == LLAMADA_ONSETS)))
            if section == "base":
                # A documented base, allowing one added or missing stroke (ghost notes).
                style_match.append(float(min(int(np.sum(block[:, PIANO] != ref)) for ref in bases) <= 1))
                accent.append(float(piece.vel[lo, PIANO] > 0 and piece.vel[lo + 3, PIANO] >= 3))
    mean = lambda values: float(np.mean(values)) if values else float("nan")  # noqa: E731
    metrics = {
        "chico_conformity": mean(chico_match),
        "piano_style_match": mean(style_match),
        "piano_accent": mean(accent),
        "repique_offbeat_ratio": mean(offbeat),
        "call_adherence": mean(call),
    }
    for name in INSTRUMENTS:
        metrics[f"hits_per_bar_{name}"] = mean(hits_per_bar[name])
    return metrics


def _bar_patterns(pieces: Sequence[Piece]) -> List[bytes]:
    """Repique + piano onset pattern per bar (the chico is constant by design)."""
    patterns = []
    for piece in pieces:
        for bar in range(piece.bars):
            block = piece.vel[bar * STEPS_PER_BAR : (bar + 1) * STEPS_PER_BAR, 1:] > 0
            patterns.append(block.tobytes())
    return patterns


def creativity_metrics(pieces: Sequence[Piece], corpus: Sequence[Piece]) -> Dict[str, float]:
    corpus_bars = set(_bar_patterns(corpus))
    own_bars = _bar_patterns(pieces)
    copy_rate = float(np.mean([bar in corpus_bars for bar in own_bars])) if own_bars else 0.0

    grids = [piece.vel[:, 1:] > 0 for piece in pieces]
    distances = []
    for a, b in combinations(range(len(grids)), 2):
        steps = min(grids[a].shape[0], grids[b].shape[0])
        distances.append(float(np.mean(grids[a][:steps] != grids[b][:steps])))
    return {
        "copy_rate": copy_rate,
        "unique_bar_ratio": len(set(own_bars)) / max(len(own_bars), 1),
        "diversity": float(np.mean(distances)) if distances else 0.0,
    }


def compare(pieces: Sequence[Piece], reference: Sequence[Piece], corpus: Sequence[Piece]) -> Dict[str, float]:
    report = style_metrics(pieces)
    report.update(creativity_metrics(pieces, corpus))

    own, ref = distributions(pieces), distributions(reference)
    js_values = []
    for name in INSTRUMENTS:
        for key in ("position", "velocity", "articulation"):
            value = js_divergence(own[name][key], ref[name][key])
            report[f"js_{key}_{name}"] = value
            if key != "articulation" or name == "chico":
                js_values.append(value)
    report["style_distance"] = float(np.mean(js_values))

    own_mt, ref_mt = microtiming_stats(pieces), microtiming_stats(reference)
    diffs = []
    for name in INSTRUMENTS:
        both = ~np.isnan(own_mt[name][:, 0]) & ~np.isnan(ref_mt[name][:, 0])
        if both.any():
            diff = np.abs(own_mt[name][both] - ref_mt[name][both]).mean(axis=0)
            report[f"microtiming_mean_diff_{name}"] = float(diff[0])
            report[f"microtiming_std_diff_{name}"] = float(diff[1])
            diffs.append(float(diff.mean()))
    report["microtiming_distance"] = float(np.mean(diffs)) if diffs else float("nan")
    return report


def generate_sets(checkpoint: Optional[str], n: int, bars: int, seed: int, temperature: float, top_p: float) -> Dict[str, List[Piece]]:
    rng = np.random.default_rng(seed)
    structures = [random_structure(rng, bars) for _ in range(n)]
    densities = rng.uniform(0.3, 0.8, size=n)
    styles = [list(TOQUES)[i % len(TOQUES)] for i in range(n)]
    tempos = [float(rng.uniform(*TOQUES[style].tempo_range)) for style in styles]
    sets = {
        "rules": [
            render_piece(np.random.default_rng(seed + i), structures[i], float(densities[i]), tempo=tempos[i], style=styles[i])
            for i in range(n)
        ]
    }
    if checkpoint:
        from generate import generate_piece, load_model

        model = load_model(checkpoint)
        sets["model"] = [
            generate_piece(model, bars, tempos[i], float(densities[i]), structures[i], style=styles[i], temperature=temperature, top_p=top_p, seed=seed + i)
            for i in range(n)
        ]
    return sets


def _format(value: float) -> str:
    return "   n/a" if value != value else f"{value:6.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--reference-glob", required=True, help="Pieces that define the target style (real recordings ideally).")
    parser.add_argument("--corpus-glob", default="", help="Training corpus for copy_rate (defaults to the reference).")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    reference = [load_piece(path) for path in sorted(glob.glob(args.reference_glob))]
    if not reference:
        raise FileNotFoundError(f"No reference pieces for {args.reference_glob}")
    corpus = [load_piece(path) for path in sorted(glob.glob(args.corpus_glob))] if args.corpus_glob else reference

    sets = generate_sets(args.checkpoint or None, args.n, args.bars, args.seed, args.temperature, args.top_p)
    sets["reference"] = reference
    report = {name: compare(pieces, reference, corpus) for name, pieces in sets.items()}

    keys = list(report["reference"].keys())
    print(f"{'metric':32s}" + "".join(f"{name:>12s}" for name in report))
    for key in keys:
        print(f"{key:32s}" + "".join(f"{_format(report[name][key]):>12s}" for name in report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
