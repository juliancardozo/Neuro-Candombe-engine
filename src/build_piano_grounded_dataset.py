import argparse
from pathlib import Path

import numpy as np

from build_repique_locked_dataset import clamp, jitter, section_for_bar, write_template


CHICO_SUPPORT = (
    None,
    {"velocity": 98},
    {"velocity": 82},
    {"velocity": 84},
    None,
    {"velocity": 100},
    {"velocity": 80},
    {"velocity": 83},
    None,
    {"velocity": 99},
    {"velocity": 81},
    {"velocity": 85},
    None,
    {"velocity": 101},
    {"velocity": 83},
    {"velocity": 86},
)

PIANO_VARIATIONS = (
    (
        {"velocity": 120},
        None,
        None,
        {"velocity": 86},
        None,
        None,
        None,
        None,
        {"velocity": 118},
        None,
        {"velocity": 58},
        {"velocity": 92},
        None,
        None,
        None,
        None,
    ),
    (
        {"velocity": 117},
        None,
        None,
        {"velocity": 84},
        None,
        None,
        None,
        None,
        {"velocity": 115},
        None,
        None,
        {"velocity": 90},
        None,
        None,
        {"velocity": 56},
        None,
    ),
    (
        {"velocity": 119},
        None,
        None,
        {"velocity": 88},
        None,
        None,
        None,
        None,
        {"velocity": 116},
        {"velocity": 55},
        None,
        {"velocity": 94},
        None,
        None,
        None,
        None,
    ),
)

REPIQUE_REFERENCE = (
    (
        None,
        {"velocity": 82},
        None,
        {"velocity": 58},
        {"velocity": 94},
        None,
        None,
        None,
        None,
        {"velocity": 100},
        None,
        {"velocity": 72},
        None,
        None,
        {"velocity": 110},
        None,
    ),
    (
        {"velocity": 78},
        None,
        None,
        None,
        {"velocity": 96},
        {"velocity": 54},
        None,
        None,
        None,
        {"velocity": 104},
        None,
        None,
        {"velocity": 84},
        None,
        {"velocity": 112},
        None,
    ),
)


def render_example(rng: np.random.Generator, bars: int) -> np.ndarray:
    total_steps = bars * 16
    data = np.zeros((total_steps, 3), dtype=np.int32)

    for bar_index in range(bars):
        step_offset = bar_index * 16
        section = section_for_bar(bar_index, bars)

        write_template(data[:, 0], CHICO_SUPPORT, step_offset, rng, spread=3)

        piano_template = PIANO_VARIATIONS[(bar_index + int(rng.integers(0, len(PIANO_VARIATIONS)))) % len(PIANO_VARIATIONS)]
        for step_in_bar, hit in enumerate(piano_template):
            if not hit:
                continue
            velocity = jitter(hit["velocity"], 6, rng)
            if hit["velocity"] <= 58 and rng.random() < 0.16:
                continue
            if section == "cierre" and step_in_bar in (11, 14):
                velocity = clamp(velocity + 6, 55, 127)
            data[step_offset + step_in_bar, 2] = velocity

        repique_template = REPIQUE_REFERENCE[(bar_index + int(rng.integers(0, len(REPIQUE_REFERENCE)))) % len(REPIQUE_REFERENCE)]
        for step_in_bar, hit in enumerate(repique_template):
            if not hit:
                continue
            if hit["velocity"] < 60 and rng.random() < 0.22:
                continue
            velocity = jitter(hit["velocity"], 8, rng)
            if section == "intro" and step_in_bar >= 12:
                velocity = clamp(velocity - 10, 40, 127)
            data[step_offset + step_in_bar, 1] = velocity

    return data


def build_dataset(out_dir: Path, count: int, bars: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        np.savez(out_dir / f"piano_grounded_{index:04d}.npz", data=render_example(rng, bars=bars))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/piano_grounded"))
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    build_dataset(out_dir=args.out_dir, count=args.count, bars=args.bars, seed=args.seed)


if __name__ == "__main__":
    main()
