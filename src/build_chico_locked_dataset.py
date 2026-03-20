import argparse
from pathlib import Path

import numpy as np

from build_repique_locked_dataset import clamp, jitter, section_for_bar, write_template


CHICO_VARIATIONS = (
    (
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
    ),
    (
        None,
        {"velocity": 97},
        {"velocity": 80},
        {"velocity": 83},
        None,
        {"velocity": 99},
        {"velocity": 79},
        {"velocity": 82},
        None,
        {"velocity": 98},
        {"velocity": 80},
        {"velocity": 84},
        None,
        {"velocity": 100},
        {"velocity": 82},
        {"velocity": 85},
    ),
    (
        None,
        {"velocity": 100},
        {"velocity": 84},
        {"velocity": 86},
        None,
        {"velocity": 101},
        {"velocity": 82},
        {"velocity": 84},
        None,
        {"velocity": 100},
        {"velocity": 83},
        {"velocity": 85},
        None,
        {"velocity": 102},
        {"velocity": 84},
        {"velocity": 87},
    ),
)

PIANO_SUPPORT = (
    (
        {"velocity": 116},
        None,
        None,
        {"velocity": 86},
        None,
        None,
        None,
        None,
        {"velocity": 118},
        None,
        None,
        {"velocity": 90},
        None,
        None,
        None,
        None,
    ),
    (
        {"velocity": 118},
        None,
        None,
        {"velocity": 88},
        None,
        None,
        None,
        None,
        {"velocity": 117},
        None,
        {"velocity": 58},
        {"velocity": 92},
        None,
        None,
        None,
        None,
    ),
)

REPIQUE_SUPPORT = (
    (
        None,
        {"velocity": 78},
        None,
        None,
        {"velocity": 88},
        None,
        None,
        None,
        None,
        {"velocity": 94},
        None,
        {"velocity": 70},
        None,
        None,
        {"velocity": 102},
        None,
    ),
    (
        None,
        {"velocity": 82},
        None,
        {"velocity": 54},
        None,
        None,
        {"velocity": 62},
        None,
        None,
        {"velocity": 98},
        None,
        None,
        {"velocity": 76},
        None,
        {"velocity": 108},
        None,
    ),
)


def render_example(rng: np.random.Generator, bars: int) -> np.ndarray:
    total_steps = bars * 16
    data = np.zeros((total_steps, 3), dtype=np.int32)
    chico_variant = CHICO_VARIATIONS[int(rng.integers(0, len(CHICO_VARIATIONS)))]

    for bar_index in range(bars):
        step_offset = bar_index * 16
        section = section_for_bar(bar_index, bars)

        write_template(data[:, 0], chico_variant, step_offset, rng, spread=2)

        piano_template = PIANO_SUPPORT[(bar_index + int(rng.integers(0, len(PIANO_SUPPORT)))) % len(PIANO_SUPPORT)]
        write_template(data[:, 2], piano_template, step_offset, rng, spread=5)

        repique_template = REPIQUE_SUPPORT[(bar_index + int(rng.integers(0, len(REPIQUE_SUPPORT)))) % len(REPIQUE_SUPPORT)]
        for step_in_bar, hit in enumerate(repique_template):
            if not hit:
                continue
            if hit["velocity"] <= 62 and rng.random() < 0.18:
                continue
            velocity = jitter(hit["velocity"], 8, rng)
            if section == "intro" and step_in_bar in (12, 14):
                velocity = clamp(velocity - 8, 40, 127)
            if section == "cierre" and step_in_bar == 14:
                velocity = clamp(velocity + 6, 40, 127)
            data[step_offset + step_in_bar, 1] = velocity

    return data


def build_dataset(out_dir: Path, count: int, bars: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        np.savez(out_dir / f"chico_locked_{index:04d}.npz", data=render_example(rng, bars=bars))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/chico_locked"))
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    build_dataset(out_dir=args.out_dir, count=args.count, bars=args.bars, seed=args.seed)


if __name__ == "__main__":
    main()
