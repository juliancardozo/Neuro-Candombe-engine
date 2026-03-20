import argparse
from pathlib import Path

import numpy as np


CHICO_TEMPLATE = (
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

PIANO_TEMPLATES = (
    (
        {"velocity": 118},
        None,
        None,
        {"velocity": 92},
        None,
        None,
        None,
        None,
        {"velocity": 120},
        None,
        {"velocity": 47},
        {"velocity": 95},
        None,
        None,
        {"velocity": 42},
        None,
    ),
    (
        {"velocity": 121},
        None,
        None,
        None,
        {"velocity": 86},
        None,
        None,
        None,
        {"velocity": 116},
        None,
        {"velocity": 46},
        None,
        {"velocity": 91},
        None,
        None,
        {"velocity": 62},
    ),
)

REPIQUE_TEMPLATES = (
    (
        None,
        {"velocity": 84},
        None,
        {"velocity": 62},
        {"velocity": 96},
        None,
        {"velocity": 54},
        None,
        None,
        {"velocity": 102},
        None,
        {"velocity": 76},
        {"velocity": 90},
        None,
        {"velocity": 114},
        None,
    ),
    (
        {"velocity": 78},
        None,
        {"velocity": 48},
        None,
        {"velocity": 98},
        {"velocity": 58},
        None,
        {"velocity": 74},
        None,
        {"velocity": 108},
        None,
        None,
        {"velocity": 86},
        None,
        {"velocity": 116},
        {"velocity": 52},
    ),
    (
        None,
        {"velocity": 82},
        {"velocity": 46},
        None,
        {"velocity": 94},
        None,
        {"velocity": 64},
        {"velocity": 72},
        None,
        {"velocity": 104},
        None,
        {"velocity": 60},
        {"velocity": 88},
        None,
        {"velocity": 118},
        None,
    ),
)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def jitter(value: int, spread: int, rng: np.random.Generator) -> int:
    return clamp(int(value + rng.integers(-spread, spread + 1)), 1, 127)


def section_for_bar(bar_index: int, total_bars: int) -> str:
    if total_bars <= 1:
        return "cierre"
    position = bar_index / max(total_bars, 1)
    if position < 0.25:
        return "intro"
    if position < 0.5:
        return "llamada_repique"
    if position < 0.75:
        return "llamada_piano"
    return "cierre"


def write_template(track: np.ndarray, template, step_offset: int, rng: np.random.Generator, spread: int) -> None:
    for step_in_bar, hit in enumerate(template):
        if not hit:
            continue
        track[step_offset + step_in_bar] = jitter(hit["velocity"], spread, rng)


def render_example(rng: np.random.Generator, bars: int) -> np.ndarray:
    total_steps = bars * 16
    data = np.zeros((total_steps, 3), dtype=np.int32)

    for bar_index in range(bars):
        step_offset = bar_index * 16
        section = section_for_bar(bar_index, bars)

        write_template(data[:, 0], CHICO_TEMPLATE, step_offset, rng, spread=3)

        piano_template = PIANO_TEMPLATES[(bar_index + rng.integers(0, 2)) % len(PIANO_TEMPLATES)]
        write_template(data[:, 2], piano_template, step_offset, rng, spread=6)

        repique_template = REPIQUE_TEMPLATES[(bar_index + rng.integers(0, len(REPIQUE_TEMPLATES))) % len(REPIQUE_TEMPLATES)]
        for step_in_bar, hit in enumerate(repique_template):
            if not hit:
                continue

            include_hit = True
            velocity_spread = 8
            if hit["velocity"] <= 60:
                include_hit = rng.random() > 0.12
                velocity_spread = 10

            if not include_hit:
                continue

            velocity = jitter(hit["velocity"], velocity_spread, rng)
            if section == "cierre" and step_in_bar in (12, 14, 15):
                velocity = clamp(velocity + 8, 40, 127)
            elif section == "intro" and step_in_bar in (12, 14, 15):
                velocity = clamp(velocity - 10, 40, 127)
            data[step_offset + step_in_bar, 1] = velocity

        # Reinforce grounded quarter anchors in repique without saturating the bar.
        for anchor_step in (4, 12):
            absolute_step = step_offset + anchor_step
            if data[absolute_step, 1] == 0 and rng.random() > 0.42:
                data[absolute_step, 1] = jitter(78, 10, rng)

    return data


def build_dataset(out_dir: Path, count: int, bars: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        example = render_example(rng, bars=bars)
        np.savez(out_dir / f"repique_locked_{index:04d}.npz", data=example)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/repique_locked"))
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build_dataset(out_dir=args.out_dir, count=args.count, bars=args.bars, seed=args.seed)


if __name__ == "__main__":
    main()
