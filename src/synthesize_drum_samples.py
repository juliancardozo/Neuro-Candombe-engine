"""Physically-inspired one-shots for chico, repique and piano.

Each stroke mixes:
  - membrane: damped modes of a circular drumhead (Bessel-zero ratios), with
    the short pitch drop of a tight, fire-tuned skin right after the hit;
  - excitation: the stick's click (palo) or a softer, darker hand contact (mano);
  - shell: wooden barrel resonances, alone for strokes on the madera.

Several takes per articulation are rendered with small random variations so
round-robin playback never repeats the exact same hit. Output goes to
web/samples/llamadas/ together with web/samples/manifest.json.

    python src/synthesize_drum_samples.py

Real recordings sound better than any synthesis: when there are good one-shots,
put them in web/samples and point the manifest at them
(build_reference_corpus.py does this from recordings).
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100

# Zeros of Bessel functions relative to the fundamental (ideal circular membrane).
MEMBRANE_RATIOS = (1.000, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.600, 3.652, 4.060, 4.154)


@dataclass(frozen=True)
class Stroke:
    f0: float  # membrane fundamental, Hz
    duration: float  # s
    decay: float  # fundamental decay time constant, s
    brightness: float  # 0..1, weight of the upper membrane modes
    glide: float  # initial pitch excess (fraction) that relaxes after the hit
    click: float  # stick click level
    hand: float  # hand-contact thump level
    shell: float  # wooden shell level
    membrane: float = 1.0
    shell_freqs: tuple = (720.0, 1310.0, 2080.0, 2950.0)
    shell_decay: float = 0.035
    level: float = 1.0  # relative loudness of this articulation


def _bandpass(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(2, [low, min(high, SR * 0.45)], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, signal)


def _lowpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    return sosfilt(butter(2, cutoff, btype="lowpass", fs=SR, output="sos"), signal)


def _unit(signal: np.ndarray) -> np.ndarray:
    return signal / (np.max(np.abs(signal)) or 1.0)


def render_stroke(stroke: Stroke, rng: np.random.Generator) -> np.ndarray:
    """Each component is normalized on its own, so the Stroke levels are real mix ratios."""
    n = int(stroke.duration * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)

    if stroke.membrane > 0:
        glide_tc = 0.035
        # Integral of f(t) = f * (1 + glide * exp(-t / tc)) gives the phase.
        warp = t + stroke.glide * glide_tc * (1.0 - np.exp(-t / glide_tc))
        skin = np.zeros(n)
        for k, ratio in enumerate(MEMBRANE_RATIOS):
            amp = (stroke.brightness ** k) / (1.0 + 0.35 * k) * rng.uniform(0.75, 1.25)
            tau = stroke.decay / (1.0 + 0.9 * k * (ratio - 1.0))
            freq = stroke.f0 * ratio * rng.uniform(0.995, 1.005)
            skin += amp * np.sin(2 * np.pi * freq * warp + rng.uniform(0, 2 * np.pi)) * np.exp(-t / tau)
        out += stroke.membrane * _unit(skin)

    if stroke.click > 0:
        burst = rng.standard_normal(n) * np.exp(-t / 0.0012)
        out += stroke.click * _unit(_bandpass(burst, 1800, 9000))
    if stroke.hand > 0:
        thump = rng.standard_normal(n) * np.exp(-t / 0.006)
        out += stroke.hand * _unit(_lowpass(thump, 900))
    if stroke.shell > 0:
        wood = np.zeros(n)
        for k, freq in enumerate(stroke.shell_freqs):
            tau = stroke.shell_decay / (1.0 + 0.4 * k)
            wood += (0.8 ** k) * np.sin(2 * np.pi * freq * rng.uniform(0.98, 1.02) * t) * np.exp(-t / tau)
        wood = _unit(wood) + 0.5 * _unit(_bandpass(rng.standard_normal(n) * np.exp(-t / 0.002), 1500, 7000))
        out += stroke.shell * _unit(wood)

    # A hand stroke has a soft attack; a stick is instantaneous.
    attack = 0.0025 if stroke.hand > stroke.click else 0.0003
    out *= np.clip(t / attack, 0.0, 1.0)
    fade = min(n, int(0.02 * SR))
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    out -= np.mean(out)
    return _unit(out) * 0.89 * stroke.level


# Tunings are typical, not measured: candombe drums are tuned tight and high.
CHICO, REPIQUE, PIANO = 330.0, 225.0, 108.0

ARTICULATIONS = {
    "chico": {
        "palo": Stroke(CHICO, 0.32, 0.075, 0.7, 0.05, click=0.35, hand=0.0, shell=0.1),
        "mano": Stroke(CHICO, 0.22, 0.045, 0.4, 0.025, click=0.0, hand=0.35, shell=0.04, level=0.7),
    },
    "repique": {
        "palo": Stroke(REPIQUE, 0.6, 0.15, 0.68, 0.06, click=0.35, hand=0.0, shell=0.1),
        "mano": Stroke(REPIQUE, 0.32, 0.07, 0.4, 0.03, click=0.0, hand=0.35, shell=0.04, level=0.65),
        "madera": Stroke(REPIQUE, 0.18, 0.02, 0.3, 0.0, click=0.25, hand=0.0, shell=1.0, membrane=0.15,
                         shell_freqs=(1050.0, 1720.0, 2600.0, 3400.0), shell_decay=0.03, level=0.9),
    },
    "piano": {
        "bass_open": Stroke(PIANO, 1.1, 0.32, 0.55, 0.08, click=0.22, hand=0.0, shell=0.08,
                            shell_freqs=(520.0, 960.0, 1500.0, 2200.0)),
        "bass_muted": Stroke(PIANO, 0.35, 0.07, 0.45, 0.035, click=0.2, hand=0.2, shell=0.06,
                             shell_freqs=(520.0, 960.0, 1500.0, 2200.0), level=0.85),
        "stick_open": Stroke(PIANO, 0.8, 0.22, 0.45, 0.05, click=0.15, hand=0.0, shell=0.06,
                             shell_freqs=(520.0, 960.0, 1500.0, 2200.0), level=0.7),
        "stick_closed": Stroke(PIANO, 0.16, 0.02, 0.3, 0.0, click=0.25, hand=0.0, shell=1.0, membrane=0.2,
                               shell_freqs=(780.0, 1380.0, 2150.0, 2900.0), shell_decay=0.03, level=0.6),
    },
}


def vary(stroke: Stroke, rng: np.random.Generator) -> Stroke:
    """A different take: the player never hits exactly the same way twice."""
    return replace(
        stroke,
        f0=stroke.f0 * rng.uniform(0.985, 1.015),
        decay=stroke.decay * rng.uniform(0.9, 1.1),
        brightness=float(np.clip(stroke.brightness * rng.uniform(0.9, 1.1), 0.05, 0.95)),
        glide=stroke.glide * rng.uniform(0.8, 1.2),
        click=stroke.click * rng.uniform(0.85, 1.15),
    )


def build(out_dir: Path, manifest_path: Path, takes: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = {}
    web_root = manifest_path.parent.parent  # web/
    for instrument, articulations in ARTICULATIONS.items():
        manifest[instrument] = {}
        for name, stroke in articulations.items():
            files = []
            for take in range(1, takes + 1):
                audio = render_stroke(vary(stroke, rng), rng)
                path = out_dir / f"{instrument}_{name}_{take:02d}.wav"
                sf.write(str(path), audio.astype(np.float32), SR, subtype="PCM_16")
                files.append("/" + path.relative_to(web_root).as_posix())
            manifest[instrument][name] = files
        manifest[instrument]["default"] = [f for files in manifest[instrument].values() for f in files][:takes]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("web/samples/llamadas"))
    parser.add_argument("--manifest", type=Path, default=Path("web/samples/manifest.json"))
    parser.add_argument("--takes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = build(args.out_dir, args.manifest, args.takes, args.seed)
    count = sum(len(files) for inst in manifest.values() for key, files in inst.items() if key != "default")
    print(f"{count} samples in {args.out_dir}, manifest {args.manifest}")


if __name__ == "__main__":
    main()
