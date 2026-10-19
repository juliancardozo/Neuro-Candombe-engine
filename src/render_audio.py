"""Render API-style events to a WAV file, offline.

Uses the per-articulation banks of web/samples/manifest.json when the files
exist (round-robin, like the web player) and falls back to the synthetic
one-shots of generate_demo_samples.py. Microtiming (`offset_ms`) is applied.

    python src/render_audio.py --checkpoint models/candombe_v2.pt --bars 8 --out out.wav
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import generate_demo_samples as demo

SAMPLE_RATE = demo.SAMPLE_RATE
WEB_DIR = Path("web")

SYNTH_FALLBACK = {
    ("chico", "mano"): demo.chico_mano_sample,
    ("chico", "palo"): demo.chico_palo_sample,
    ("repique", "palo"): demo.repique_talk_open,
    ("repique", "mano"): demo.repique_talk_support,
    ("repique", "madera"): lambda: demo.normalize(demo.wood_click(0.07, 2400, 1500, decay=40)),
    ("piano", "bass_open"): demo.piano_sample,
    ("piano", "bass_muted"): lambda: demo.piano_sample()[: int(0.09 * SAMPLE_RATE)],
    ("piano", "stick_open"): lambda: demo.normalize(demo.wood_click(0.09, 900, 420, decay=26)),
    ("piano", "stick_closed"): lambda: demo.normalize(demo.wood_click(0.05, 1000, 520, decay=48)),
    ("clave", "palmas"): demo.clave_palmas_sample,
}


class SampleBank:
    def __init__(self, manifest_path: Optional[Path] = None):
        self.banks: Dict[tuple, List[np.ndarray]] = {}
        self.turn: Dict[tuple, int] = {}
        manifest_path = manifest_path or WEB_DIR / "samples" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for instrument, entries in manifest.items():
                for key, sources in entries.items():
                    loaded = [self._load(src) for src in ([sources] if isinstance(sources, str) else sources)]
                    loaded = [audio for audio in loaded if audio is not None]
                    if loaded:
                        self.banks[(instrument, key)] = loaded

    @staticmethod
    def _load(src: str) -> Optional[np.ndarray]:
        path = WEB_DIR / src.lstrip("/")
        if not path.exists():
            return None
        audio, sr = sf.read(str(path), always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = resample_poly(audio, SAMPLE_RATE, sr)
        return audio.astype(np.float64)

    def get(self, instrument: str, articulation: str) -> np.ndarray:
        for key in ((instrument, articulation), (instrument, "default")):
            if key in self.banks:
                index = self.turn.get(key, 0)
                self.turn[key] = index + 1
                return self.banks[key][index % len(self.banks[key])]
        key = (instrument, articulation)
        if key not in self.banks:
            synth = SYNTH_FALLBACK.get(key) or SYNTH_FALLBACK.get((instrument, "palo")) or demo.chico_palo_sample
            self.banks[key] = [synth()]
        return self.banks[key][0]


def render_events(events: List[dict], duration: float, bank: Optional[SampleBank] = None, gains: Optional[Dict[str, float]] = None) -> np.ndarray:
    bank = bank or SampleBank()
    gains = gains or {"chico": 0.55, "repique": 0.8, "piano": 0.9, "clave": 0.35}
    out = np.zeros(int((duration + 1.0) * SAMPLE_RATE))
    for event in events:
        start = event["time"] + event.get("offset_ms", 0.0) / 1000.0
        index = max(0, int(round(start * SAMPLE_RATE)))
        sample = bank.get(event["instrument"], event.get("articulation") or event.get("hit_type") or "default")
        level = (event["velocity"] / 127.0) ** 1.4 * gains.get(event["instrument"], 0.7)
        end = min(out.size, index + sample.size)
        out[index:end] += sample[: end - index] * level
    peak = np.max(np.abs(out)) or 1.0
    return np.tanh(out / peak * 1.2) * 0.9


def render_to_file(payload: dict, out_path: str, bank: Optional[SampleBank] = None) -> None:
    duration = payload["step_duration"] * payload["bars"] * 16
    sf.write(out_path, render_events(payload["events"], duration, bank), SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="Without it the rule engine is used.")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--tempo", type=int, default=100)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="render.wav")
    args = parser.parse_args()

    # app reads MODEL_PATH at import; a missing path selects the rule engine.
    os.environ["MODEL_PATH"] = args.checkpoint or "__none__"
    import app

    payload = app.generate(tempo=args.tempo, bars=args.bars, density=args.density, temperature=args.temperature, random_seed=args.seed)
    render_to_file(payload, args.out)
    print(f"{payload['engine']}: {len(payload['events'])} golpes -> {args.out}")


if __name__ == "__main__":
    main()
