import argparse
import json
import os
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf

from analyze_reference import BANDS, INSTRUMENTS, bandpass_filter


WINDOW_MS = {
    "chico": {"pre": 8, "post": 180},
    "repique": {"pre": 12, "post": 320},
    "piano": {"pre": 15, "post": 420},
}

HIT_TYPE_BONUS = {
    "chico": {"accent": 0.12, "pulse": 0.06},
    "repique": {"accent_open": 0.14, "phrase_hit": 0.1, "response": 0.05},
    "piano": {"bass_open": 0.12, "bass_muted": 0.05},
}


def load_analysis(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def score_event(event: dict) -> float:
    velocity_score = float(event.get("velocity", 0)) / 127.0
    confidence_score = float(event.get("confidence", 0.0))
    energy_score = float(event.get("band_energy", 0.0))
    hit_bonus = HIT_TYPE_BONUS.get(event["instrument"], {}).get(event.get("hit_type", ""), 0.0)
    return 0.45 * confidence_score + 0.35 * velocity_score + 0.20 * energy_score + hit_bonus


def select_event(events: List[dict], instrument: str, total_duration_s: float) -> Optional[dict]:
    window = WINDOW_MS[instrument]
    valid_events = []
    for event in events:
        if event["instrument"] != instrument:
            continue
        start_s = max(0.0, float(event["time"]) - window["pre"] / 1000.0)
        end_s = float(event["time"]) + window["post"] / 1000.0
        if end_s <= total_duration_s and start_s < total_duration_s:
            valid_events.append(event)

    if not valid_events:
        return None

    return max(valid_events, key=score_event)


def trim_and_fade(sample: np.ndarray, sr: int) -> np.ndarray:
    if sample.size == 0:
        return sample

    energy = np.abs(sample)
    peak = float(np.max(energy))
    if peak <= 1e-6:
        return sample

    threshold = max(peak * 0.12, 1e-4)
    active = np.flatnonzero(energy >= threshold)
    if active.size:
        lead = int(sr * 0.002)
        tail = int(sr * 0.03)
        start = max(0, int(active[0]) - lead)
        end = min(sample.size, int(active[-1]) + tail + 1)
        sample = sample[start:end]

    fade_in = min(int(sr * 0.004), max(1, sample.size // 6))
    fade_out = min(int(sr * 0.018), max(1, sample.size // 5))

    if fade_in > 1:
        sample[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 1:
        sample[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    peak = float(np.max(np.abs(sample)))
    if peak > 1e-6:
        sample = 0.95 * (sample / peak)

    return sample.astype(np.float32)


def extract_sample(audio: np.ndarray, sr: int, event: dict) -> np.ndarray:
    instrument = event["instrument"]
    band_low, band_high = BANDS[instrument]
    filtered = bandpass_filter(audio, sr, band_low, band_high)

    window = WINDOW_MS[instrument]
    start = max(0, int((float(event["time"]) - window["pre"] / 1000.0) * sr))
    end = min(audio.size, int((float(event["time"]) + window["post"] / 1000.0) * sr))
    segment = filtered[start:end].copy()
    return trim_and_fade(segment, sr)


def save_samples(analysis_json: str, output_dir: str, manifest_path: Optional[str]) -> Dict[str, dict]:
    payload = load_analysis(analysis_json)
    summary = payload.get("summary", {})
    events = payload.get("events", [])
    audio_path = summary.get("audio_path")
    if not audio_path:
        raise ValueError("The analysis JSON does not include the original audio path.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")

    analysis_start = float(summary.get("start", 0.0) or 0.0)
    explicit_duration = summary.get("duration")
    max_event_time = max((float(event.get("time", 0.0)) for event in events), default=0.0)
    load_duration = explicit_duration if explicit_duration is not None else max_event_time + 1.0

    audio, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
        offset=analysis_start,
        duration=load_duration,
    )
    if audio.size == 0:
        raise ValueError("The reference audio is empty.")

    os.makedirs(output_dir, exist_ok=True)
    selected: Dict[str, dict] = {}
    total_duration_s = audio.size / float(sr)

    for instrument in INSTRUMENTS:
        event = select_event(events, instrument, total_duration_s=total_duration_s)
        if event is None:
            continue

        sample = extract_sample(audio, sr, event)
        if sample.size == 0:
            continue

        out_path = os.path.join(output_dir, f"{instrument}.wav")
        sf.write(out_path, sample, sr)
        selected[instrument] = {
            "path": out_path,
            "source_time": round(float(event["time"]), 4),
            "velocity": int(event.get("velocity", 0)),
            "hit_type": event.get("hit_type", "unknown"),
            "confidence": float(event.get("confidence", 0.0)),
            "duration_ms": round(sample.size / sr * 1000.0, 2),
        }

    if manifest_path:
        parent = os.path.dirname(manifest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "analysis_json": analysis_json,
                    "audio_path": audio_path,
                    "analysis_start": analysis_start,
                    "analysis_duration": load_duration,
                    "samples": selected,
                },
                handle,
                indent=2,
                ensure_ascii=True,
            )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one-shot chico/repique/piano samples from a previously analyzed reference."
    )
    parser.add_argument(
        "--analysis-json",
        default="data/analysis/reference_events.json",
        help="Path to the analysis JSON generated by analyze_reference.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="web/samples",
        help="Directory where chico.wav, repique.wav and piano.wav will be written.",
    )
    parser.add_argument(
        "--manifest",
        default="data/analysis/reference_samples.json",
        help="Optional JSON manifest describing the selected sample sources.",
    )
    args = parser.parse_args()

    selected = save_samples(
        analysis_json=args.analysis_json,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )

    if not selected:
        print("No samples could be extracted from the analysis.")
        return

    print(f"Extracted {len(selected)} samples from {args.analysis_json}")
    for instrument in INSTRUMENTS:
        if instrument not in selected:
            continue
        sample = selected[instrument]
        print(
            f"- {instrument}: {sample['path']} "
            f"(t={sample['source_time']}s, hit_type={sample['hit_type']}, duration={sample['duration_ms']}ms)"
        )


if __name__ == "__main__":
    main()
