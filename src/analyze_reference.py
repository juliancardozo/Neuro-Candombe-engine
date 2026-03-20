import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


INSTRUMENTS = ("chico", "repique", "piano")
BANDS = {
    "piano": (45.0, 240.0),
    "repique": (240.0, 1800.0),
    "chico": (1800.0, 8000.0),
}
IDX_MAP = {"chico": 0, "repique": 1, "piano": 2}


@dataclass
class ReferenceEvent:
    instrument: str
    time: float
    step: int
    bar: int
    step_in_bar: int
    offset_ms: float
    velocity: int
    hit_type: str
    confidence: float
    band_energy: float


def bandpass_filter(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sr * 0.5
    low = max(1.0, low_hz) / nyquist
    high = min(high_hz / nyquist, 0.999)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio)


def detect_onsets(band_audio: np.ndarray, sr: int, hop_length: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=band_audio, sr=sr, hop_length=hop_length)
    if onset_env.size == 0:
        return np.array([], dtype=np.int64), onset_env

    norm_env = onset_env / max(np.max(onset_env), 1e-6)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=norm_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        pre_max=2,
        post_max=2,
        pre_avg=4,
        post_avg=4,
        delta=threshold,
        wait=2,
    )
    return onset_frames, norm_env


def quantize_time(time_s: float, step_duration: float) -> Tuple[int, float]:
    step = int(np.round(time_s / step_duration))
    quantized_time = step * step_duration
    offset_ms = (time_s - quantized_time) * 1000.0
    return step, offset_ms


def infer_hit_type(instrument: str, velocity: int, prev_time: Optional[float], next_time: Optional[float]) -> str:
    vel_norm = velocity / 127.0
    prev_gap = None if prev_time is None else prev_time
    next_gap = None if next_time is None else next_time

    if instrument == "chico":
        return "accent" if vel_norm >= 0.72 else "pulse"

    if instrument == "piano":
        return "bass_open" if vel_norm >= 0.68 else "bass_muted"

    if vel_norm <= 0.34:
        return "ghost"
    if vel_norm >= 0.8:
        return "accent_open"
    if (prev_gap is None or prev_gap > 0.42) and (next_gap is None or next_gap > 0.42):
        return "phrase_hit"
    return "response"


def events_to_grid(events: List[ReferenceEvent], total_steps: int) -> np.ndarray:
    arr = np.zeros((total_steps, len(INSTRUMENTS)), dtype=np.uint8)
    for event in events:
        if 0 <= event.step < total_steps:
            inst_idx = IDX_MAP[event.instrument]
            arr[event.step, inst_idx] = max(arr[event.step, inst_idx], event.velocity)
    return arr


def analyze_reference(
    audio_path: str,
    bars: int,
    tempo: Optional[float],
    start: float,
    duration: Optional[float],
    resolution: int,
    hop_length: int,
) -> Tuple[dict, List[ReferenceEvent], np.ndarray]:
    audio, sr = librosa.load(audio_path, sr=22050, mono=True, offset=start, duration=duration)
    if audio.size == 0:
        raise ValueError("The selected audio segment is empty.")

    audio = librosa.util.normalize(audio)

    estimated_tempo = tempo
    if estimated_tempo is None:
        estimated_tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length, trim=False)
        estimated_tempo = float(np.atleast_1d(estimated_tempo)[0])
        if estimated_tempo <= 0:
            estimated_tempo = 90.0

    step_duration = 60.0 / float(estimated_tempo) / 4.0
    expected_steps = bars * resolution

    raw_events: List[ReferenceEvent] = []
    for instrument in INSTRUMENTS:
        band_low, band_high = BANDS[instrument]
        band_audio = bandpass_filter(audio, sr, band_low, band_high)
        onset_frames, onset_env = detect_onsets(
            band_audio,
            sr=sr,
            hop_length=hop_length,
            threshold=0.13 if instrument == "repique" else 0.16,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        band_rms = librosa.feature.rms(y=band_audio, hop_length=hop_length, frame_length=2048)[0]

        for frame, time_s in zip(onset_frames, onset_times):
            step, offset_ms = quantize_time(float(time_s), step_duration)
            if step < 0 or step >= expected_steps:
                continue

            env_strength = float(onset_env[min(frame, len(onset_env) - 1)]) if onset_env.size else 0.0
            rms_strength = float(band_rms[min(frame, len(band_rms) - 1)]) if band_rms.size else 0.0
            energy = np.clip(0.65 * env_strength + 0.35 * (rms_strength / max(np.max(band_rms), 1e-6)), 0.0, 1.0)
            velocity = int(np.clip(18 + energy * 109, 1, 127))
            confidence = float(np.clip(0.5 * env_strength + 0.5 * energy, 0.0, 1.0))

            raw_events.append(
                ReferenceEvent(
                    instrument=instrument,
                    time=round(float(time_s), 4),
                    step=step,
                    bar=(step // resolution) + 1,
                    step_in_bar=(step % resolution) + 1,
                    offset_ms=round(float(offset_ms), 2),
                    velocity=velocity,
                    hit_type="pending",
                    confidence=round(confidence, 3),
                    band_energy=round(float(energy), 3),
                )
            )

    raw_events.sort(key=lambda event: (event.instrument, event.time))

    by_instrument: Dict[str, List[ReferenceEvent]] = {instrument: [] for instrument in INSTRUMENTS}
    for event in raw_events:
        by_instrument[event.instrument].append(event)

    final_events: List[ReferenceEvent] = []
    for instrument in INSTRUMENTS:
        instrument_events = by_instrument[instrument]
        for idx, event in enumerate(instrument_events):
            prev_gap = None if idx == 0 else event.time - instrument_events[idx - 1].time
            next_gap = None if idx == len(instrument_events) - 1 else instrument_events[idx + 1].time - event.time
            event.hit_type = infer_hit_type(instrument, event.velocity, prev_gap, next_gap)
            final_events.append(event)

    final_events.sort(key=lambda event: event.time)
    grid = events_to_grid(final_events, total_steps=expected_steps)
    summary = {
        "audio_path": audio_path,
        "sample_rate": sr,
        "tempo": round(float(estimated_tempo), 3),
        "bars": bars,
        "start": round(float(start), 4),
        "duration": None if duration is None else round(float(duration), 4),
        "resolution": resolution,
        "step_duration": round(step_duration, 6),
        "event_count": len(final_events),
        "counts_by_instrument": {instrument: sum(1 for event in final_events if event.instrument == instrument) for instrument in INSTRUMENTS},
    }
    return summary, final_events, grid


def save_outputs(summary: dict, events: List[ReferenceEvent], grid: np.ndarray, out_json: Optional[str], out_npz: Optional[str]) -> None:
    def ensure_parent(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if out_json:
        ensure_parent(out_json)
        payload = {
            "summary": summary,
            "events": [asdict(event) for event in events],
        }
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    if out_npz:
        ensure_parent(out_npz)
        np.savez_compressed(out_npz, data=grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a candombe reference audio and extract drum events.")
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument("--out-json", default="data/analysis/reference_events.json", help="Output JSON path.")
    parser.add_argument("--out-npz", default="data/processed/reference_from_audio.npz", help="Output NPZ path compatible with training.")
    parser.add_argument("--bars", type=int, default=4, help="Number of bars to analyze.")
    parser.add_argument("--tempo", type=float, default=None, help="Tempo in BPM. If omitted, it is estimated.")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds.")
    parser.add_argument("--resolution", type=int, default=16, help="Steps per bar.")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length for onset detection.")
    args = parser.parse_args()

    summary, events, grid = analyze_reference(
        audio_path=args.input,
        bars=args.bars,
        tempo=args.tempo,
        start=args.start,
        duration=args.duration,
        resolution=args.resolution,
        hop_length=args.hop_length,
    )
    save_outputs(summary, events, grid, args.out_json, args.out_npz)

    print(f"Analyzed: {args.input}")
    print(f"Tempo: {summary['tempo']} BPM")
    print(f"Events: {summary['event_count']}")
    print(f"Counts: {summary['counts_by_instrument']}")
    if args.out_json:
        print(f"JSON: {args.out_json}")
    if args.out_npz:
        print(f"NPZ: {args.out_npz}")


if __name__ == "__main__":
    main()
