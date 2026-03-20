import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf

from analyze_reference import INSTRUMENTS, analyze_reference, save_outputs
from extract_reference_samples import extract_sample, score_event


VARIANT_MAP = {
    "chico": ("default", "mano", "palo"),
    "repique": (
        "default",
        "talk_open",
        "talk_support",
        "talk_answer",
        "talk_close",
        "call_open",
        "call_support",
        "call_close",
        "call_tail",
    ),
    "piano": ("default", "bass_open", "bass_muted", "stick_open", "stick_closed"),
}


def estimate_tempo_and_bars(audio_path: Path, resolution: int) -> Tuple[float, int, float]:
    info = sf.info(str(audio_path))
    estimate_duration = min(float(info.duration), 45.0)
    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=estimate_duration)
    if audio.size == 0:
        raise ValueError(f"Audio vacio: {audio_path}")

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=256, trim=False)
    tempo = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
    if tempo <= 0:
        tempo = 92.0

    duration = estimate_duration
    bar_duration = 60.0 / tempo * 4.0
    bars = max(4, int(round(duration / max(bar_duration, 1e-6))))
    return tempo, bars, duration


def pick_top_events(events: List[dict], instrument: str, count: int) -> List[dict]:
    scored = [event for event in events if event["instrument"] == instrument]
    scored.sort(key=score_event, reverse=True)
    selected: List[dict] = []
    selected_times: List[float] = []

    for event in scored:
        time_s = float(event["time"])
        if any(abs(time_s - prev_time) < 0.09 for prev_time in selected_times):
            continue
        selected.append(event)
        selected_times.append(time_s)
        if len(selected) >= count:
            break

    return selected


def build_sample_bank(
    analyses: List[dict],
    sample_dir: Path,
    samples_per_instrument: int,
) -> Dict[str, List[str]]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    bank: Dict[str, List[str]] = {instrument: [] for instrument in INSTRUMENTS}

    for analysis in analyses:
        audio_path = Path(analysis["summary"]["audio_path"])
        audio, sr = librosa.load(
            str(audio_path),
            sr=None,
            mono=True,
            offset=float(analysis["summary"].get("start", 0.0) or 0.0),
            duration=float(analysis["summary"].get("duration", 45.0) or 45.0),
        )
        events = analysis["events"]

        for instrument in INSTRUMENTS:
            for event in pick_top_events(events, instrument, samples_per_instrument):
                sample = extract_sample(audio, sr, event)
                if sample.size == 0:
                    continue
                out_name = f"{instrument}_{audio_path.stem}_{len(bank[instrument]) + 1:02d}.wav"
                out_path = sample_dir / out_name
                sf.write(str(out_path), sample, sr)
                bank[instrument].append(f"/samples/reference_bank/{out_name}")

    return bank


def build_frontend_manifest(bank: Dict[str, List[str]]) -> dict:
    chico_bank = bank.get("chico", [])
    repique_bank = bank.get("repique", [])
    piano_bank = bank.get("piano", [])
    chico_palo_bank = [
        source for source in chico_bank
        if any(token in source for token in ("_04", "_06", "_07", "_01", "_05"))
    ] or chico_bank[:]
    chico_mano_bank = [
        source for source in chico_bank
        if source not in chico_palo_bank
    ] or chico_bank[:]

    return {
        "chico": {
            "default": [*chico_palo_bank, *chico_mano_bank, "/samples/chico_palo.wav"],
            "mano": [*chico_mano_bank, "/samples/chico_mano.wav"],
            "palo": [*chico_palo_bank, "/samples/chico_palo.wav"],
        },
        "repique": {
            variant: [*repique_bank, f"/samples/{fallback}"]
            for variant, fallback in {
                "default": "repique_talk_support.wav",
                "talk_open": "repique_talk_open.wav",
                "talk_support": "repique_talk_support.wav",
                "talk_answer": "repique_talk_answer.wav",
                "talk_close": "repique_talk_close.wav",
                "call_open": "repique_call_open.wav",
                "call_support": "repique_call_support.wav",
                "call_close": "repique_call_close.wav",
                "call_tail": "repique_call_tail.wav",
            }.items()
        },
        "piano": {
            "default": [*piano_bank, "/samples/piano.wav"],
            "bass_open": [*piano_bank, "/samples/piano.wav"],
            "bass_muted": [*piano_bank, "/samples/piano.wav"],
            "stick_open": [*piano_bank, "/samples/piano.wav"],
            "stick_closed": [*piano_bank, "/samples/piano.wav"],
        },
        "clave": {
            "default": ["/samples/clave.wav"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="data/analysis/muestras/*.wav")
    parser.add_argument("--analysis-dir", default="data/analysis/reference_corpus")
    parser.add_argument("--processed-dir", default="data/processed/reference_corpus")
    parser.add_argument("--sample-dir", default="web/samples/reference_bank")
    parser.add_argument("--samples-per-instrument", type=int, default=3)
    parser.add_argument("--frontend-manifest", default="web/samples/manifest.json")
    parser.add_argument("--summary-json", default="data/analysis/reference_corpus/summary.json")
    args = parser.parse_args()

    wav_paths = sorted(Path().glob(args.input_glob))
    if not wav_paths:
        raise FileNotFoundError(f"No se encontraron WAV para: {args.input_glob}")

    analysis_dir = Path(args.analysis_dir)
    processed_dir = Path(args.processed_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for wav_path in wav_paths:
        tempo, bars, duration = estimate_tempo_and_bars(wav_path, resolution=16)
        out_json = analysis_dir / f"{wav_path.stem}.json"
        out_npz = processed_dir / f"{wav_path.stem}.npz"
        summary, events, grid = analyze_reference(
            audio_path=str(wav_path),
            bars=bars,
            tempo=tempo,
            start=0.0,
            duration=duration,
            resolution=16,
            hop_length=256,
        )
        save_outputs(summary, events, grid, str(out_json), str(out_npz))
        analyses.append(
            {
                "summary": summary,
                "events": [event.__dict__ for event in events],
                "json": str(out_json),
                "npz": str(out_npz),
            }
        )

    bank = build_sample_bank(analyses, Path(args.sample_dir), args.samples_per_instrument)
    frontend_manifest = build_frontend_manifest(bank)

    manifest_path = Path(args.frontend_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(frontend_manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_payload = {
        "sources": [str(path) for path in wav_paths],
        "analyses": analyses,
        "sample_bank": bank,
        "frontend_manifest": str(manifest_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Procesados {len(wav_paths)} audios")
    print(f"Analisis JSON: {analysis_dir}")
    print(f"NPZ: {processed_dir}")
    print(f"Banco de samples: {args.sample_dir}")
    print(f"Manifest frontend: {manifest_path}")


if __name__ == "__main__":
    main()
