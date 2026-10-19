"""Analyze a folder of candombe recordings into training pieces and a sample bank.

For every audio file: bar-aligned grid with microtiming (analyze_reference.py),
saved as .npz for training. Then the strongest hits per drum are cut into
one-shots and grouped by articulation, so the web player gets e.g. distinct
repique `madera` / `palo` / `mano` banks instead of one bank for every stroke.

Articulation grouping: the chico uses its position in the beat (known from the
grid); repique and piano are clustered on spectral centroid and decay time and
the clusters are named by ordering heuristics. Listen to the result and move
files between banks in the manifest if a cluster is wrong.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from analyze_reference import INSTRUMENTS, analyze_reference, save_outputs
from extract_reference_samples import extract_sample, score_event
from representation import ARTICULATIONS

# Synthetic one-shots from generate_demo_samples.py used when a bank is empty.
FALLBACK_SAMPLES = {
    "chico": {"mano": "/samples/chico_mano.wav", "palo": "/samples/chico_palo.wav"},
    "repique": {"palo": "/samples/repique_talk_open.wav", "mano": "/samples/repique_talk_support.wav"},
    "piano": {"abierto": "/samples/piano.wav"},
}


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


def sample_features(sample: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """(spectral centroid Hz, decay time to -20 dB in s, rms)."""
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=sample, sr=sr)))
    envelope = np.abs(sample)
    peak_idx = int(np.argmax(envelope))
    below = np.flatnonzero(envelope[peak_idx:] < envelope[peak_idx] * 0.1)
    decay = float(below[0] / sr) if below.size else float((sample.size - peak_idx) / sr)
    return centroid, decay, float(np.sqrt(np.mean(sample**2)))


def kmeans(features: np.ndarray, k: int, seed: int = 0, iterations: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (features - features.mean(0)) / np.maximum(features.std(0), 1e-9)
    centers = z[rng.choice(len(z), size=k, replace=False)]
    labels = np.zeros(len(z), dtype=int)
    for _ in range(iterations):
        labels = np.argmin(((z[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new_centers = np.array([z[labels == c].mean(0) if np.any(labels == c) else centers[c] for c in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def name_clusters(instrument: str, features: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
    """Map cluster ids to articulation names by ordering heuristics."""
    clusters = sorted(set(labels.tolist()))
    mean = {c: features[labels == c].mean(0) for c in clusters}  # centroid, decay, rms
    names: Dict[int, str] = {}
    if instrument == "repique":
        # madera: brightest and shortest; then palo: loudest; mano: the rest.
        remaining = list(clusters)
        madera = max(remaining, key=lambda c: mean[c][0] / max(mean[c][1], 1e-3))
        names[madera] = "madera"
        remaining.remove(madera)
        if remaining:
            palo = max(remaining, key=lambda c: mean[c][2])
            names[palo] = "palo"
            remaining.remove(palo)
        for c in remaining:
            names[c] = "mano"
    elif instrument == "piano":
        # madera: brightest; then stick strokes (abierto rings, apagado is short)
        # are brighter than hand strokes (mano rings, masa is dry and short).
        remaining = sorted(clusters, key=lambda c: mean[c][0])
        if len(remaining) >= 5:
            names[remaining.pop()] = "madera"
        half = len(remaining) // 2
        for group, (ringing, dry) in ((remaining[half:], ("abierto", "apagado")), (remaining[:half], ("mano", "masa"))):
            for rank, c in enumerate(sorted(group, key=lambda c: -mean[c][1])):
                names[c] = ringing if rank == 0 else dry
        for c in clusters:
            names.setdefault(c, "abierto")
    else:
        by_brightness = sorted(clusters, key=lambda c: mean[c][0])
        for rank, c in enumerate(by_brightness):
            names[c] = "mano" if rank == 0 else "madera" if rank == 2 else "palo"
    return names


def build_sample_bank(analyses: List[dict], sample_dir: Path, samples_per_instrument: int) -> Dict[str, Dict[str, List[str]]]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    cut: Dict[str, List[Tuple[str, np.ndarray, str]]] = {instrument: [] for instrument in INSTRUMENTS}
    rates: Dict[str, int] = {}

    for analysis in analyses:
        summary = analysis["summary"]
        stems = summary.get("stems") or {}
        source_name = Path(summary.get("audio_path") or next(iter(stems.values()))).stem
        loaded = {}
        for instrument in INSTRUMENTS:
            path = stems.get(instrument) or summary.get("audio_path")
            if path not in loaded:
                loaded[path] = librosa.load(
                    path, sr=None, mono=True, offset=float(summary.get("start", 0.0) or 0.0), duration=summary.get("duration")
                )
            audio, sr = loaded[path]
            for event in pick_top_events(analysis["events"], instrument, samples_per_instrument):
                sample = extract_sample(audio, sr, event, isolated=instrument in stems)
                if sample.size == 0:
                    continue
                out_name = f"{instrument}_{source_name}_{len(cut[instrument]) + 1:03d}.wav"
                sf.write(str(sample_dir / out_name), sample, sr)
                rates[out_name] = sr
                cut[instrument].append((out_name, sample, event.get("articulation", "")))

    bank: Dict[str, Dict[str, List[str]]] = {}
    for instrument in INSTRUMENTS:
        groups: Dict[str, List[str]] = {name: [] for name in ARTICULATIONS[instrument]}
        items = cut[instrument]
        if items:
            known = [item for item in items if item[2]]
            unknown = [item for item in items if not item[2]]
            for name, _, art in known:
                groups[art].append(name)
            if unknown:
                features = np.array([sample_features(sample, rates[name]) for name, sample, _ in unknown])
                k = min(len(ARTICULATIONS[instrument]), len(unknown))
                labels = kmeans(features, k) if k > 1 else np.zeros(len(unknown), dtype=int)
                names = name_clusters(instrument, features, labels)
                for (name, _, _), label in zip(unknown, labels):
                    groups[names[int(label)]].append(name)
        bank[instrument] = {art: [f"/samples/reference_bank/{name}" for name in files] for art, files in groups.items()}
    return bank


def build_frontend_manifest(bank: Dict[str, Dict[str, List[str]]]) -> dict:
    manifest = {}
    for instrument in INSTRUMENTS:
        entry = {}
        everything = []
        for art in ARTICULATIONS[instrument]:
            files = list(bank.get(instrument, {}).get(art, []))
            fallback = FALLBACK_SAMPLES.get(instrument, {}).get(art)
            if fallback:
                files.append(fallback)
            entry[art] = files
            everything.extend(files)
        entry["default"] = everything
        manifest[instrument] = entry
    manifest["clave"] = {"default": ["/samples/clave.wav"]}
    return manifest


def find_beats_file(audio_path: Path, beats_dir: Optional[str]) -> Optional[str]:
    if not beats_dir:
        return None
    for candidate in sorted(Path(beats_dir).glob(f"{audio_path.stem}*.csv")):
        return str(candidate)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="data/analysis/muestras/*.wav")
    parser.add_argument("--beats-dir", default="", help="Folder with <audio stem>*.csv beat annotations (time,bar.beat).")
    parser.add_argument("--analysis-dir", default="data/analysis/reference_corpus")
    parser.add_argument("--processed-dir", default="data/processed/reference_corpus")
    parser.add_argument("--sample-dir", default="web/samples/reference_bank")
    parser.add_argument("--samples-per-instrument", type=int, default=12)
    parser.add_argument("--max-duration", type=float, default=None, help="Analyze at most this many seconds per file.")
    parser.add_argument("--frontend-manifest", default="web/samples/manifest.json")
    parser.add_argument("--summary-json", default="data/analysis/reference_corpus/summary.json")
    args = parser.parse_args()

    audio_paths = sorted(Path().glob(args.input_glob))
    if not audio_paths:
        raise FileNotFoundError(f"No se encontraron audios para: {args.input_glob}")

    analysis_dir = Path(args.analysis_dir)
    processed_dir = Path(args.processed_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for audio_path in audio_paths:
        out_json = analysis_dir / f"{audio_path.stem}.json"
        out_npz = processed_dir / f"{audio_path.stem}.npz"
        summary, events, arrays = analyze_reference(
            audio_path=str(audio_path),
            beats_file=find_beats_file(audio_path, args.beats_dir),
            duration=args.max_duration,
        )
        save_outputs(summary, events, arrays, str(out_json), str(out_npz))
        print(f"{audio_path.name}: {summary['bars']} compases, {summary['tempo']} BPM, fase {summary['phase_source']} ({summary['phase_confidence']})")
        analyses.append({"summary": summary, "events": [event.__dict__ for event in events], "json": str(out_json), "npz": str(out_npz)})

    bank = build_sample_bank(analyses, Path(args.sample_dir), args.samples_per_instrument)
    frontend_manifest = build_frontend_manifest(bank)

    manifest_path = Path(args.frontend_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(frontend_manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_payload = {
        "sources": [str(path) for path in audio_paths],
        "analyses": [{key: value for key, value in analysis.items() if key != "events"} for analysis in analyses],
        "sample_bank": bank,
        "frontend_manifest": str(manifest_path),
    }
    Path(args.summary_json).write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Procesados {len(audio_paths)} audios")
    print(f"NPZ: {processed_dir}")
    print(f"Banco de samples: {args.sample_dir}")
    print(f"Manifest frontend: {manifest_path}")


if __name__ == "__main__":
    main()
