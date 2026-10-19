"""Extract chico / repique / piano onsets from audio onto a bar-aligned 16th grid.

The grid follows the actual beats (annotated or tracked), so tempo drift and the
characteristic acceleration of a llamada do not smear the quantization, and
each onset keeps its microtiming as a fraction of the local 16th step.

Bar alignment, in order of preference:
  1. --beats-file with bar.beat labels (Candombe dataset CSV format: `time,bar.beat`),
  2. --downbeat <seconds> given by hand,
  3. estimated phase: the chico never strikes the first 16th of a beat, and the
     piano + repique (clave) accents follow the bar template of candombe_rules.
     This is a heuristic; check `summary.phase_confidence`.

Instruments come from full-band stems (--stems, best) or, for a mix, from band
filters (heuristic: the drums overlap a lot in frequency).
"""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from candombe_rules import CLAVE_STEPS, PIANO_POOL, REPIQUE_POOL
from representation import N_INSTR, STEPS_PER_BAR, STEPS_PER_BEAT, articulation_id, save_piece

INSTRUMENTS = ("chico", "repique", "piano")
BANDS = {
    "piano": (45.0, 240.0),
    "repique": (240.0, 1800.0),
    "chico": (1800.0, 8000.0),
}
IDX_MAP = {"chico": 0, "repique": 1, "piano": 2}
ONSET_THRESHOLD = {"chico": 0.16, "repique": 0.13, "piano": 0.16}
BLEED_RATIO = 0.35
SR = 22050


@dataclass
class ReferenceEvent:
    instrument: str
    time: float
    step: int
    bar: int
    step_in_bar: int
    offset_ms: float
    offset_frac: float
    velocity: int
    hit_type: str
    articulation: str
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


def refine_onsets(band_audio: np.ndarray, sr: int, times: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Move each detected onset to its attack start, at sample resolution.

    Frame-based onset detection fires one or two hops after the attack, which
    would add a constant fake delay to every microtiming offset. Within a short
    window we find the attack peak and walk back to `threshold` of it.
    """
    # The analytic-signal envelope does not ripple at the drum's fundamental, so
    # low drums (piano) are not detected half a period late.
    envelope = np.abs(hilbert(band_audio))
    refined = np.empty_like(times)
    for k, time_s in enumerate(times):
        lo = max(0, int((time_s - 0.03) * sr))
        hi = min(envelope.size, int((time_s + 0.03) * sr))
        if hi - lo < 2:
            refined[k] = time_s
            continue
        peak = lo + int(np.argmax(envelope[lo:hi]))
        below = np.flatnonzero(envelope[lo : peak + 1] < threshold * envelope[peak])
        refined[k] = (lo + (below[-1] + 1 if below.size else 0)) / sr
    return refined


def infer_hit_type(instrument: str, velocity: int, prev_gap: Optional[float], next_gap: Optional[float]) -> str:
    vel_norm = velocity / 127.0
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


# --- beat grid --------------------------------------------------------------------------

def load_beat_annotations(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Rows `time,bar.beat` -> (beat times, beat number in bar starting at 1)."""
    times, beats = [], []
    with open(path, "r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0].strip():
                continue
            times.append(float(row[0]))
            label = row[1].strip()
            beats.append(int(label.split(".")[1]) if "." in label else int(float(label)))
    return np.asarray(times), np.asarray(beats)


def track_beats(audio: np.ndarray, sr: int, hop_length: int, tempo_hint: Optional[float]) -> np.ndarray:
    kwargs = {"bpm": tempo_hint} if tempo_hint else {}
    tempo, frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length, trim=False, **kwargs)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    if times.size >= 4:
        return times
    bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None and float(np.atleast_1d(tempo)[0]) > 0 else (tempo_hint or 90.0)
    return np.arange(0.0, audio.size / sr, 60.0 / bpm)


def subdivide(beat_times: np.ndarray) -> np.ndarray:
    """16th-note times: each beat interval split in 4, last beat extrapolated."""
    if beat_times.size < 2:
        raise ValueError("Need at least two beats to build a grid")
    extended = np.append(beat_times, beat_times[-1] + (beat_times[-1] - beat_times[-2]))
    steps = [np.linspace(a, b, STEPS_PER_BEAT, endpoint=False) for a, b in zip(extended[:-1], extended[1:])]
    return np.concatenate(steps + [extended[-1:]])


def snap(times: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest grid index for each time and the offset as a fraction of the local step."""
    idx = np.clip(np.searchsorted(grid, times), 1, grid.size - 1)
    left_closer = (times - grid[idx - 1]) < (grid[idx] - times)
    idx = np.where(left_closer, idx - 1, idx)
    local_step = np.diff(grid, append=grid[-1] + (grid[-1] - grid[-2]))
    offsets = (times - grid[idx]) / np.maximum(local_step[idx], 1e-6)
    return idx, offsets


def _bar_template() -> np.ndarray:
    """Expected onset weight per 16th position for piano and repique (2, 16)."""
    template = np.zeros((2, STEPS_PER_BAR))
    for pattern in PIANO_POOL:
        template[1] += [hit["velocity"] if hit else 0 for hit in pattern]
    for pattern in REPIQUE_POOL:
        template[0] += [hit["velocity"] if hit else 0 for hit in pattern]
    template[0, list(CLAVE_STEPS)] += template[0].max() * 0.5
    template -= template.mean(axis=1, keepdims=True)
    return template / np.linalg.norm(template, axis=1, keepdims=True)


def estimate_phase(steps_by_inst: Dict[str, np.ndarray], weights_by_inst: Dict[str, np.ndarray]) -> Tuple[int, float]:
    """Grid shift (0..15) that puts step 0 on a downbeat, and a 0..1 confidence."""
    hist = {inst: np.bincount(steps_by_inst[inst] % STEPS_PER_BAR, weights=weights_by_inst[inst], minlength=STEPS_PER_BAR) for inst in INSTRUMENTS}
    template = _bar_template()
    scores = np.zeros(STEPS_PER_BAR)
    for shift in range(STEPS_PER_BAR):
        chico = np.roll(hist["chico"], -shift)
        chico_on_beat = chico[0::STEPS_PER_BEAT].sum() / max(chico.sum(), 1e-9)
        bar_score = 0.0
        for row, inst in enumerate(("repique", "piano")):
            rolled = np.roll(hist[inst], -shift)
            centered = rolled - rolled.mean()
            norm = np.linalg.norm(centered)
            if norm > 0:
                bar_score += float(centered @ template[row]) / norm
        scores[shift] = bar_score - 2.0 * chico_on_beat
    order = np.argsort(scores)[::-1]
    margin = scores[order[0]] - scores[order[1]]
    return int(order[0]), float(np.clip(margin / 0.5, 0.0, 1.0))


# --- analysis ---------------------------------------------------------------------------

def _load(path: str, start: float, duration: Optional[float]) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=duration)
    if audio.size == 0:
        raise ValueError(f"The selected audio segment is empty: {path}")
    return librosa.util.normalize(audio)


def analyze_reference(
    audio_path: Optional[str] = None,
    stems: Optional[Dict[str, str]] = None,
    beats_file: Optional[str] = None,
    downbeat: Optional[float] = None,
    bars: Optional[int] = None,
    tempo: Optional[float] = None,
    start: float = 0.0,
    duration: Optional[float] = None,
    hop_length: int = 256,
) -> Tuple[dict, List[ReferenceEvent], Dict[str, np.ndarray]]:
    if not audio_path and not stems:
        raise ValueError("Pass audio_path (a mix) or stems (one file per drum)")

    sources = {inst: _load(stems[inst], start, duration) for inst in INSTRUMENTS if stems and inst in stems}
    mix = _load(audio_path, start, duration) if audio_path else np.sum([s[: min(map(len, sources.values()))] for s in sources.values()], axis=0)

    beat_labels = None
    if beats_file:
        beat_times, beat_labels = load_beat_annotations(beats_file)
        beat_times = beat_times - start
        keep = beat_times >= 0
        beat_times, beat_labels = beat_times[keep], beat_labels[keep]
    else:
        beat_times = track_beats(mix, SR, hop_length, tempo)
    grid = subdivide(beat_times)

    onsets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for instrument in INSTRUMENTS:
        if instrument in sources:
            band_audio = sources[instrument]
        else:
            band_audio = bandpass_filter(mix, SR, *BANDS[instrument])
        frames, env = detect_onsets(band_audio, SR, hop_length, ONSET_THRESHOLD[instrument])
        times = refine_onsets(band_audio, SR, librosa.frames_to_time(frames, sr=SR, hop_length=hop_length))
        rms = librosa.feature.rms(y=band_audio, hop_length=hop_length, frame_length=2048)[0]
        env_strength = env[np.minimum(frames, env.size - 1)] if env.size else np.zeros(frames.size)
        rms_strength = rms[np.minimum(frames, rms.size - 1)] / max(float(np.max(rms)) if rms.size else 0.0, 1e-6)
        energy = np.clip(0.65 * env_strength + 0.35 * rms_strength, 0.0, 1.0)
        inside = (times >= grid[0] - 0.05) & (times <= grid[-1])
        if instrument not in sources and rms_strength.size:
            # In a mix, strokes of the other drums bleed into this band as weak
            # onsets; keep those reasonably close to this drum's own strong hits.
            inside &= rms_strength >= BLEED_RATIO * np.percentile(rms_strength, 90)
        onsets[instrument] = (times[inside], energy[inside], env_strength[inside])

    snapped = {inst: snap(onsets[inst][0], grid) for inst in INSTRUMENTS}

    phase_confidence = 1.0
    if beat_labels is not None:
        first_downbeat = int(np.flatnonzero(beat_labels == 1)[0]) if np.any(beat_labels == 1) else 0
        shift = first_downbeat * STEPS_PER_BEAT
        phase_source = "annotations"
    elif downbeat is not None:
        shift = int(snap(np.array([downbeat - start]), grid)[0][0])
        phase_source = "manual"
    else:
        shift, phase_confidence = estimate_phase({i: snapped[i][0] for i in INSTRUMENTS}, {i: onsets[i][1] for i in INSTRUMENTS})
        phase_source = "estimated"

    total_steps = ((grid.size - 1 - shift) // STEPS_PER_BAR) * STEPS_PER_BAR
    if bars:
        total_steps = min(total_steps, bars * STEPS_PER_BAR)
    if total_steps <= 0:
        raise ValueError("Audio too short for one full bar after alignment")

    velocities = np.zeros((total_steps, N_INSTR), dtype=np.uint8)
    offsets = np.full((total_steps, N_INSTR), np.nan, dtype=np.float32)
    articulation = np.full((total_steps, N_INSTR), -1, dtype=np.int8)
    events: List[ReferenceEvent] = []
    for instrument in INSTRUMENTS:
        inst = IDX_MAP[instrument]
        times, energy, env_strength = onsets[instrument]
        idx, offs = snapped[instrument]
        step_list = idx - shift
        for k in range(times.size):
            step = int(step_list[k])
            if not 0 <= step < total_steps:
                continue
            velocity = int(np.clip(18 + energy[k] * 109, 1, 127))
            art_name = ""
            if instrument == "chico" and step % STEPS_PER_BEAT:
                art_name = "mano" if step % STEPS_PER_BEAT == 1 else "palo"
            if velocity > velocities[step, inst]:
                velocities[step, inst] = velocity
                offsets[step, inst] = float(offs[k])
                articulation[step, inst] = articulation_id(instrument, art_name) if art_name else -1
            local_step = grid[idx[k] + 1] - grid[idx[k]] if idx[k] + 1 < grid.size else grid[idx[k]] - grid[idx[k] - 1]
            events.append(
                ReferenceEvent(
                    instrument=instrument,
                    time=round(float(times[k]), 4),
                    step=step,
                    bar=step // STEPS_PER_BAR + 1,
                    step_in_bar=step % STEPS_PER_BAR + 1,
                    offset_ms=round(float(offs[k] * local_step * 1000.0), 2),
                    offset_frac=round(float(offs[k]), 4),
                    velocity=velocity,
                    hit_type="pending",
                    articulation=art_name,
                    confidence=round(float(np.clip(0.5 * env_strength[k] + 0.5 * energy[k], 0.0, 1.0)), 3),
                    band_energy=round(float(energy[k]), 3),
                )
            )

    for instrument in INSTRUMENTS:
        inst_events = sorted((e for e in events if e.instrument == instrument), key=lambda e: e.time)
        for i, event in enumerate(inst_events):
            prev_gap = None if i == 0 else event.time - inst_events[i - 1].time
            next_gap = None if i == len(inst_events) - 1 else inst_events[i + 1].time - event.time
            event.hit_type = infer_hit_type(instrument, event.velocity, prev_gap, next_gap)
    events.sort(key=lambda e: e.time)

    beat_intervals = np.diff(beat_times)
    median_tempo = float(60.0 / np.median(beat_intervals)) if beat_intervals.size else (tempo or 0.0)
    summary = {
        "audio_path": audio_path,
        "stems": stems,
        "sample_rate": SR,
        "tempo": round(median_tempo, 3),
        "tempo_start": round(float(60.0 / np.median(beat_intervals[:8])), 3) if beat_intervals.size else None,
        "tempo_end": round(float(60.0 / np.median(beat_intervals[-8:])), 3) if beat_intervals.size else None,
        "bars": total_steps // STEPS_PER_BAR,
        "start": round(float(start), 4),
        "duration": None if duration is None else round(float(duration), 4),
        "resolution": STEPS_PER_BAR,
        "phase_source": phase_source,
        "phase_confidence": round(phase_confidence, 3),
        "event_count": len(events),
        "counts_by_instrument": {inst: sum(1 for e in events if e.instrument == inst) for inst in INSTRUMENTS},
    }
    return summary, events, {"velocities": velocities, "offsets_frac": offsets, "articulation": articulation}


def save_outputs(summary: dict, events: List[ReferenceEvent], arrays: Dict[str, np.ndarray], out_json: Optional[str], out_npz: Optional[str], style: Optional[str] = None) -> None:
    if out_json:
        parent = os.path.dirname(out_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "events": [asdict(e) for e in events]}, handle, indent=2, ensure_ascii=True)
    if out_npz:
        save_piece(out_npz, arrays["velocities"], arrays["offsets_frac"], arrays["articulation"], tempo=summary["tempo"], style=style)


def parse_stems(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    stems = dict(part.split("=", 1) for part in value.split(",") if "=" in part)
    unknown = set(stems) - set(INSTRUMENTS)
    if unknown:
        raise ValueError(f"Unknown stems {unknown}; use {INSTRUMENTS}")
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a candombe recording into a bar-aligned grid with microtiming.")
    parser.add_argument("--input", help="Mixed audio file.")
    parser.add_argument("--stems", help="One file per drum: chico=a.wav,repique=b.wav,piano=c.wav")
    parser.add_argument("--beats-file", help="Beat annotations CSV (time,bar.beat).")
    parser.add_argument("--downbeat", type=float, default=None, help="Time in seconds of a downbeat (beat 1).")
    parser.add_argument("--out-json", default="data/analysis/reference_events.json")
    parser.add_argument("--out-npz", default="data/processed/reference_from_audio.npz")
    parser.add_argument("--bars", type=int, default=None, help="Keep at most this many bars.")
    parser.add_argument("--tempo", type=float, default=None, help="Tempo hint in BPM for the beat tracker.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--style", default=None, help="Optional style label (montevideo, cuareim, ansina, cordon).")
    args = parser.parse_args()

    summary, events, arrays = analyze_reference(
        audio_path=args.input,
        stems=parse_stems(args.stems),
        beats_file=args.beats_file,
        downbeat=args.downbeat,
        bars=args.bars,
        tempo=args.tempo,
        start=args.start,
        duration=args.duration,
        hop_length=args.hop_length,
    )
    save_outputs(summary, events, arrays, args.out_json, args.out_npz, style=args.style)

    print(f"Tempo: {summary['tempo']} BPM ({summary['tempo_start']} -> {summary['tempo_end']})")
    print(f"Bars: {summary['bars']}  phase: {summary['phase_source']} (confidence {summary['phase_confidence']})")
    print(f"Counts: {summary['counts_by_instrument']}")
    if summary["phase_source"] == "estimated" and summary["phase_confidence"] < 0.3:
        print("WARNING: low bar-phase confidence; pass --downbeat or --beats-file")


if __name__ == "__main__":
    main()
