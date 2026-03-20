import os
import numpy as np
from typing import Tuple
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

try:
    from .generate import load_model, sample, events_from_array, INPUT_DIM
except ImportError:
    from generate import load_model, sample, events_from_array, INPUT_DIM

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CHICO_BLOCK = slice(0, 4)
CHICO_BAR_PATTERN = (
    None, "mano", "palo", "palo",
    None, "mano", "palo", "palo",
    None, "mano", "palo", "palo",
    None, "mano", "palo", "palo",
)
CHICO_DEFAULT_BUCKET = {"mano": 2, "palo": 1}
CHICO_STEP_TO_HIT = {
    step_in_bar: hit_type
    for step_in_bar, hit_type in enumerate(CHICO_BAR_PATTERN)
    if hit_type is not None
}
REPIQUE_BLOCK = slice(4, 8)
CLAVE_BAR_STEPS = (0, 3, 6, 10, 12)
CLAVE_BAR_VELOCITIES = (92, 74, 86, 74, 108)
REPIQUE_TALK_STEPS = tuple(range(16))
REPIQUE_TALK_HITS = (
    "talk_open", "talk_support", "talk_answer", "talk_close",
    "talk_open", "talk_support", "talk_answer", "talk_close",
    "talk_open", "talk_support", "talk_answer", "talk_close",
    "talk_open", "talk_support", "talk_answer", "talk_close",
)
REPIQUE_CALL_STEPS = (11, 13, 14, 15)
REPIQUE_CALL_HITS = ("call_open", "call_support", "call_close", "call_tail")
REPIQUE_DEFAULT_BUCKET = {
    "talk_open": 2,
    "talk_support": 1,
    "talk_answer": 1,
    "talk_close": 3,
    "call_open": 3,
    "call_support": 2,
    "call_close": 2,
    "call_tail": 1,
}
REPIQUE_TALK_STEP_TO_HIT = dict(zip(REPIQUE_TALK_STEPS, REPIQUE_TALK_HITS))
REPIQUE_STEP_TO_HIT = dict(zip(REPIQUE_CALL_STEPS, REPIQUE_CALL_HITS))
PIANO_BLOCK = slice(8, 12)
PIANO_CADENCE_STEPS = (0, 3, 8, 11)
PIANO_CADENCE_HITS = ("bass_open", "bass_muted", "bass_open", "bass_muted")
PIANO_STICK_STEPS = (10, 14)
PIANO_STICK_HITS = ("stick_open", "stick_closed")
PIANO_CALL_STEPS = (1, 9, 14)
PIANO_CALL_HITS = ("stick_open", "stick_open", "stick_closed")
PIANO_CLOSE_STEPS = (9, 13, 15)
PIANO_CLOSE_HITS = ("stick_open", "stick_open", "stick_closed")
PIANO_DEFAULT_BUCKET = {"bass_open": 3, "bass_muted": 1, "stick_open": 2, "stick_closed": 1}
PIANO_STEP_TO_HIT = dict(zip(PIANO_CADENCE_STEPS, PIANO_CADENCE_HITS))

MODEL_PATH = os.environ.get("MODEL_PATH", "models/candombe_epoch5.pt")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception:
        model = None


def _seed_pattern(kind: str, steps: int = 16):
    arr = np.zeros((steps, INPUT_DIM), dtype=np.float32)
    kind = (kind or "random").lower()

    if kind in ("clave", "base"):
        for i in range(steps):
            if i % 4 == 0:
                arr[i, 0] = 1.0  # chico
            if i in (0, 8):
                arr[i, 8] = 1.0  # piano
    elif kind in ("energia", "energy"):
        for i in range(steps):
            if i % 2 == 0:
                arr[i, 0] = 1.0
            if i in (3, 7, 11, 15):
                arr[i, 4] = 1.0  # repique accents
            if i in (0, 8):
                arr[i, 9] = 1.0  # piano stronger bucket

    return arr


def _pick_chico_bucket(chico_view: np.ndarray, step: int, hit_type: str) -> int:
    # Keep chico fully locked so the cadence never drifts from
    # the steady "ta ra ca" pulse, regardless of model suggestions.
    return CHICO_DEFAULT_BUCKET[hit_type]


def _section_for_bar(bar_index: int, total_bars: int) -> str:
    if total_bars <= 1:
        return "cierre"

    position = bar_index / max(total_bars, 1)
    if position < 0.25:
        return "intro"
    if position < 0.50:
        return "llamada_repique"
    if position < 0.75:
        return "llamada_piano"
    return "cierre"


def _apply_chico_cadence(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < CHICO_BLOCK.stop:
        return arr

    out = arr.copy()
    chico_view = out[:, CHICO_BLOCK].copy()
    out[:, CHICO_BLOCK] = 0.0

    for bar_start in range(0, out.shape[0], 16):
        for step_in_bar, hit_type in enumerate(CHICO_BAR_PATTERN):
            if hit_type is None:
                continue
            step = bar_start + step_in_bar
            if step >= out.shape[0]:
                break
            bucket = _pick_chico_bucket(chico_view, step, hit_type)
            out[step, bucket] = 1.0

    return out


def _pick_piano_bucket(piano_view: np.ndarray, step: int, hit_type: str) -> int:
    default_bucket = PIANO_DEFAULT_BUCKET[hit_type]
    best_bucket = -1
    for candidate in range(max(0, step - 1), min(piano_view.shape[0], step + 2)):
        block = piano_view[candidate]
        if block.sum() > 0:
            best_bucket = max(best_bucket, int(block.argmax()))
    return max(default_bucket, best_bucket)


def _pick_repique_bucket(repique_view: np.ndarray, step: int, hit_type: str) -> int:
    default_bucket = REPIQUE_DEFAULT_BUCKET[hit_type]
    best_bucket = -1
    for candidate in range(max(0, step - 1), min(repique_view.shape[0], step + 2)):
        block = repique_view[candidate]
        if block.sum() > 0:
            best_bucket = max(best_bucket, int(block.argmax()))
    return max(default_bucket, best_bucket)


def _repique_hit_type_for_step(step: int, total_bars: int, density: float) -> str:
    step_in_bar = step % 16
    bar_index = step // 16
    section = _section_for_bar(bar_index, total_bars)

    if section in ("llamada_repique", "cierre"):
        call_steps = REPIQUE_CALL_STEPS if density >= 0.48 or section == "cierre" else (13, 15)
        if step_in_bar in call_steps:
            return REPIQUE_STEP_TO_HIT[step_in_bar]

    return REPIQUE_TALK_STEP_TO_HIT.get(step_in_bar, "talk_support")


def _apply_repique_talk(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < REPIQUE_BLOCK.stop:
        return arr

    out = arr.copy()
    repique_view = out[:, REPIQUE_BLOCK].copy()
    out[:, REPIQUE_BLOCK] = 0.0

    for bar_start in range(0, out.shape[0], 16):
        for step_in_bar, hit_type in zip(REPIQUE_TALK_STEPS, REPIQUE_TALK_HITS):
            step = bar_start + step_in_bar
            if step >= out.shape[0]:
                break
            bucket = _pick_repique_bucket(repique_view, step, hit_type)
            out[step, REPIQUE_BLOCK.start + bucket] = 1.0

    return out


def _stabilize_repique(arr: np.ndarray, density: float, total_bars: int) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < REPIQUE_BLOCK.stop:
        return arr

    out = arr.copy()
    repique_view = out[:, REPIQUE_BLOCK].copy()
    out[:, REPIQUE_BLOCK] = 0.0
    syncopated_steps = {1, 2, 4, 5, 7, 9, 11, 12, 14, 15}
    anchor_steps = (1, 4, 9, 12)

    for bar_start in range(0, out.shape[0], 16):
        bar_slice = repique_view[bar_start : bar_start + 16]
        if bar_slice.size == 0:
            continue

        target_hits = int(round(5 + density * 3))
        candidate_info = []
        for step_in_bar in range(bar_slice.shape[0]):
            block = bar_slice[step_in_bar]
            strength = float(block.max())
            if strength <= 0:
                continue
            bucket = int(block.argmax())
            sync_bonus = 0.12 if step_in_bar in syncopated_steps else 0.0
            candidate_info.append((strength + sync_bonus, step_in_bar, bucket))

        selected = {}
        for anchor_step in anchor_steps:
            if anchor_step >= bar_slice.shape[0]:
                continue
            anchor_candidates = [item for item in candidate_info if abs(item[1] - anchor_step) <= 1]
            if anchor_candidates:
                _, step_in_bar, bucket = max(anchor_candidates, key=lambda item: (item[0], -abs(item[1] - anchor_step)))
                selected[step_in_bar] = bucket

        for _, step_in_bar, bucket in sorted(candidate_info, key=lambda item: item[0], reverse=True):
            if step_in_bar in selected:
                continue
            if len(selected) >= target_hits:
                break
            if (step_in_bar - 1 in selected) and (step_in_bar - 2 in selected):
                continue
            selected[step_in_bar] = bucket

        if len(selected) < 5:
            fallback_steps = (1, 4, 9, 11, 14)
            for step_in_bar in fallback_steps:
                if step_in_bar >= bar_slice.shape[0] or step_in_bar in selected:
                    continue
                hit_type = _repique_hit_type_for_step(bar_start + step_in_bar, total_bars, density)
                selected[step_in_bar] = REPIQUE_DEFAULT_BUCKET[hit_type]
                if len(selected) >= 5:
                    break

        for step_in_bar, bucket in selected.items():
            out[bar_start + step_in_bar, REPIQUE_BLOCK.start + bucket] = 1.0

    return out


def _apply_repique_closure(arr: np.ndarray, density: float, total_bars: int) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < REPIQUE_BLOCK.stop:
        return arr

    out = arr.copy()
    repique_view = out[:, REPIQUE_BLOCK].copy()

    for bar_start in range(0, out.shape[0], 16):
        bar_index = bar_start // 16
        section = _section_for_bar(bar_index, total_bars)
        if section not in ("llamada_repique", "cierre"):
            continue

        call_steps = REPIQUE_CALL_STEPS if density >= 0.48 or section == "cierre" else (13, 15)
        for step_in_bar in call_steps:
            step = bar_start + step_in_bar
            if step >= out.shape[0]:
                break
            hit_type = REPIQUE_STEP_TO_HIT[step_in_bar]
            bucket = _pick_repique_bucket(repique_view, step, hit_type)
            out[step, REPIQUE_BLOCK] = 0.0
            out[step, REPIQUE_BLOCK.start + bucket] = 1.0

    return out


def _build_clave_events(step_duration: float, total_bars: int) -> list:
    events = []
    for bar_index in range(total_bars):
        section = _section_for_bar(bar_index, total_bars)
        bar_start = bar_index * 16
        for step_in_bar, velocity in zip(CLAVE_BAR_STEPS, CLAVE_BAR_VELOCITIES):
            step = bar_start + step_in_bar
            events.append(
                {
                    "time": step * step_duration,
                    "instrument": "clave",
                    "velocity": velocity,
                    "step": step,
                    "section": section,
                    "hit_type": "palmas",
                }
            )
    return events


def _piano_bar_pattern(section: str) -> Tuple[Tuple[int, str], ...]:
    pattern = list(zip(PIANO_CADENCE_STEPS, PIANO_CADENCE_HITS))

    if section == "llamada_piano":
        pattern.extend(zip(PIANO_STICK_STEPS, PIANO_STICK_HITS))
        pattern.extend(zip(PIANO_CALL_STEPS, PIANO_CALL_HITS))
    elif section == "cierre":
        pattern.extend(zip(PIANO_STICK_STEPS, PIANO_STICK_HITS))
        pattern.extend(zip(PIANO_CLOSE_STEPS, PIANO_CLOSE_HITS))

    return tuple(pattern)


def _piano_hit_type_for_step(step: int, total_bars: int) -> str:
    step_in_bar = step % 16
    bar_index = step // 16
    section = _section_for_bar(bar_index, total_bars)

    hits_by_step = {pattern_step: hit_type for pattern_step, hit_type in _piano_bar_pattern(section)}
    return hits_by_step.get(step_in_bar, "bass_muted")


def _apply_piano_cadence(arr: np.ndarray, total_bars: int) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < PIANO_BLOCK.stop:
        return arr

    out = arr.copy()
    piano_view = out[:, PIANO_BLOCK].copy()
    out[:, PIANO_BLOCK] = 0.0

    for bar_start in range(0, out.shape[0], 16):
        bar_index = bar_start // 16
        section = _section_for_bar(bar_index, total_bars)
        for step_in_bar, hit_type in _piano_bar_pattern(section):
            step = bar_start + step_in_bar
            if step >= out.shape[0]:
                break
            bucket = _pick_piano_bucket(piano_view, step, hit_type)
            out[step, PIANO_BLOCK.start + bucket] = 1.0

    return out


def _annotate_events(events: list, step_duration: float, total_bars: int, density: float) -> list:
    annotated = []
    for event in events:
        enriched = dict(event)
        step = int(round(float(enriched["time"]) / step_duration))
        bar_index = step // 16
        enriched["step"] = step
        enriched["section"] = _section_for_bar(bar_index, total_bars)
        if enriched["instrument"] == "chico":
            enriched["hit_type"] = CHICO_STEP_TO_HIT.get(step % 16, "palo")
        elif enriched["instrument"] == "repique":
            enriched["hit_type"] = _repique_hit_type_for_step(step, total_bars, density)
        elif enriched["instrument"] == "piano":
            enriched["hit_type"] = _piano_hit_type_for_step(step, total_bars)
        else:
            enriched["hit_type"] = "stroke"
        annotated.append(enriched)
    return annotated


@app.get("/generate")
def generate(
    tempo: int = 90,
    bars: int = 4,
    seed: str = "clave",
    temperature: float = 0.9,
    density: float = 0.55,
):
    tempo = max(50, min(220, tempo))
    bars = max(4, min(16, bars))
    temperature = max(0.3, min(1.8, temperature))
    density = max(0.05, min(0.95, density))

    step_duration = 60.0 / tempo / 4  # 16th-note grid
    steps = bars * 16

    seed_arr = _seed_pattern(seed, steps=16)

    if model is not None:
        arr = sample(
            model,
            seed_arr,
            length_steps=steps,
            temperature=temperature,
            density=density,
        )
        arr = arr[seed_arr.shape[0] : seed_arr.shape[0] + steps]
    else:
        # Rule-based fallback when there is no checkpoint.
        arr = np.zeros((steps, INPUT_DIM), dtype=np.float32)
        for i in range(steps):
            if i % 4 == 0:
                arr[i, 0] = 1.0  # chico pulse
            if i in (0, 8):
                arr[i, 8] = 1.0  # piano anchors
            if i % 8 in (3, 7):
                arr[i, 4] = 1.0  # repique pickup

    arr = _apply_chico_cadence(arr)
    arr = _stabilize_repique(arr, density=density, total_bars=bars) if model is not None else _apply_repique_talk(arr)
    arr = _apply_repique_closure(arr, density=density, total_bars=bars)
    arr = _apply_piano_cadence(arr, total_bars=bars)
    events = _annotate_events(
        events_from_array(arr, step_duration=step_duration),
        step_duration=step_duration,
        total_bars=bars,
        density=density,
    )
    events.extend(_build_clave_events(step_duration=step_duration, total_bars=bars))
    events.sort(key=lambda event: (float(event["time"]), event["instrument"]))
    return {
        "tempo": tempo,
        "bars": bars,
        "step_duration": step_duration,
        "model_loaded": model is not None,
        "structure": [_section_for_bar(bar_index, bars) for bar_index in range(bars)],
        "events": events,
    }


# Serve web/ directory at root after API routes so `/generate` is not shadowed.
app.mount("/", StaticFiles(directory="web", html=True), name="web")
