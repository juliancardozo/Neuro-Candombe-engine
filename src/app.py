import os
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from generate import load_model, sample, events_from_array, INPUT_DIM

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve web/ directory at root.
app.mount("/", StaticFiles(directory="web", html=True), name="web")

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


@app.get("/generate")
def generate(
    tempo: int = 90,
    bars: int = 4,
    seed: str = "clave",
    temperature: float = 0.9,
    density: float = 0.55,
):
    tempo = max(50, min(220, tempo))
    bars = max(1, min(16, bars))
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

    events = events_from_array(arr, step_duration=step_duration)
    return {
        "tempo": tempo,
        "bars": bars,
        "step_duration": step_duration,
        "model_loaded": model is not None,
        "events": events,
    }
