import os
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from generate import load_model, sample, events_from_array

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve the web/ directory at root
app.mount("/", StaticFiles(directory="web", html=True), name="web")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/candombe_epoch5.pt")
# model load is optional; if missing, generation endpoint returns a simple rule-based pattern
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception:
        model = None

@app.get("/generate")
def generate(tempo: int = 90, bars: int = 4, seed: str = "random"):
    step_duration = 60.0 / tempo / 4  # 16th step in seconds
    steps = bars * 16
    seed_arr = np.zeros((16, 12), dtype=np.float32)
    if model is not None:
        arr = sample(model, seed_arr, length_steps=steps, temperature=0.9)
    else:
        # fallback: simple rule-based pattern (chico steady, piano on 1 & 3, random repique)
        arr = np.zeros((steps, 12), dtype=np.float32)
        for i in range(steps):
            if i % 4 == 0:  # quarter beats: chico
                arr[i, 0] = 1.0
            if i % 8 ==*
