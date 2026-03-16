# Neuro-Candombe-engine — MVP scaffold

This repository contains an MVP scaffold for symbolic generation of Candombe drum patterns and a tiny web demo to play them.

Quick start (local)
1. Create a Python venv:
   python -m venv .venv
   source .venv/bin/activate
2. Install:
   pip install -r requirements.txt
3. Run the server:
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
4. Open http://localhost:8000 in your browser.

What’s included
- src/: preprocessing, model, training, generation, FastAPI app
- web/: frontend and sample placeholders
- models/: (place checkpoints here)
- data/: input and processed data

Notes
- This scaffold uses a small Transformer-style model and sample-triggered playback for fast iteration.
- Replace the placeholder WAV samples in web/samples/ with short drum hits for chico, repique, piano.