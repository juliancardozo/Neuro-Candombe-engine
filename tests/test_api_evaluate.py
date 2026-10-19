import importlib
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from candombe_rules import random_structure, render_piece
from evaluate import compare
from model import save_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _client(model_path: str) -> TestClient:
    os.environ["MODEL_PATH"] = model_path
    os.chdir(ROOT)  # StaticFiles serves ./web
    import app

    importlib.reload(app)
    return TestClient(app.app)


@pytest.mark.parametrize("engine", ["rules", "model"])
def test_generate_endpoint(engine, tmp_path, tiny_model):
    path = "__missing__.pt"
    if engine == "model":
        path = str(tmp_path / "tiny.pt")
        save_checkpoint(path, tiny_model)
    client = _client(path)

    res = client.get("/generate", params={"tempo": 100, "bars": 8, "density": 0.6, "random_seed": 1})
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == 2 and data["engine"] == engine
    assert len(data["structure"]) == 8
    drums = [e for e in data["events"] if e["instrument"] != "clave"]
    assert drums and all({"offset_ms", "articulation", "step", "section"} <= set(e) for e in drums)
    assert {e["instrument"] for e in drums} == {"chico", "repique", "piano"}
    assert any(e["offset_ms"] != 0 for e in drums)


def test_structure_and_validation(tmp_path):
    client = _client("__missing__.pt")
    res = client.get("/generate", params={"bars": 4, "structure": "base,cierre", "palmas": False})
    assert res.json()["structure"] == ["base", "cierre", "base", "cierre"]
    assert all(e["instrument"] != "clave" for e in res.json()["events"])
    assert client.get("/generate", params={"structure": "salsa"}).status_code == 422
    assert client.get("/health").json()["model_loaded"] is False


def test_evaluation_metrics_are_sane():
    rng = np.random.default_rng(0)
    pieces = [render_piece(rng, random_structure(rng, 8)) for _ in range(12)]
    report = compare(pieces, pieces, pieces)
    assert report["copy_rate"] == 1.0
    assert report["style_distance"] < 1e-6
    assert report["chico_conformity"] == 1.0
    assert report["piano_bar_anchor"] > 0.9
    assert report["call_adherence"] == 1.0
    assert 0 < report["diversity"] < 1


def _drum_cells(events, instruments=None, bars=None):
    return sorted(
        (e["instrument"], e["step"], e["velocity"], e["articulation"], round(e["offset_ms"], 1))
        for e in events
        if e["instrument"] != "clave"
        and (instruments is None or e["instrument"] in instruments)
        and (bars is None or e["step"] // 16 in bars)
    )


@pytest.mark.parametrize("engine", ["rules", "model"])
def test_regenerate_keeps_what_was_not_selected(engine, tmp_path, tiny_model):
    path = "__missing__.pt"
    if engine == "model":
        path = str(tmp_path / "tiny.pt")
        save_checkpoint(path, tiny_model)
    client = _client(path)

    params = {"tempo": 100, "bars": 4, "density": 0.6, "random_seed": 3}
    first = client.get("/generate", params=params).json()
    assert isinstance(first["random_seed"], int)

    body = {**params, "random_seed": 99, "events": first["events"], "keep_instruments": ["chico", "piano"]}
    second = client.post("/regenerate", json=body).json()
    assert _drum_cells(second["events"], {"chico", "piano"}) == _drum_cells(first["events"], {"chico", "piano"})

    body = {**params, "random_seed": 99, "events": first["events"], "regenerate_bars": [2]}
    third = client.post("/regenerate", json=body).json()
    assert _drum_cells(third["events"], bars={0, 1, 3}) == _drum_cells(first["events"], bars={0, 1, 3})

    everything = {**params, "events": first["events"], "keep_instruments": ["chico", "repique", "piano"]}
    assert client.post("/regenerate", json=everything).status_code == 422


def test_same_seed_repeats_generation():
    client = _client("__missing__.pt")
    a = client.get("/generate", params={"bars": 4, "random_seed": 5}).json()
    b = client.get("/generate", params={"bars": 4, "random_seed": 5}).json()
    assert a["events"] == b["events"]


@pytest.mark.parametrize("engine", ["rules", "model"])
def test_desfile_entrance_and_repique_voices(engine, tmp_path, tiny_model):
    path = "__missing__.pt"
    if engine == "model":
        path = str(tmp_path / "tiny.pt")
        save_checkpoint(path, tiny_model)
    client = _client(path)
    data = client.get("/generate", params={"bars": 32, "repiques": 3, "random_seed": 4}).json()
    assert data["bars"] == 32 and data["repiques"] == 3
    assert data["structure"][0] == "intro" and data["structure"][1] == "llamada_repique" and data["structure"][-1] == "cierre"
    assert all(e["instrument"] != "clave" for e in data["events"])  # no palmas by default

    intro = [e for e in data["events"] if e["step"] < 16]
    assert intro and all(e["instrument"] == "repique" for e in intro)
    voices = {v: _drum_cells([e for e in data["events"] if e.get("voice") == v], {"repique"}, bars=set(range(2, 32))) for v in range(3)}
    assert voices[0] != voices[1] and voices[1] != voices[2]
    intro_lines = {v: _drum_cells([e for e in intro if e["voice"] == v]) for v in range(3)}
    assert intro_lines[0] == intro_lines[1] == intro_lines[2]
