import json

import numpy as np
import pytest

from motion_sport import pipeline
from motion_sport.backends import decision
from motion_sport.conditions import build_view
from motion_sport.controls import PRESETS, apply_controls
from motion_sport.loaders import make_toy_clips
from motion_sport.serialize import view_to_text

D = decision.Decision("snapshot 1: (1,2)", "Which team sport is this?",
                      {"soccer": "association football (soccer)", "rugby_union": "rugby union"})


@pytest.fixture()
def captured(monkeypatch):
    seen = {}

    def fake_post(url, headers, payload):
        seen.update(url=url, headers=headers, payload=payload)
        return {"answers": {"sport": {"type": "choice", "choice": "rugby_union",
                                      "probabilities": {"soccer": 0.2, "rugby_union": 0.8},
                                      "confidence": 0.6}},
                "usage": {"input_tokens": 321, "output_tokens": 0}}

    monkeypatch.setattr(decision, "_post", fake_post)
    return seen


def test_jev_wire_format(captured, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    out = decision.decide("jev:jev-latest", D)
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"] == {"Authorization": "Bearer k"}
    q = captured["payload"]["questions"]["sport"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["soccer", "rugby_union"]
    assert captured["payload"]["model"] == "jev-latest"
    assert out["label"] == "rugby_union" and out["probs"]["rugby_union"] == pytest.approx(0.8)
    assert out["input_tokens"] == 321


def test_laya_http_same_protocol_without_key(captured, monkeypatch):
    monkeypatch.setenv("LAYA_BASE_URL", "http://localhost:8000/v1/")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    decision.decide("laya-http:multilingual", D)
    assert captured["url"] == "http://localhost:8000/v1/systemone"
    assert captured["headers"] == {}


def test_choice_outside_criteria_falls_back_to_argmax(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(decision, "_post", lambda *a: {"answers": {"sport": {
        "choice": "cricket", "probabilities": {"soccer": 3, "rugby_union": 1}}}})
    out = decision.decide("jev:x", D)
    assert out["label"] == "soccer" and out["probs"]["soccer"] == 0.75


def test_pipeline_routes_text_items_to_decision_models(tmp_path):
    for c in make_toy_clips(6, seconds=6):
        c.save(tmp_path / "clips")
    items = pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict",
                             conditions=["motion", "motion_shuffled"], reprs=["sheet", "text"],
                             image_size=64)
    seen = []

    def fake_decide(model_id, d):
        seen.append(d)
        return {"label": next(iter(d.criteria)), "probs": {}, "raw": "", "error": None}

    p = pipeline.run_model(str(items), "laya:multilingual", condition="motion_shuffled",
                           rep="text", state_format="json", decide=fake_decide)
    assert p.name.startswith("laya__multilingual+json")
    assert len(seen) == 12 and all("t" not in s for d in seen for s in d.state["snapshots"])
    assert all("t=" not in d.instructions for d in seen)
    with pytest.raises(ValueError):
        pipeline.run_model(str(items), "jev:jev-latest", condition="motion", rep="sheet")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert all(r["model"] == "laya:multilingual+json" for r in rows)


def test_text_state_is_compact():
    # 12 players x 8 snapshots must stay well inside Laya's multilingual context
    clip = make_toy_clips(1, seconds=6)[0]
    c = apply_controls(clip, PRESETS["strict"], np.random.default_rng(0))
    text = view_to_text(build_view(c, "motion", np.random.default_rng(0)))
    assert len(text) < 1500  # roughly 600-700 tokens
