import json

import pytest

from motion_sport import pipeline
from motion_sport.backends import chat
from motion_sport.loaders import make_toy_clips


@pytest.fixture()
def prepared(tmp_path):
    for c in make_toy_clips(12, seconds=6):
        c.save(tmp_path / "clips")
    out = pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict",
                           conditions=["formation", "motion", "motion_shuffled", "kinematics"],
                           reprs=["sheet", "trails", "text"], image_size=96)
    return out


def test_prepare_writes_consistent_items(prepared):
    cfg = json.loads((prepared / "config.json").read_text())
    items = pipeline.load_items(prepared)
    assert cfg["n_clips_kept"] == 24 and cfg["candidates"] == ["rugby_union", "soccer"]
    for it in items:
        assert it["repr"] in pipeline.VALID[it["condition"]]
        for f in it["files"]:
            assert (prepared / f).exists()
        if it["repr"] == "text":
            assert "snapshot" in it["prompt"] and not it["files"]


def test_run_is_resumable_and_report_has_contrasts(prepared):
    calls = []

    def fake(model_id, req):
        calls.append(req)
        return chat.complete("dummy:first", req)

    for cond in ("motion", "motion_shuffled"):
        pipeline.run_model(str(prepared), "fake:m", condition=cond, rep="sheet", complete=fake)
    n = len(calls)
    assert n == 48 and all(len(r.images) == 1 for r in calls)
    pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet", complete=fake)
    assert len(calls) == n  # nothing re-queried
    rep = pipeline.report(str(prepared), n_boot=50)
    assert any(c["b"].startswith("motion_shuffled") for c in rep["contrasts"])


def test_backend_errors_are_recorded_not_raised(prepared):
    def boom(model_id, req):
        raise chat.BackendError("HTTP 500")

    p = pipeline.run_model(str(prepared), "x:y", condition="formation", rep="sheet",
                           complete=boom, limit=3)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == 3 and all(r["error"] and r["label"] is None for r in rows)


def test_nuisance_baseline_is_at_chance_after_strict(prepared):
    p = pipeline.run_baseline(str(prepared), "nuisance")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    acc = sum(r["label"] == r["sport"] for r in rows) / len(rows)
    assert acc < 0.8  # toy data: player count and field size would give 1.0 without controls
