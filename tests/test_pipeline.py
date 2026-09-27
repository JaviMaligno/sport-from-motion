import json

import numpy as np
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
    s = next(iter(rep["runs"].values()))
    for k in ("macro_f1", "kappa", "prior_corrected_accuracy", "per_class_recall"):
        assert k in s and f"{k}_ci" in s
    assert set(s["per_class_recall"]) == {"rugby_union", "soccer"}


def test_report_cli_renders_new_columns(prepared, capsys):
    from motion_sport import cli

    pipeline.run_model(str(prepared), "dummy:first", condition="motion", rep="sheet")
    cli.main(["report", "--items", str(prepared), "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "pc-acc" in out and "kappa" in out and "per-class recall" in out


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


def test_interleave_balances_any_prefix_and_is_condition_independent():
    from motion_sport.pipeline import interleave

    def items(cond):
        return [{"clip_id": f"{s}-{m}-{k}", "sport": s, "match_id": f"{s}{m}", "condition": cond}
                for s in ("a", "b") for m in range(3) for k in range(4)]

    out = interleave(items("motion"))
    assert {i["sport"] for i in out[:2]} == {"a", "b"}
    assert sum(i["sport"] == "a" for i in out[:10]) == 5
    assert len({i["match_id"] for i in out[:6]}) == 6  # spreads across matches too
    assert [i["clip_id"] for i in out] == [i["clip_id"] for i in interleave(items("shuffled")[::-1])]


def _coin(seed=0):
    """Fake chat model answering at random (different in every replicate)."""
    import random

    rnd = random.Random(seed)

    def complete(model_id, req):
        opts = ["rugby union", "association football (soccer)"]
        a = rnd.choice(opts)
        return json.dumps({"probabilities": {o: float(o == a) for o in opts}, "answer": a})
    return complete


def test_replicates_have_own_files_and_are_grouped(prepared):
    calls = []

    def counting(model_id, req, _c=_coin(1)):
        calls.append(1)
        return _c(model_id, req)

    p1 = pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet", complete=counting)
    p2 = pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet",
                            complete=counting, replicate=2, limit=10)
    assert p1.name == "fake__m__motion__sheet.jsonl" and p2.name == "fake__m__motion__sheet__r2.jsonl"
    assert len(calls) == 24 + 10  # replicate 2 did not reuse replicate 1's answers
    assert all(json.loads(line)["replicate"] == 2 for line in p2.read_text().splitlines())
    pipeline.run_model(str(prepared), "fake:m", condition="motion_shuffled", rep="sheet",
                       complete=_coin(2))
    rep = pipeline.report(str(prepared), n_boot=50)
    (g,) = rep["replicates"]
    assert g["k"] == 2 and g["n_common"] == 10 and 0 <= g["agreement"] <= 1
    assert g["mean_accuracy"] == pytest.approx(sum(g["accuracy_per_replicate"]) / 2)
    c = next(c for c in rep["contrasts"] if c["b"] == "motion_shuffled/sheet")
    assert c["n"] == 10  # only items present in every replicate


def test_report_ignores_stale_clips_and_missing_cells(prepared):
    pdir = prepared / "predictions"
    pdir.mkdir(exist_ok=True)
    stale = {"item_id": "gone__motion__sheet", "clip_id": "gone", "sport": "soccer",
             "match_id": "x", "tags": [], "model": "old:m", "condition": "motion",
             "repr": "sheet", "label": "soccer", "probs": {"soccer": 1.0}, "error": None}
    (pdir / "old__m__motion__sheet.jsonl").write_text(json.dumps(stale) + "\n")
    # a model with a single, non-motion cell: nothing to contrast, must not crash
    pipeline.run_model(str(prepared), "fake:solo", condition="formation", rep="text",
                       complete=_coin(3), limit=4)
    rep = pipeline.report(str(prepared), n_boot=20)
    assert rep["ignored"]["rows_not_in_items"] == 1
    assert not any(r["model"] == "old:m" for r in rep["runs"].values())
    assert not any(c["model"] == "fake:solo" for c in rep["contrasts"])


def test_informed_prompt_style_runs_to_its_own_file(prepared):
    seen = []

    def fake(model_id, req):
        seen.append(req.prompt)
        return chat.complete("dummy:first", req)

    items = pipeline.load_items(prepared)
    assert all(it["prompt_informed"] != it["prompt"] for it in items)
    pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet", complete=fake)
    p = pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet",
                           complete=fake, prompt_style="informed")
    assert p.name == "fake__m__motion__sheet__informed.jsonl"
    assert all("How players typically move" in q for q in seen[24:])
    assert all("How players typically move" not in q for q in seen[:24])
    assert all(json.loads(line)["prompt_style"] == "informed" for line in p.read_text().splitlines())
    rep = pipeline.report(str(prepared), n_boot=20)
    assert any(c["a"] == "motion/sheet[informed]" and c["b"] == "motion/sheet"
               for c in rep["contrasts"])
    styles = {r["prompt_style"] for r in rep["runs"].values()}
    assert styles == {"neutral", "informed"}


def test_old_items_fail_clearly_only_for_informed(prepared):
    path = prepared / "items.jsonl"
    old = [{k: v for k, v in json.loads(line).items()
            if k not in ("prompt_informed", "decision_criteria_informed")}
           for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(i) + "\n" for i in old))
    pipeline.run_model(str(prepared), "dummy:first", condition="formation", rep="sheet", limit=2)
    with pytest.raises(ValueError, match="re-run `motion-sport prepare`"):
        pipeline.run_model(str(prepared), "dummy:first", condition="formation", rep="sheet",
                           prompt_style="informed")


def test_informed_criteria_reach_decision_models(prepared):
    got = []

    def decide(model_id, d):
        got.append(d.criteria)
        c = next(iter(d.criteria))
        return {"label": c, "probs": {c: 1.0}, "error": None, "raw": ""}

    pipeline.run_model(str(prepared), "jev:x", condition="motion", rep="text", decide=decide,
                       prompt_style="informed", limit=2)
    assert len(got) == 2 and all(":" in v for crit in got for v in crit.values())


def test_prepare_player_mode_random(tmp_path):
    from motion_sport import cli
    from motion_sport.schema import load_clips

    for c in make_toy_clips(4, seconds=6):
        c.save(tmp_path / "clips")
    outs = {}
    for mode in ("central", "random"):
        out = tmp_path / mode
        cli.main(["prepare", "--clips", str(tmp_path / "clips"), "--out", str(out),
                  "--conditions", "motion", "--reprs", "text", "--n-players", "6",
                  "--player-mode", mode])
        assert json.loads((out / "config.json").read_text())["controls"]["player_mode"] == mode
        outs[mode] = load_clips(out / "clips")
        for c in outs[mode]:
            log = next(x for x in c.meta["controls"] if x["name"] == "fix_player_count")
            assert log == {"name": "fix_player_count", "n": 6, "mode": mode}
    # different subsets: at least one clip keeps other players
    assert any(not np.allclose(np.sort(a.xy.ravel()), np.sort(b.xy.ravel()))
               for a, b in zip(outs["central"], outs["random"]))
    with pytest.raises(ValueError):
        pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "x"), conditions=["motion"],
                         reprs=["text"], player_mode="nearest")
