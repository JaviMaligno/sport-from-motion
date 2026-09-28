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


def test_primary_contrasts_are_holm_corrected_as_one_family(prepared):
    from motion_sport.evaluate import holm

    for model in ("fake:a", "fake:b"):
        for cond, rep in (("motion", "sheet"), ("motion_shuffled", "sheet"),
                          ("formation", "sheet"), ("motion", "text"), ("kinematics", "sheet")):
            pipeline.run_model(str(prepared), model, condition=cond, rep=rep,
                               complete=_coin(hash((model, cond, rep)) % 1000))
    pipeline.run_model(str(prepared), "fake:a", condition="motion", rep="sheet",
                       complete=_coin(7), prompt_style="informed")
    rep = pipeline.report(str(prepared), n_boot=100)
    prim = rep["primary_contrasts"]
    # a: order, shape, text, prompt; b: no informed cell -> no prompt contrast
    assert sorted(c["contrast"] for c in prim if c["model"] == "fake:a") == \
        ["motion_over_shape", "order", "prompt", "text_vs_image"]
    assert len([c for c in prim if c["model"] == "fake:b"]) == 3
    assert rep["holm"]["family_size"] == 7
    assert [c["p_holm"] for c in prim] == pytest.approx(holm([c["p"] for c in prim]))
    assert all(c["p_holm"] >= c["p"] for c in prim)
    sec = rep["secondary_contrasts"]
    assert sec and not any(c["primary"] or "p_holm" in c for c in sec)
    assert any(c["b"] == "kinematics/sheet" for c in sec)


@pytest.mark.parametrize("n_frames", [10, 40])
def test_prepare_n_frames_sets_the_window_length(tmp_path, n_frames):
    for c in make_toy_clips(4, seconds=8):  # 40 frames at 5 Hz
        c.save(tmp_path / "clips")
    out = pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict_smooth",
                           conditions=["motion"], reprs=["text"], n_frames=n_frames)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["controls"]["n_frames"] == n_frames and cfg["n_clips_kept"] == 8
    from motion_sport.schema import load_clips
    assert {c.n_frames for c in load_clips(out / "clips")} == {n_frames}


def test_prepare_n_frames_longer_than_the_clips_keeps_nothing(tmp_path):
    for c in make_toy_clips(2, seconds=4):  # 20 frames
        c.save(tmp_path / "clips")
    out = pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict",
                           conditions=["motion"], reprs=["text"], n_frames=40)
    assert json.loads((out / "config.json").read_text())["n_clips_kept"] == 0
    with pytest.raises(ValueError):
        pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "x"), preset="strict",
                         conditions=["motion"], reprs=["text"], n_frames=1)


def test_cli_prepare_passes_n_frames(tmp_path):
    from motion_sport import cli

    for c in make_toy_clips(2, seconds=4):
        c.save(tmp_path / "clips")
    cli.main(["prepare", "--clips", str(tmp_path / "clips"), "--out", str(tmp_path / "items"),
              "--conditions", "motion", "--reprs", "text", "--n-frames", "10"])
    assert json.loads((tmp_path / "items" / "config.json").read_text())["controls"]["n_frames"] == 10


def _walkers(tmp_path, source, n, teleport, seed):
    """n clips of 12 players walking at 4 m/s; `teleport` of them with one 100 m/s step."""
    from motion_sport.schema import Clip

    rng = np.random.default_rng(seed)
    for i in range(n):
        start = rng.uniform(0, 40, (12, 2))
        v = rng.normal(0, 1, (12, 2))
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * 4.0
        xy = start[None] + np.arange(24)[:, None, None] / 5.0 * v[None]
        if i < teleport:
            xy[12:, :] += np.array([20.0, 0.0])  # everyone jumps 20 m in one 0.2 s step
        Clip(clip_id=f"{source}-{i:03d}", sport="handball", source=source,
             match_id=f"{source}-m{i % 3}", fps=5.0, xy=xy).save(tmp_path / "clips")


def test_prepare_reports_rejections_per_sport_and_source(tmp_path, capsys):
    from motion_sport import cli

    _walkers(tmp_path, "clean", 10, 0, seed=1)
    _walkers(tmp_path, "jumpy", 10, 3, seed=2)
    cli.main(["prepare", "--clips", str(tmp_path / "clips"), "--out", str(tmp_path / "items"),
              "--preset", "strict_smooth", "--conditions", "motion", "--reprs", "text",
              "--n-players", "10"])
    out = capsys.readouterr().out
    cfg = json.loads((tmp_path / "items" / "config.json").read_text())
    assert cfg["controls"]["max_speed_ms"] == 12.0
    assert cfg["clips_in_by_source"] == {"clean": 10, "jumpy": 10}
    assert cfg["clips_kept_by_source"] == {"clean": 10, "jumpy": 7}
    assert cfg["rejected_by_source"] == {"jumpy": {"teleport": 3}}
    assert cfg["rejected_by_sport"] == {"handball": {"teleport": 3}}
    assert "WARNING: source jumpy: lost 3/10" in out and "source clean" not in out
    # the ceiling is configurable; 0 switches it off
    cli.main(["prepare", "--clips", str(tmp_path / "clips"), "--out", str(tmp_path / "off"),
              "--conditions", "motion", "--reprs", "text", "--n-players", "10",
              "--max-speed-ms", "0"])
    off = json.loads((tmp_path / "off" / "config.json").read_text())
    assert off["controls"]["max_speed_ms"] is None and off["n_clips_kept"] == 20
    hi = pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "hi"), conditions=["motion"],
                          reprs=["text"], n_players=10, max_speed_ms=200.0)
    assert json.loads((hi / "config.json").read_text())["n_clips_kept"] == 20


def _boom(model_id, req):
    raise chat.BackendError("HTTP 404: model not enabled")


PRIMARY_CELLS = (("motion", "sheet"), ("motion_shuffled", "sheet"), ("formation", "sheet"),
                 ("motion", "text"))


def test_all_error_model_is_out_of_the_contrasts_and_the_holm_family(prepared, capsys):
    from motion_sport import cli

    for cond, rep in PRIMARY_CELLS:
        pipeline.run_model(str(prepared), "fake:ok", condition=cond, rep=rep, complete=_coin(5))
        pipeline.run_model(str(prepared), "fake:dead", condition=cond, rep=rep, complete=_boom)
    # a model with one working cell is not excluded as a model, but its all-error cell is
    # a systemic failure (> 50 %, section 8) and leaves the contrasts
    pipeline.run_model(str(prepared), "fake:half", condition="motion", rep="sheet", complete=_coin(6))
    pipeline.run_model(str(prepared), "fake:half", condition="motion_shuffled", rep="sheet",
                       complete=_boom)
    rep = pipeline.report(str(prepared), n_boot=50)
    assert [e["model"] for e in rep["excluded_models"]] == ["fake:dead"]
    assert not any(c["model"] == "fake:dead" for c in rep["contrasts"])
    assert rep["holm"]["family_size"] == 3  # fake:ok (order, shape, text); fake:half none
    assert ("fake:half", "motion_shuffled/sheet") in {(c["model"], c["cell"]) for c in rep["excluded_cells"]}
    assert any(r["model"] == "fake:dead" for r in rep["runs"].values())  # still in the table
    cli.main(["report", "--items", str(prepared), "--n-boot", "20"])
    assert "NOTE: fake:dead excluded from every contrast and from the Holm family" in \
        capsys.readouterr().out


def test_plan_json_flags_missing_rows_and_error_heavy_cells(prepared, capsys):
    from motion_sport import cli

    pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet", complete=_coin(1),
                       limit=10)
    pipeline.run_model(str(prepared), "fake:m", condition="formation", rep="sheet", complete=_boom,
                       limit=24)
    pipeline.run_model(str(prepared), "fake:m", condition="motion_shuffled", rep="sheet",
                       complete=_coin(2), limit=24)
    (prepared / "plan.json").write_text(json.dumps({"cells": {
        "fake__m__motion__sheet": 24, "fake__m__formation__sheet": 24,
        "fake__m__motion_shuffled__sheet": 24, "fake__m__motion__text": 24}}))
    st = {s["file"]: s for s in pipeline.plan_status(prepared)}
    assert st["fake__m__motion__sheet"] == {"file": "fake__m__motion__sheet", "planned": 24,
                                            "rows": 10, "errors": 0, "missing": 14, "flagged": True}
    assert st["fake__m__motion__text"]["rows"] == 0 and st["fake__m__motion__text"]["missing"] == 24
    assert st["fake__m__formation__sheet"]["missing"] == 0 and st["fake__m__formation__sheet"]["flagged"]
    assert not st["fake__m__motion_shuffled__sheet"]["flagged"]
    rep = pipeline.report(str(prepared), n_boot=20, tag="nonexistent-tag")  # tag-independent
    assert {c["file"] for c in rep["incomplete"]} == {
        "fake__m__motion__sheet", "fake__m__formation__sheet", "fake__m__motion__text"}
    cli.main(["report", "--items", str(prepared), "--n-boot", "20"])
    assert "fake__m__motion__sheet: 10/24 rows (14 missing)" in capsys.readouterr().out


def test_no_plan_json_no_incomplete(prepared):
    pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet", complete=_coin(1),
                       limit=5)
    assert pipeline.plan_status(prepared) == [] and pipeline.report(str(prepared), n_boot=20)["incomplete"] == []


def _errors_on(error_ids, seed=0):
    """A coin-flipping fake backend that fails on the given item numbers (0-based order)."""
    coin, seen = _coin(seed), []

    def complete(model_id, req):
        seen.append(1)
        if len(seen) - 1 in error_ids:
            raise chat.BackendError("HTTP 500")
        return coin(model_id, req)
    return complete


def test_systemic_failure_cell_leaves_every_contrast_and_the_holm_family(prepared, capsys):
    from motion_sport import cli

    # fake:m: formation/sheet 13 of 24 errors (> 50 %) -> excluded; motion_shuffled/sheet
    # 3 of 24 (12.5 %) -> kept and flagged; motion/sheet, motion/text and kinematics clean
    for cond, rep in PRIMARY_CELLS + (("kinematics", "sheet"),):
        errs = {"formation": set(range(13)), "motion_shuffled": {0, 1, 2}}.get(cond, set())
        pipeline.run_model(str(prepared), "fake:m", condition=cond, rep=rep, workers=1,
                           complete=_errors_on(errs, seed=hash((cond, rep)) % 100))
    # exactly 50 % is not above the threshold: stays, flagged
    pipeline.run_model(str(prepared), "fake:m", condition="kinematics", rep="text", workers=1,
                       complete=_errors_on(set(range(12))))
    rep = pipeline.report(str(prepared), n_boot=50)
    assert [(c["cell"], c["errors"], c["n_rows"]) for c in rep["excluded_cells"]] == \
        [("formation/sheet", 13, 24)]
    assert {c["cell"] for c in rep["flagged_cells"]} == {"motion_shuffled/sheet", "kinematics/text"}
    assert not any("formation/sheet" in (c["a"], c["b"]) for c in rep["contrasts"])
    assert sorted(c["contrast"] for c in rep["primary_contrasts"]) == ["order", "text_vs_image"]
    assert rep["holm"]["family_size"] == 2
    order = next(c for c in rep["primary_contrasts"] if c["contrast"] == "order")
    assert order["flagged_cells"] == ["motion_shuffled/sheet"]
    assert any(c["b"] == "kinematics/sheet" for c in rep["secondary_contrasts"])  # still there
    # the excluded cell's row stays in the table
    formation = rep["runs"]["fake__m__formation__sheet"]
    assert formation["cell_status"] == "excluded" and formation["n"] == 24
    cli.main(["report", "--items", str(prepared), "--n-boot", "20"])
    out = capsys.readouterr().out
    table_row = next(line for line in out.splitlines() if line.startswith("fake:m") and "formation" in line)
    assert "cell excluded: systemic failure (13/24 errors)" in table_row
    assert "formation/sheet          cell excluded: systemic failure (13/24 unrecovered errors)" in out
    assert "! errors > 2 % in motion_shuffled/sheet" in out


def test_launcher_summary_names_the_systemic_failure(prepared):
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "run_plan", pathlib.Path(__file__).parents[1] / "scripts" / "run_plan.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    pipeline.run_model(str(prepared), "fake:m", condition="formation", rep="sheet", workers=1,
                       complete=_errors_on(set(range(13))))
    pipeline.run_model(str(prepared), "fake:m", condition="motion", rep="sheet", workers=1,
                       complete=_errors_on({0}))
    out = rp.check(prepared)
    assert any(line.startswith("fake__m__formation__sheet.jsonl: 13/24") and "SYSTEMIC FAILURE" in line
               for line in out)
    assert any(line.startswith("fake__m__motion__sheet.jsonl: 1/24") and "REPORT AS SUCH" in line
               for line in out)
