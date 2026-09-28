"""Tests for the helper scripts in scripts/ (loaded by path: they are not a package)."""
import importlib.util
import json
import os
import pathlib

import numpy as np
import pytest

from motion_sport.loaders import load_long_csv
from motion_sport.schema import Clip, load_clips

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_teamtrack_handball_halves_are_one_match():
    tt = _script("teamtrack_to_long_csv")
    assert tt.match_id("Handball", "1st_fisheye_0-30") == "tt-handball"
    assert tt.match_id("Handball", "2nd_fisheye_660-690") == "tt-handball"
    assert tt.match_id("Soccer", "F_20200220_1_0000_0030") == "20200220"
    assert tt.match_id("Basketball", "P3_0-30") == "P3"


def test_long_csv_keeps_hyphenated_match_ids_with_group_extra(tmp_path):
    rows = ["match_id,segment,frame,track_id,x,y"]
    for seg in ("train-1st_fisheye_0-30", "val-2nd_fisheye_60-90"):
        rows += [f"tt-handball,{seg},{f},{t},{t + f * 0.1},{t}" for f in range(20) for t in range(10)]
    p = tmp_path / "handball.csv"
    p.write_text("\n".join(rows))
    clips = list(load_long_csv(p, sport="handball", source="teamtrack", fps=10, clip_seconds=2,
                               stride_seconds=2, group_extra=["segment"]))
    assert len(clips) == 2 and {c.match_id for c in clips} == {"tt-handball"}
    assert clips[0].clip_id == "teamtrack-tt-handball-train-1st_fisheye_0-30-00000"


def _tt_clip(cid, match, sport="handball"):
    return Clip(clip_id=cid, sport=sport, source="teamtrack", match_id=match, fps=5.0,
                xy=np.arange(40, dtype=np.float32).reshape(4, 5, 2), tags=["x"],
                meta={"start_frame": 3})


def test_relabel_match_replaces_symlinks_and_never_touches_the_source(tmp_path):
    rl = _script("relabel_match")
    src, pool = tmp_path / "clips", tmp_path / "pool"
    for c in (_tt_clip("teamtrack-1st-a", "1st"), _tt_clip("teamtrack-2nd-b", "2nd"),
              _tt_clip("teamtrack-1st-soc", "1st", sport="soccer"),
              _tt_clip("eigd-x", "g1")):
        c.save(src)
    pool.mkdir()
    for p in sorted(src.glob("*.npz")):
        os.symlink(p, pool / p.name)
    before = {p.name: p.read_bytes() for p in src.glob("*.npz")}
    c = rl.relabel(pool, ["teamtrack-1st-*", "teamtrack-2nd-*"], "tt-handball", sport="handball")
    assert c["relabelled"] == 2 and c["wrong_sport"] == 1 and c["symlinks_replaced"] == 2
    assert c["old_match_ids"] == {"1st": 1, "2nd": 1}
    assert {p.name: p.read_bytes() for p in src.glob("*.npz")} == before  # source intact
    got = {c.clip_id: c for c in load_clips(pool)}
    assert got["teamtrack-1st-a"].match_id == got["teamtrack-2nd-b"].match_id == "tt-handball"
    assert got["teamtrack-1st-soc"].match_id == "1st" and got["eigd-x"].match_id == "g1"
    a = got["teamtrack-1st-a"]
    assert a.tags == ["x"] and a.meta == {"start_frame": 3} and np.allclose(a.xy, np.arange(40).reshape(4, 5, 2))
    assert not (pool / "teamtrack-1st-a.npz").is_symlink() and (pool / "eigd-x.npz").is_symlink()
    assert not list(pool.glob(".*.tmp"))
    again = rl.relabel(pool, ["teamtrack-1st-*", "teamtrack-2nd-*"], "tt-handball", sport="handball")
    assert again["relabelled"] == 0 and again["unchanged"] == 2


def test_relabel_match_dry_run_writes_nothing(tmp_path):
    rl = _script("relabel_match")
    _tt_clip("teamtrack-1st-a", "1st").save(tmp_path)
    before = (tmp_path / "teamtrack-1st-a.npz").read_bytes()
    assert rl.relabel(tmp_path, ["teamtrack-1st-*"], "tt-handball", dry_run=True)["relabelled"] == 1
    assert (tmp_path / "teamtrack-1st-a.npz").read_bytes() == before
    assert json.loads(str(np.load(tmp_path / "teamtrack-1st-a.npz")["meta_json"]))["match_id"] == "1st"


# ------------------------------------------------------------- run plan / primary view

def _prepared(tmp_path):
    from motion_sport import pipeline
    from motion_sport.loaders import make_toy_clips

    for c in make_toy_clips(12, seconds=6):
        c.save(tmp_path / "clips")
    return pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict",
                            conditions=["motion", "motion_shuffled"], reprs=["sheet", "text"],
                            image_size=64)


def _fake(model_id, req):
    a = "rugby union"
    return json.dumps({"probabilities": {a: 1.0}, "answer": a})


def test_plan_from_final_run_commands_and_check(tmp_path):
    from motion_sport import pipeline

    rp = _script("run_plan")
    items = _prepared(tmp_path)
    lines = [
        "# --- a comment line from DRY_RUN",
        f".venv/bin/motion-sport run --items {items} --model azure-openai:gpt-5.6-sol "
        "--condition motion --repr sheet --limit 10 --workers 6",
        f".venv/bin/motion-sport run --items {items} --model azure-openai:gpt-5.6-sol "
        "--condition motion --repr sheet --limit 5 --workers 6 --replicate 2",
        f".venv/bin/motion-sport run --items {items} --model azure-openai:gpt-5.6-sol "
        "--condition motion --repr sheet --limit 10 --workers 6 --prompt-style informed",
        f".venv/bin/motion-sport run --items {items} --model jev-openrouter:~typesafe/jev-latest "
        "--condition motion --repr text --limit 10 --workers 6 --state-format json",
    ]
    assert rp.plan_from_commands(items, lines)["cells"] == {
        "azure-openai__gpt-5.6-sol__motion__sheet": 10,
        "azure-openai__gpt-5.6-sol__motion__sheet__r2": 5,
        "azure-openai__gpt-5.6-sol__motion__sheet__informed": 10,
        "jev-openrouter__~typesafe_jev-latest+json__motion__text": 10,
    }
    rp.write_plan(items, lines[:3])
    rp.write_plan(items, lines[3:])  # a relaunch for one model merges, never drops cells
    assert len(json.loads((items / "plan.json").read_text())["cells"]) == 4
    p = pipeline.run_model(str(items), "azure-openai:gpt-5.6-sol", condition="motion", rep="sheet",
                           limit=10, complete=_fake)
    assert p.stem == "azure-openai__gpt-5.6-sol__motion__sheet"
    out = rp.check(items)
    assert "azure-openai__gpt-5.6-sol__motion__sheet__r2.jsonl: no file, 5 planned rows MISSING" in out
    assert not any(line.startswith("azure-openai__gpt-5.6-sol__motion__sheet.jsonl") for line in out)


def test_primary_view_restricts_specialists_to_the_models_clips(tmp_path):
    from motion_sport import pipeline

    pv, rp = _script("primary_view"), _script("run_plan")
    items = _prepared(tmp_path)
    for cond in ("motion", "motion_shuffled"):
        pipeline.run_model(str(items), "azure-openai:gpt-5.6-sol", condition=cond, rep="sheet",
                           limit=10, complete=_fake)
    pipeline.run_model(str(items), "azure-openai:gpt-5.6-sol", condition="motion", rep="sheet",
                       limit=4, complete=_fake, replicate=2)
    pipeline.run_model(str(items), "other:model", condition="motion", rep="sheet", limit=10,
                       complete=_fake)
    base = pipeline.run_baseline(str(items), "nuisance")
    base_before = base.read_bytes()
    rp.write_plan(items, [
        f"motion-sport run --items {items} --model azure-openai:gpt-5.6-sol --condition motion "
        "--repr sheet --limit 10",
        f"motion-sport run --items {items} --model azure-openai:gpt-5.6-sol --condition motion "
        "--repr sheet --limit 4 --replicate 2",
        f"motion-sport run --items {items} --model vertex-anthropic:claude-opus-5-5 "
        "--condition motion --repr sheet --limit 10",
    ])
    info = pv.build_view(items, tmp_path / "view", clips_from_models=True)
    assert info["model_clips"] == 10 and info["model_clip_sets_equal"]
    assert info["restricted"] == {base.name: (24, 10)}
    view = tmp_path / "view" / "predictions"
    assert not (view / base.name).is_symlink() and base.read_bytes() == base_before
    model_clips = {json.loads(line)["clip_id"] for line in
                   (view / "azure-openai__gpt-5.6-sol__motion__sheet.jsonl").read_text().splitlines()}
    assert {json.loads(line)["clip_id"] for line in (view / base.name).read_text().splitlines()} \
        == model_clips
    assert set(info["skipped"]) == {"azure-openai__gpt-5.6-sol__motion__sheet__r2.jsonl",
                                    "other__model__motion__sheet.jsonl"}
    # the view's plan: replicate-1 cells of planned models (the missing Opus cell is flagged)
    plan = json.loads((tmp_path / "view" / "plan.json").read_text())["cells"]
    assert plan == {"azure-openai__gpt-5.6-sol__motion__sheet": 10,
                    "vertex-anthropic__claude-opus-5-5__motion__sheet": 10}
    rep = pipeline.report(str(tmp_path / "view"), n_boot=20)
    assert [c["file"] for c in rep["incomplete"]] == ["vertex-anthropic__claude-opus-5-5__motion__sheet"]
    # without the flag the specialist keeps every clip
    info2 = pv.build_view(items, tmp_path / "view2")
    assert info2["restricted"] == {} and (tmp_path / "view2" / "predictions" / base.name).is_symlink()


def _final_like(tmp_path, *, controls=None, af_tags=("mid_play", "random_phase"),
                preset="strict_smooth", reprs=("sheet", "text", "trails", "video")):
    root = tmp_path / "final"
    root.mkdir()
    sports = ["american_football", "basketball", "handball", "soccer"]
    ctrl = {"n_players": 10, "player_mode": "random", "space": "spread", "rotate": True,
            "tempo": None, "n_frames": 20, "drop_team": True, "smooth": 2.0, "max_speed_ms": 12.0}
    ctrl.update(controls or {})
    (root / "config.json").write_text(json.dumps({
        "preset": preset, "controls": ctrl, "candidates": sports,
        "clips_kept_by_sport": {s: 100 for s in sports}}))
    items = [{"clip_id": f"{s}-{k}", "sport": s, "repr": r, "prompt_informed": "...",
              "tags": list(af_tags) if s == "american_football" else []}
             for s in sports for k in range(3) for r in reprs]
    (root / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    return root


MODELS = ["azure-openai:gpt-5.6-sol", "vertex:gemini-3.1-pro-preview"]


def test_preflight_accepts_the_preregistered_set(tmp_path):
    assert _script("run_plan").preflight(_final_like(tmp_path), 400, MODELS) == []


@pytest.mark.parametrize("kw, expect", [
    ({"controls": {"player_mode": "central"}}, "controls.player_mode is 'central'"),
    ({"controls": {"n_players": 12}}, "controls.n_players is 12"),
    ({"controls": {"max_speed_ms": None}}, "controls.max_speed_ms is None"),
    ({"preset": "strict"}, "preset is 'strict'"),
    ({"af_tags": ("play", "mid_play")}, "3/3 american_football clips not tagged random_phase"),
    ({"reprs": ("sheet", "text", "trails")}, "missing representations ['video']"),
])
def test_preflight_rejects_what_is_not_preregistered(tmp_path, kw, expect):
    bad = _script("run_plan").preflight(_final_like(tmp_path, **kw), 400, MODELS)
    assert len(bad) == 1 and bad[0].startswith(expect)


def test_preflight_cli_exit_code(tmp_path):
    import subprocess
    import sys

    ok = _final_like(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "run_plan.py"), "preflight", str(ok), "400", " ".join(MODELS)]
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS.parent / "src")}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "preflight ok" in r.stdout
    cfg = json.loads((ok / "config.json").read_text())
    cfg["controls"]["player_mode"] = "central"
    (ok / "config.json").write_text(json.dumps(cfg))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "controls.player_mode" in r.stderr


def test_plan_keeps_only_the_commands_on_its_own_items(tmp_path):
    rp = _script("run_plan")
    main, d8 = tmp_path / "final", tmp_path / "final-d8"
    lines = [
        f"motion-sport run --items {main} --model azure-openai:gpt-5.6-sol --condition motion "
        "--repr sheet --limit 400",
        f"motion-sport run --items {d8} --model azure-openai:gpt-5.6-sol --condition motion "
        "--repr sheet --limit 300",
    ]
    assert rp.plan_from_commands(main, lines)["cells"] == {"azure-openai__gpt-5.6-sol__motion__sheet": 400}
    assert rp.plan_from_commands(d8, lines)["cells"] == {"azure-openai__gpt-5.6-sol__motion__sheet": 300}


def _d8_like(tmp_path, *, n_frames=40, candidates=("basketball", "handball", "soccer"),
             cells=(("motion", "sheet"), ("motion_shuffled", "sheet"))):
    root = tmp_path / "final-d8"
    root.mkdir(parents=True)
    ctrl = {"n_players": 10, "player_mode": "random", "n_frames": n_frames, "smooth": 2.0,
            "max_speed_ms": 12.0}
    (root / "config.json").write_text(json.dumps({
        "preset": "strict_smooth", "controls": ctrl, "candidates": list(candidates),
        "clips_kept_by_sport": {s: 100 for s in candidates}}))
    items = [{"clip_id": f"{s}-{k}", "sport": s, "condition": c, "repr": r}
             for s in candidates for k in range(3) for c, r in cells]
    (root / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    return root


def test_preflight_a7b(tmp_path):
    rp = _script("run_plan")
    assert rp.preflight_a7b(_d8_like(tmp_path / "ok"), 300) == []
    bad = rp.preflight_a7b(_d8_like(tmp_path / "d4", n_frames=20), 300)
    assert bad == ["controls.n_frames is 20, pre-registered 40"]
    bad = rp.preflight_a7b(_d8_like(tmp_path / "af", candidates=(
        "american_football", "basketball", "handball", "soccer")), 300)
    assert len(bad) == 1 and bad[0].startswith("candidates are")
    bad = rp.preflight_a7b(_d8_like(tmp_path / "cells", cells=(("motion", "sheet"),)), 300)
    assert bad == ["missing cells [('motion_shuffled', 'sheet')]"]
