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


def test_teamtrack_one_match_per_sport():
    tt = _script("teamtrack_to_long_csv")
    assert tt.match_id("Handball", "1st_fisheye_0-30") == "tt-handball"
    assert tt.match_id("Handball", "2nd_fisheye_660-690") == "tt-handball"
    assert tt.match_id("Soccer", "F_20200220_1_0000_0030") == "tt-soccer"
    assert tt.match_id("Soccer", "F_20220220_1_0900_0930") == "tt-soccer"  # same recording
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


def test_teamtrack_missing_zero_becomes_nan():
    tt = _script("teamtrack_to_long_csv")
    xy = np.array([[0.0, 0.0, 1.0, 2.0, 0.0, 3.0],     # player 0 missing; x = 0 alone is real
                   [5.0, 0.0, 0.0, 0.0, 4.0, 4.0]])    # y = 0 alone is real; player 1 missing
    out = tt.mask_missing(xy)
    assert np.isnan(out[0, :2]).all() and np.isnan(out[1, 2:4]).all()
    assert out[0, 2:].tolist() == [1.0, 2.0, 0.0, 3.0] and out[1, [0, 1, 4, 5]].tolist() == [5, 0, 4, 4]
    assert xy[0, 0] == 0.0  # input untouched


def test_teamtrack_missing_rows_read_back_as_unobserved(tmp_path):
    """A (0, 0) track written as NaN by the converter is not a candidate player after ingest."""
    tt = _script("teamtrack_to_long_csv")
    rows = ["match_id,segment,frame,track_id,x,y"]
    for f in range(25):
        xy = np.array([[t + 0.3 * f, t * 0.5 + 0.1 * f] for t in range(10)] + [[0.0, 0.0]]).ravel()
        xy = tt.mask_missing(xy[None])[0]
        rows += [f"tt-handball,train-a,{f},{t},{xy[2 * t]},{xy[2 * t + 1]}" for t in range(11)]
    (tmp_path / "h.csv").write_text("\n".join(rows))
    clips = list(load_long_csv(tmp_path / "h.csv", sport="handball", source="teamtrack", fps=5,
                               clip_seconds=4, stride_seconds=4, group_extra=["segment"]))
    assert len(clips) == 1 and clips[0].n_players == 10  # never-seen track dropped at ingest
    assert not ((clips[0].xy[..., 0] == 0) & (clips[0].xy[..., 1] == 0)).any()


def test_teamtrack_cache_file_list(tmp_path):
    tt = _script("teamtrack_to_long_csv")
    for rel in ("Handball/train/1st_fisheye_0-30_0.txt", "Soccer/val/F_1_0000_0030_20.txt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("0")
    assert tt.cache_file_list(tmp_path) == [
        "teamtrack-trajectory/Handball/train/1st_fisheye_0-30_0.txt",
        "teamtrack-trajectory/Soccer/val/F_1_0000_0030_20.txt"]


def test_mask_teamtrack_zeros_writes_copies_and_repoints_only_teamtrack(tmp_path):
    mz = _script("mask_teamtrack_zeros")
    src, old, fixed, new = (tmp_path / d for d in ("clips", "pool", "fixed", "pool2"))
    xy = np.arange(1, 41, dtype=np.float32).reshape(4, 5, 2)
    tt = xy.copy()
    tt[:, 2] = 0.0          # a whole-window (0, 0) track
    tt[1, 4] = 0.0          # one missing point
    tt[2, 0, 0] = 0.0       # x = 0 alone is a position
    Clip(clip_id="teamtrack-1st-a", sport="handball", source="teamtrack", match_id="tt-handball",
         fps=5.0, xy=tt, tags=["t"], meta={"start_frame": 3}).save(src)
    Clip(clip_id="eigd-x", sport="handball", source="eigd", match_id="g", fps=5.0,
         xy=np.zeros((4, 5, 2), np.float32)).save(src)
    old.mkdir()
    for p in sorted(src.glob("*.npz")):
        os.symlink(p, old / p.name)
    before = {p.name: p.read_bytes() for p in src.glob("*.npz")}
    counts = mz.rebuild(old, fixed, new)
    assert counts["teamtrack"] == 1 and counts["points_masked"] == 5 and counts["clips"] == 2
    assert counts["by_sport"] == {"handball": {"clips": 1, "with_zero": 1, "points": 5}}
    assert {p.name: p.read_bytes() for p in src.glob("*.npz")} == before  # sources intact
    assert os.path.realpath(new / "teamtrack-1st-a.npz") == str((fixed / "teamtrack-1st-a.npz").resolve())
    assert os.path.realpath(new / "eigd-x.npz") == str((src / "eigd-x.npz").resolve())
    got = {c.clip_id: c for c in load_clips(new)}
    a = got["teamtrack-1st-a"]
    assert np.isnan(a.xy[:, 2]).all() and np.isnan(a.xy[1, 4]).all()
    assert a.xy[2, 0, 0] == 0.0 and a.xy[2, 0, 1] == tt[2, 0, 1]
    keep = ~np.isnan(a.xy)
    assert np.array_equal(a.xy[keep], tt[keep])
    assert (a.clip_id, a.match_id, a.tags) == ("teamtrack-1st-a", "tt-handball", ["t"])
    assert a.meta == {"start_frame": 3, "missing_zero_masked": 5}
    assert (got["eigd-x"].xy == 0).all()  # only TeamTrack is touched
    with pytest.raises(SystemExit):
        mz.rebuild(old, fixed, new)  # never over an existing pool


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
                preset="strict_smooth", reprs=("sheet", "text", "trails", "video"),
                candidates=None, drop_cells=()):
    from motion_sport.pipeline import VALID

    root = tmp_path / "final"
    root.mkdir()
    sports = ["american_football", "basketball", "handball", "soccer"]
    ctrl = {"n_players": 10, "player_mode": "random", "space": "spread", "rotate": True,
            "tempo": None, "n_frames": 20, "drop_team": True, "smooth": 2.0, "max_speed_ms": 12.0,
            "drop_frozen": True, "drop_linear": True, "linear_min_s": 4.0,
            "drop_duplicates": True, "duplicate_tol_m": 0.05}
    ctrl.update(controls or {})
    (root / "config.json").write_text(json.dumps({
        "preset": preset, "controls": ctrl, "candidates": list(candidates or sports),
        "clips_kept_by_sport": {s: 150 for s in sports}}))
    # like prepare: every condition in every representation that can show it
    items = [{"clip_id": f"{s}-{k}", "sport": s, "condition": c, "repr": r,
              "prompt_informed": "...", "tags": list(af_tags) if s == "american_football" else []}
             for s in sports for k in range(3) for r in reprs for c in VALID
             if r in VALID[c] and (c, r) not in drop_cells]
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
    # pinned exactly (2026-09-28): a preset edit or a stale set must not slip through
    ({"controls": {"n_frames": 40}}, "controls.n_frames is 40, pre-registered 20"),
    ({"controls": {"n_frames": None}}, "controls.n_frames is None"),
    ({"controls": {"max_speed_ms": 15.0}}, "controls.max_speed_ms is 15.0, pre-registered 12.0"),
    ({"controls": {"max_speed_ms": 12.5}}, "controls.max_speed_ms is 12.5"),
    ({"controls": {"max_speed_ms": 0}}, "controls.max_speed_ms is 0"),
    ({"controls": {"max_speed_ms": True}}, "controls.max_speed_ms is True"),
    ({"controls": {"smooth": 1.0}}, "controls.smooth is 1.0, pre-registered 2.0"),
    ({"controls": {"smooth": None}}, "controls.smooth is None"),
    # D18: frozen tracks are not players
    ({"controls": {"drop_frozen": False}}, "controls.drop_frozen is False, pre-registered True"),
    ({"controls": {"drop_frozen": None}}, "controls.drop_frozen is None"),
    ({"controls": {"drop_frozen": 1}}, "controls.drop_frozen is 1"),
    # D19: one player under two ids is one player
    ({"controls": {"drop_duplicates": False}},
     "controls.drop_duplicates is False, pre-registered True (D19)"),
    ({"controls": {"drop_duplicates": None}}, "controls.drop_duplicates is None"),
    ({"controls": {"drop_duplicates": 1}}, "controls.drop_duplicates is 1"),
    ({"controls": {"duplicate_tol_m": 0.1}}, "controls.duplicate_tol_m is 0.1, pre-registered 0.05"),
    ({"controls": {"duplicate_tol_m": None}}, "controls.duplicate_tol_m is None"),
    ({"controls": {"duplicate_tol_m": True}}, "controls.duplicate_tol_m is True"),
    ({"controls": {"drop_linear": False}}, "controls.drop_linear is False, pre-registered True (D19)"),
    ({"controls": {"drop_linear": None}}, "controls.drop_linear is None"),
    ({"controls": {"linear_min_s": 2.0}}, "controls.linear_min_s is 2.0, pre-registered 4.0 (D19)"),
    ({"controls": {"linear_min_s": True}}, "controls.linear_min_s is True"),
    ({"candidates": ["american_football", "basketball", "handball", "soccer", "rugby_union"]},
     "candidates are"),
    ({"candidates": ["basketball", "handball", "soccer"]}, "candidates are"),
    ({"drop_cells": {("formation", "sheet")}}, "missing cells [('formation', 'sheet')]"),
    ({"drop_cells": {("kinematics_solo", "text")}}, "missing cells [('kinematics_solo', 'text')]"),
    ({"drop_cells": {("motion_shuffled", "video")}}, "missing cells [('motion_shuffled', 'video')]"),
])
def test_preflight_rejects_what_is_not_preregistered(tmp_path, kw, expect):
    bad = _script("run_plan").preflight(_final_like(tmp_path, **kw), 400, MODELS)
    assert len(bad) == 1 and bad[0].startswith(expect)


def test_preflight_rejects_a_set_prepared_before_d18(tmp_path):
    root = _final_like(tmp_path)
    cfg = json.loads((root / "config.json").read_text())
    del cfg["controls"]["drop_frozen"]  # config.json written before D18
    (root / "config.json").write_text(json.dumps(cfg))
    assert _script("run_plan").preflight(root, 400, MODELS) == [
        "controls.drop_frozen is None, pre-registered True (D18)"]


@pytest.mark.parametrize("which", ["final", "a7b"])
def test_preflight_rejects_a_kept_player_that_never_moves(tmp_path, which):
    rp = _script("run_plan")
    root = _final_like(tmp_path) if which == "final" else _d8_like(tmp_path)
    check = (lambda: rp.preflight(root, 400, MODELS)) if which == "final" else \
        (lambda: rp.preflight_a7b(root, 300))
    rng = np.random.default_rng(0)
    moving = rng.normal(size=(20, 10, 2)).astype(np.float32)
    Clip(clip_id="ok", sport="soccer", source="metrica", match_id="m", fps=5.0,
         xy=moving, meta={"space_scale": 5.0}).save(root / "clips")
    assert check() == []
    frozen = moving.copy()
    frozen[:, 3] = frozen[0, 3]  # one dot that never moves, among moving ones
    Clip(clip_id="teamtrack-x", sport="handball", source="teamtrack", match_id="m", fps=5.0,
         xy=frozen, meta={"space_scale": 5.0}).save(root / "clips")
    bad = check()
    assert len(bad) == 1 and bad[0].startswith("1 clips with a kept player that never moves")
    assert "teamtrack-x (1)" in bad[0]


def test_preflight_rejects_a_set_prepared_before_d19(tmp_path):
    root = _final_like(tmp_path)
    cfg = json.loads((root / "config.json").read_text())
    for k in ("drop_linear", "linear_min_s", "drop_duplicates", "duplicate_tol_m"):
        del cfg["controls"][k]  # config.json written before D19
    (root / "config.json").write_text(json.dumps(cfg))
    assert _script("run_plan").preflight(root, 400, MODELS) == [
        "controls.drop_linear is None, pre-registered True (D19)",
        "controls.linear_min_s is None, pre-registered 4.0 (D19)",
        "controls.drop_duplicates is None, pre-registered True (D19)",
        "controls.duplicate_tol_m is None, pre-registered 0.05 (D19)"]


@pytest.mark.parametrize("which", ["final", "a7b"])
def test_preflight_rejects_two_kept_players_within_5_cm(tmp_path, which):
    """D19: checked on the prepared clips in metres (post-control units x space_scale)."""
    rp = _script("run_plan")
    root = _final_like(tmp_path) if which == "final" else _d8_like(tmp_path)
    check = (lambda: rp.preflight(root, 400, MODELS)) if which == "final" else \
        (lambda: rp.preflight_a7b(root, 300))
    xy = np.random.default_rng(0).normal(size=(20, 10, 2)).astype(np.float32)
    xy[:, 7] = xy[:, 2] + [0.004, 0.0]       # 0.004 units apart every frame
    far = Clip(clip_id="close-but-real", sport="soccer", source="metrica", match_id="m",
               fps=5.0, xy=xy, meta={"space_scale": 20.0})  # 8 cm: two players
    far.save(root / "clips")
    assert check() == []
    far.replace(clip_id="sportvu-x", sport="basketball", source="sportvu",
                meta={"space_scale": 10.0}).save(root / "clips")  # 4 cm: one player twice
    bad = check()
    assert len(bad) == 1 and bad[0].startswith(
        "1 clips with two kept players within 0.05 m over the whole window (D19)")
    assert "sportvu-x (1)" in bad[0]


def test_preflight_accepts_integral_values_and_skips_video_without_gemini(tmp_path):
    rp = _script("run_plan")
    root = _final_like(tmp_path, controls={"max_speed_ms": 12, "smooth": 2},
                       drop_cells={("motion", "video")})
    assert rp.preflight(root, 400, ["azure-openai:gpt-5.6-sol"]) == []
    assert rp.preflight(root, 400, MODELS) == ["missing cells [('motion', 'video')]"]


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
            "max_speed_ms": 12.0, "drop_frozen": True, "drop_linear": True, "linear_min_s": 4.0,
            "drop_duplicates": True, "duplicate_tol_m": 0.05}
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


@pytest.mark.parametrize("controls, expect", [
    ({"max_speed_ms": 15.0}, "controls.max_speed_ms is 15.0, pre-registered 12.0"),
    ({"max_speed_ms": None}, "controls.max_speed_ms is None, pre-registered 12.0"),
    ({"max_speed_ms": 0}, "controls.max_speed_ms is 0, pre-registered 12.0"),
    ({"smooth": 1.0}, "controls.smooth is 1.0, pre-registered 2.0"),
    ({"smooth": None}, "controls.smooth is None, pre-registered 2.0"),
])
def test_preflight_a7b_pins_speed_ceiling_and_smoothing(tmp_path, controls, expect):
    rp = _script("run_plan")
    root = _d8_like(tmp_path)
    cfg = json.loads((root / "config.json").read_text())
    cfg["controls"].update(controls)
    (root / "config.json").write_text(json.dumps(cfg))
    assert rp.preflight_a7b(root, 300) == [expect]


# ------------------------------------------------------------- final_run.sh (sandboxed)

FAKE_PYTHON = """#!/usr/bin/env bash
echo "$*" >> "$SANDBOX_LOG/python.log"
case "$2" in
  write) cat > /dev/null; [ -n "${FAIL_WRITE:-}" ] && exit 1; echo "plan written" ;;
  preflight*) [ -n "${FAIL_PREFLIGHT:-}" ] && exit 1; echo "preflight ok" ;;
  check) echo "no unrecovered errors" ;;
esac
exit 0
"""
FAKE_MOTION_SPORT = """#!/usr/bin/env bash
echo "$*" >> "$SANDBOX_LOG/calls.log"
case "$*" in *kinematics_solo*) exit 3 ;; esac   # one failing cell: not fatal
exit 0
"""


def _sandbox(tmp_path):
    """A copy of final_run.sh whose .venv/bin/{python,motion-sport} only log their calls."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "final_run.sh").write_text((SCRIPTS / "final_run.sh").read_text())
    for name, body in (("python", FAKE_PYTHON), ("motion-sport", FAKE_MOTION_SPORT)):
        f = repo / ".venv" / "bin" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
        f.chmod(0o755)
    (tmp_path / "log").mkdir()
    return repo


def _launch(tmp_path, repo, **env):
    import subprocess

    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),  # no real key file is read
            "SANDBOX_LOG": str(tmp_path / "log"), "AZURE_OPENAI_KEY": "x",
            "OPENROUTER_API_KEY": "x", "PASSES": "1",
            "MODELS": "azure-openai:gpt-5.6-sol vertex:gemini-3.1-pro-preview"}
    return subprocess.run(["bash", str(repo / "scripts" / "final_run.sh")], capture_output=True,
                          text=True, env={**base, **env}, timeout=120)


def _calls(tmp_path):
    f = tmp_path / "log" / "calls.log"
    return f.read_text().splitlines() if f.exists() else []


@pytest.mark.parametrize("fail", ["FAIL_WRITE", "FAIL_PREFLIGHT"])
def test_final_run_aborts_before_any_model_call_when_bookkeeping_fails(tmp_path, fail):
    repo = _sandbox(tmp_path)
    r = _launch(tmp_path, repo, **{fail: "1"})
    assert r.returncode != 0
    assert _calls(tmp_path) == []
    if fail == "FAIL_WRITE":
        assert "write_plan failed: no model was called" in r.stderr
        assert not (repo / "runs" / "final" / "logs").exists()


def test_final_run_survives_a_failing_cell_and_runs_every_command(tmp_path):
    repo = _sandbox(tmp_path)
    r = _launch(tmp_path, repo)
    assert r.returncode == 0, r.stderr
    assert "ALL DONE" in r.stdout
    calls = _calls(tmp_path)
    # the kinematics_solo cell fails for both models; everything after it still runs
    assert len(calls) == 14 + 16
    for m in ("azure-openai:gpt-5.6-sol", "vertex:gemini-3.1-pro-preview"):
        mine = [c for c in calls if f"--model {m} " in c]
        assert "--items runs/final-d8" in mine[-1] and "--items runs/final-d8" in mine[-2]
    log = (tmp_path / "log" / "python.log").read_text().splitlines()
    assert [line for line in log if " write " in line] == [
        "scripts/run_plan.py write runs/final",
        "scripts/run_plan.py write runs/final-d8 --exploratory"]  # A7b is exploratory
    assert sum(" check " in line for line in log) == 2


def _dry_run_cells(tmp_path, models):
    repo = _sandbox(tmp_path)
    r = _launch(tmp_path, repo, DRY_RUN="1", MODELS=models)
    assert r.returncode == 0, r.stderr
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("#"):
            continue
        w = line.split()
        opt = {w[i]: w[i + 1] for i in range(2, len(w) - 1) if w[i].startswith("--")}
        out.append((opt["--condition"], opt["--repr"], opt.get("--prompt-style", "neutral"),
                    int(opt.get("--replicate", 1)), int(opt["--limit"]),
                    "d8" if opt["--items"].endswith("final-d8") else "final"))
    assert not _calls(tmp_path)  # a dry run calls nothing
    return out


def test_final_run_order_primary_then_secondaries_then_a7b_last(tmp_path):
    primary = [("motion", "sheet", "neutral", 1, 400), ("motion_shuffled", "sheet", "neutral", 1, 400),
               ("formation", "sheet", "neutral", 1, 400), ("motion", "text", "neutral", 1, 400),
               ("motion", "sheet", "informed", 1, 400)]
    secondary = [("kinematics", "sheet", "neutral", 1, 400),
                 ("kinematics_solo", "sheet", "neutral", 1, 400), ("motion", "trails", "neutral", 1, 400)]
    video = [("motion", "video", "neutral", 1, 400), ("motion_shuffled", "video", "neutral", 1, 400)]
    replicates = [(c, "sheet", "neutral", k, 200) for k in (2, 3) for c in ("motion", "motion_shuffled")]
    a7b = [("motion", "sheet", "neutral", 1, 300, "d8"), ("motion_shuffled", "sheet", "neutral", 1, 300, "d8")]
    fin = lambda cells: [(*c, "final") for c in cells]  # noqa: E731
    assert _dry_run_cells(tmp_path / "chat", "azure-openai:gpt-5.6-sol") == \
        fin(primary + secondary + replicates) + a7b
    assert _dry_run_cells(tmp_path / "gemini", "vertex:gemini-3.1-pro-preview") == \
        fin(primary + secondary + video + replicates) + a7b


def test_estimate_run_plan_matches_the_launcher_order(tmp_path):
    est = _script("estimate_run")
    for m in ("azure-openai:gpt-5.6-sol", "vertex:gemini-3.1-pro-preview"):
        want = [(c, r, s, k, n, items) for c, r, s, k, n, _fmt, items in est.plan()[m]]
        assert _dry_run_cells(tmp_path / m.replace(":", "_"), m) == want


def test_exploratory_plan_is_sticky_and_reaches_the_view(tmp_path):
    from motion_sport import pipeline

    pv, rp = _script("primary_view"), _script("run_plan")
    items = _prepared(tmp_path)
    line = (f"motion-sport run --items {items} --model azure-openai:gpt-5.6-sol --condition {{}} "
            "--repr sheet --limit 10")
    rp.write_plan(items, [line.format("motion")], exploratory=True)
    rp.write_plan(items, [line.format("motion_shuffled")])  # a relaunch keeps the mark
    plan = json.loads((items / "plan.json").read_text())
    assert plan["exploratory"] is True and len(plan["cells"]) == 2
    for cond in ("motion", "motion_shuffled"):
        pipeline.run_model(str(items), "azure-openai:gpt-5.6-sol", condition=cond, rep="sheet",
                           limit=10, complete=_fake)
    info = pv.build_view(items, tmp_path / "view")
    assert info["exploratory"] and pipeline.is_exploratory(tmp_path / "view")
    rep = pipeline.report(str(tmp_path / "view"), n_boot=20)
    assert rep["primary_contrasts"] == [] and any(c["contrast"] == "order" for c in rep["contrasts"])
    # a plain directory is not exploratory
    other = tmp_path / "other"
    other.mkdir()
    rp.write_plan(other, [])
    assert "exploratory" not in json.loads((other / "plan.json").read_text())
