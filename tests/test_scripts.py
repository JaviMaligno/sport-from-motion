"""Tests for the helper scripts in scripts/ (loaded by path: they are not a package)."""
import importlib.util
import json
import os
import pathlib

import numpy as np

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
