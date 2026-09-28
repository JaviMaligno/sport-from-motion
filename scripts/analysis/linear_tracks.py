"""Audit of exactly linear stretches: stretches where a track moves in a straight line at
constant speed (second difference ~ 0), the signature of a linear interpolation (D19).

A measured player never moves at exactly constant velocity for long: detector / sensor
noise and real acceleration give a second difference of millimetres to centimetres per
5 Hz step. A stretch with |second difference| < TOL in both axes for many frames is
interpolated, not measured.

    python scripts/analysis/linear_tracks.py pool data/pool_final4 [--n-frames 20] [--json out.json]
  Per source and sport: fully observed tracks; their longest exactly linear run (in frames
  of the window, i.e. run of near-zero second differences + 2) by bin; share of all
  track-frames that lie inside a linear run of >= 5 frames (1 s at 5 Hz); the tracks whose
  whole window is one straight line at constant speed.
    python scripts/analysis/linear_tracks.py items runs/final --pool data/pool_final4
  The same on the kept players (replayed with `controls.select_players`), per source and
  in the first N interleaved clips (--first).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from motion_sport.controls import (ControlConfig, drop_frozen, fill_short_gaps, select_players,
                                   window_span)
from motion_sport.pipeline import interleave, stable_seed
from motion_sport.schema import Clip

TOL = 1e-4  # metres per 5 Hz step (0.1 mm); float32 noise at pitch coordinates is ~1e-5
RUN_BINS = [3, 5, 10, 15, 20, 25, 30, 40]


def linear_runs(xy: np.ndarray, tol: float = TOL) -> tuple[np.ndarray, np.ndarray]:
    """xy [T, N, 2] -> (longest exactly linear run per track in frames [N], mask [T, N] of
    frames inside a linear run of >= 5 frames). NaN tracks give 0."""
    t, n = xy.shape[:2]
    a = np.abs(np.diff(xy.astype(np.float64), 2, axis=0)).max(axis=2)  # [T-2, N]
    flat = a < tol  # NaN -> False
    longest = np.zeros(n, int)
    mask = np.zeros((t, n), bool)
    for j in range(n):
        run = 0
        for k in range(t - 1):  # sentinel at the end
            if k < t - 2 and flat[k, j]:
                run += 1
                continue
            if run:
                frames = run + 2
                longest[j] = max(longest[j], frames)
                if frames >= 5:
                    mask[k - run:k + 2, j] = True
            run = 0
    return longest, mask


def _summ(longest: list[int], masks: list[np.ndarray], t: int) -> dict:
    lg = np.array(longest)
    edges = RUN_BINS + [t + 1]
    hist = {f">={a}": int((lg >= a).sum()) for a in edges[:-1]}
    frames = sum(m.size for m in masks)
    inside = sum(int(m.sum()) for m in masks)
    return {"tracks": len(lg), "longest_run_at_least": hist,
            "tracks_whole_window_linear": int((lg >= t).sum()),
            "share_frames_in_run_ge5": inside / frames if frames else 0.0}


def audit_pool(pool: pathlib.Path, n_frames: int) -> dict:
    lg: dict[str, list[int]] = collections.defaultdict(list)
    ms: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for p in sorted(pool.glob("*.npz")):
        c = Clip.load(p)
        span = window_span(c.n_frames, n_frames)
        f, _ = drop_frozen(fill_short_gaps(c), span)
        w = f.xy[span]
        full = ~np.isnan(w).any(axis=(0, 2))
        longest, mask = linear_runs(w[:, full])
        key = f"{c.source}/{c.sport}"
        lg[key].extend(longest.tolist())
        ms[key].append(mask)
    return {k: _summ(lg[k], ms[k], n_frames) for k in sorted(lg)}


def audit_items(items: pathlib.Path, pool: pathlib.Path, n_first: int) -> dict:
    cfg = json.loads((items / "config.json").read_text())
    ctl = dict(cfg["controls"])
    ctl.setdefault("drop_frozen", False)
    ctl.setdefault("drop_duplicates", False)
    ctl.setdefault("drop_linear", False)
    cc = ControlConfig(**ctl)
    rows = [json.loads(line) for line in (items / "items.jsonl").open()]
    ms_ = [it for it in rows if it["condition"] == "motion" and it["repr"] == "sheet"]
    order = [it["clip_id"] for it in interleave(ms_)]
    first = set(order[:n_first])
    out: dict[str, dict] = {}
    worst = []
    for scope in ("all", f"first{n_first}"):
        lg: dict[str, list[int]] = collections.defaultdict(list)
        mk: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        for pos, cid in enumerate(order):
            if scope != "all" and cid not in first:
                continue
            raw = Clip.load(pool / f"{cid}.npz")
            k = select_players(raw, cc, np.random.default_rng(stable_seed(cid, str(cfg.get("seed", 0)))))
            w = k.xy[window_span(k.n_frames, cc.n_frames)]
            longest, mask = linear_runs(w)
            key = f"{raw.source}/{raw.sport}"
            lg[key].extend(longest.tolist())
            mk[key].append(mask)
            if scope == "all" and longest.max() >= 10:
                worst.append((pos, cid, int(longest.max()), int((longest >= 10).sum())))
        out[scope] = {k: _summ(lg[k], mk[k], cc.n_frames) for k in sorted(lg)}
    out["clips_with_run_ge10"] = worst
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "items"])
    ap.add_argument("path")
    ap.add_argument("--pool")
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--first", type=int, default=400)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.mode == "pool":
        res = audit_pool(pathlib.Path(a.path), a.n_frames)
        for k, v in res.items():
            print(f"{k:28s} tracks={v['tracks']} whole_window_linear={v['tracks_whole_window_linear']} "
                  f"frames_in_run>=5={v['share_frames_in_run_ge5']:.4f} longest>= {v['longest_run_at_least']}")
    else:
        res = audit_items(pathlib.Path(a.path), pathlib.Path(a.pool), a.first)
        for scope in ("all", f"first{a.first}"):
            print("==", scope)
            for k, v in res[scope].items():
                print(f"  {k:28s} tracks={v['tracks']} whole_window_linear={v['tracks_whole_window_linear']} "
                      f"frames_in_run>=5={v['share_frames_in_run_ge5']:.4f} longest>= {v['longest_run_at_least']}")
        w = res["clips_with_run_ge10"]
        print(f"clips with a kept track linear for >= 10 frames: {len(w)}", w[:15])
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
