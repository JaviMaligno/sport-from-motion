"""Ramp statistic: per clip, mean player speed over the first 5 steps / over the last 5
steps, measured after the same controls as runs/final (strict_smooth, N=10 random,
12 m/s, 20 frames; the ratio is scale-free so spread normalisation does not matter).

NFL pools (raw ingests) are controlled here with prepare's per-clip seed; the other
sports are read from an already-prepared items dir (its clips/ are post-control).

    PYTHONPATH=src .venv/bin/python scripts/analysis/ramp.py <items_dir_for_others> \
        <nfl_pool_a> [<nfl_pool_b> ...] [--out ramp.json]
    # D16: scripts/analysis/ramp.py runs/final data/clips_nflrand data/clips_nflrand10 \
    #          --out runs/analysis-2026-09-28/ramp.json

The results go to --out (default: <items_dir>/ramp.json; the copy that D16 cites is
runs/analysis-2026-09-28/ramp.json, written next to the script when it lived there).
Since D18 the preset drops frozen tracks (`drop_frozen`); no NFL clip has one, so the NFL
numbers do not change.
"""
import collections
import dataclasses
import json
import pathlib
import sys

import numpy as np

from motion_sport.controls import PRESETS, apply_controls
from motion_sport.pipeline import stable_seed
from motion_sport.schema import load_clips

CFG = dataclasses.replace(PRESETS["strict_smooth"], n_players=10, player_mode="random")


def ramp(xy: np.ndarray) -> float:
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=2).mean(axis=1)  # [T-1]
    return float(sp[:5].mean() / max(sp[-5:].mean(), 1e-9))


def summary(name, rows, n_boot=2000, seed=0):
    """rows: list of (match_id, ratio). Geometric mean with a match-cluster bootstrap."""
    bad = sum(1 for _, x in rows if not (np.isfinite(x) and x > 0))
    rows = [(m, x) for m, x in rows if np.isfinite(x) and x > 0]  # a still first/last second
    r = np.array([x for _, x in rows])
    lg = np.log(r)
    by = collections.defaultdict(list)
    for m, x in rows:
        by[m].append(np.log(x))
    ms = list(by)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ms), len(ms))
        vals = np.concatenate([by[ms[k]] for k in pick])
        boots.append(np.exp(vals.mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    print(f"{name:34s} n={len(r):5d} matches={len(ms):4d}  geo-mean {np.exp(lg.mean()):.3f} "
          f"[{lo:.3f}; {hi:.3f}]  median {med:.3f} (IQR {q1:.3f}-{q3:.3f})  "
          f"share>1.25 {np.mean(r > 1.25):.2f}  share<0.8 {np.mean(r < 0.8):.2f}  degenerate {bad}")
    return {"n": int(len(r)), "matches": len(ms), "geo_mean": float(np.exp(lg.mean())),
            "ci": [float(lo), float(hi)], "median": float(med), "iqr": [float(q1), float(q3)],
            "share_gt_1_25": float(np.mean(r > 1.25)), "share_lt_0_8": float(np.mean(r < 0.8))}


def main():
    out = {}
    args = sys.argv[1:]
    dest = None
    if "--out" in args:
        k = args.index("--out")
        dest = pathlib.Path(args[k + 1])
        del args[k:k + 2]
    items = pathlib.Path(args[0])
    clips = load_clips(items / "clips")
    by_src = collections.defaultdict(list)
    for c in clips:
        by_src[f"{c.sport}/{c.source}"].append((c.match_id, ramp(c.xy)))
    print(f"-- prepared clips of {items} (post-control)")
    for k in sorted(by_src):
        out[f"{items.name}:{k}"] = summary(k, by_src[k])
    for pool in args[1:]:
        rows, off_rows = [], collections.defaultdict(list)
        rej = collections.Counter()
        for c in load_clips(pool):
            rng = np.random.default_rng(stable_seed(c.clip_id, "0"))
            cc = apply_controls(c, CFG, rng, reasons=rej)
            if cc is None:
                continue
            x = ramp(cc.xy)
            rows.append((cc.match_id, x))
            off = c.meta.get("snap_offset_s", np.nan)
            b = "<1.0s" if off < 1.0 else "1.0-2.0s" if off < 2.0 else ">=2.0s"
            off_rows[b].append((cc.match_id, x))
        print(f"-- NFL pool {pool} (controlled here; rejected {dict(rej)})")
        out[f"{pool}:all"] = summary("american_football/nfl_bdb (all)", rows)
        for b in ("<1.0s", "1.0-2.0s", ">=2.0s"):
            if off_rows[b]:
                out[f"{pool}:{b}"] = summary(f"  offset {b}", off_rows[b])
    (dest or items / "ramp.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
