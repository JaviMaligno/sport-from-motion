"""Kept-player notes behind D18 / D19 (what the controls leave in, measured):

  - partial holds: a kept track whose position is *exactly* constant for >= 5 consecutive
    frames (1 s at 5 Hz) but not for the whole window (the whole window is `drop_frozen`,
    >= 4 s is `drop_linear`). Shown, it is a player who stops.
  - near-still: a kept track that moves less than 5 cm in the whole window (largest range
    of x or y) without being exactly constant: a player standing, with sensor noise.
  - filled frames: window frames of a kept track that were NaN in the raw clip and were
    filled by `fill_short_gaps` (interior gaps of <= 3 frames, linear).

    python scripts/analysis/track_notes.py runs/final --pool data/pool_final4 --first 400 \
        [--json out.json]

Replays the kept players from the raw clips with prepare's seed and the set's controls
(`controls.select_players`), in metres over the window.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from motion_sport.controls import ControlConfig, select_players, window_span
from motion_sport.pipeline import interleave, stable_seed
from motion_sport.schema import Clip

HOLD_MIN = 5          # frames
STILL_RANGE_M = 0.05  # metres over the window


def longest_hold(x: np.ndarray) -> int:
    """x [T, 2] -> longest run of frames with exactly the same position."""
    same = (np.diff(x, axis=0) == 0).all(axis=1)
    best = run = 0
    for s in same:
        run = run + 1 if s else 0
        best = max(best, run)
    return best + 1 if best else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--first", type=int, default=400)
    ap.add_argument("--json")
    a = ap.parse_args()
    root, pool = pathlib.Path(a.items), pathlib.Path(a.pool)
    cfg = json.loads((root / "config.json").read_text())
    ctl = dict(cfg["controls"])
    for k in ("drop_frozen", "drop_linear", "drop_duplicates"):
        ctl.setdefault(k, False)
    cc = ControlConfig(**ctl)
    rows = [json.loads(line) for line in (root / "items.jsonl").open()]
    order = [it["clip_id"] for it in interleave(
        [it for it in rows if it["condition"] == "motion" and it["repr"] == "sheet"])]
    cnt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ex: dict[str, list] = collections.defaultdict(list)
    for pos, cid in enumerate(order):
        raw = Clip.load(pool / f"{cid}.npz")
        span = window_span(raw.n_frames, cc.n_frames)
        k = select_players(raw, cc, np.random.default_rng(stable_seed(cid, str(cfg.get("seed", 0)))))
        # which raw tracks were kept: match the kept (filled) tracks to the raw ones where observed
        w = k.xy[span].astype(np.float64)
        r = raw.xy[span].astype(np.float64)
        s = cnt[raw.source]
        scopes = ["all"] + (["first"] if pos < a.first else [])
        for sc in scopes:
            cnt[f"{raw.source}|{sc}"]["kept_players"] += k.n_players
        for j in range(k.n_players):
            x = w[:, j]
            # raw track: the one identical to x on every frame the raw track is observed
            ok = ~np.isnan(r).any(axis=2)                                    # [T, N]
            eq = np.where(ok, (r == x[:, None]).all(axis=2), True).all(axis=0) & ok.any(axis=0)
            src = np.flatnonzero(eq)
            filled = int((~ok[:, src[0]]).sum()) if len(src) else -1
            hold = longest_hold(x)
            rng = float(np.ptp(x, axis=0).max())
            for sc in scopes:
                t = cnt[f"{raw.source}|{sc}"]
                if filled > 0:
                    t["players_with_filled"] += 1
                    t["filled_frames"] += filled
                if filled < 0:
                    t["unmatched"] += 1
                if HOLD_MIN <= hold < x.shape[0]:
                    t["partial_holds"] += 1
                if rng < STILL_RANGE_M and hold < x.shape[0]:
                    t["near_still"] += 1
            if HOLD_MIN <= hold < x.shape[0] and len(ex["hold", raw.source]) < 6:
                ex["hold", raw.source].append((pos, cid, j, hold, round(rng, 3)))
            if rng < STILL_RANGE_M and len(ex["still", raw.source]) < 6:
                step = np.linalg.norm(np.diff(x, axis=0), axis=1)
                ex["still", raw.source].append((pos, cid, j, round(rng, 4),
                                                round(float(np.median(step)) * 1000, 2)))
        del s
    frames = cc.n_frames
    out = {"by_source_scope": {k: dict(v) for k, v in sorted(cnt.items()) if "|" in k},
           "examples": {f"{a_}/{b_}": v for (a_, b_), v in sorted(ex.items())},
           "n_frames": frames, "first": a.first}
    for k, v in out["by_source_scope"].items():
        kp = v.get("kept_players", 0)
        print(f"{k:22s} kept={kp} filled_players={v.get('players_with_filled', 0)} "
              f"filled_frames={v.get('filled_frames', 0)} "
              f"({v.get('filled_frames', 0) / max(kp * frames, 1):.4%} of frames) "
              f"partial_holds={v.get('partial_holds', 0)} near_still={v.get('near_still', 0)} "
              f"unmatched={v.get('unmatched', 0)}")
    for k, v in out["examples"].items():
        print(k, v)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
