"""Audit of missing-as-zero and frozen tracks (D18).

Two artefacts that make a track look observed when it is not:
  - exact (0.0, 0.0): TeamTrack writes a missing detection as the origin;
  - frozen: a fully observed track whose position is *exactly* constant over the clip.
    A real player's measured position always moves a little (sensor / detector noise),
    so an exactly constant track is a held or placeholder value, not a player.

Pool mode (raw ingested clips, metres), per source:
    python scripts/analysis/frozen_tracks.py pool data/pool_final3 [--json out.json]
  clips; clips with any exact (0,0) point; (0,0) points; tracks entirely (0,0);
  fully observed tracks; frozen tracks (constant over the whole clip); clips with at
  least one frozen track; clips where every fully observed track is frozen
  ("all_frozen", i.e. a static stoppage or a frozen feed), with their ids.

Items mode (a prepared set: its clips/ are post-control, 10 kept players), per source:
    python scripts/analysis/frozen_tracks.py items runs/final [--pool data/pool_final4]
  kept players whose post-control position does not move (range < 1e-6 in units of
  spread); with --pool, the kept players are also replayed from the raw pool clips with
  prepare's per-clip seed and `controls.select_players` (the same draws as `prepare`, with
  the set's own controls; a config.json without `drop_frozen` predates D18 and is replayed
  without it) and checked in metres over the window, including any exact (0,0) point
  that reached a kept player.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from motion_sport.controls import ControlConfig, frozen_tracks, select_players, window_span
from motion_sport.pipeline import stable_seed
from motion_sport.schema import Clip

POST_TOL = 1e-6  # post-control units (unit spread): a moving player is ~1e-2 or more


def zero_mask(xy: np.ndarray) -> np.ndarray:
    """[T, N] True where the point is exactly (0, 0)."""
    return (xy[..., 0] == 0) & (xy[..., 1] == 0)


def audit_pool(pool: pathlib.Path) -> dict:
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ids: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(pool.glob("*.npz")):
        c = Clip.load(p)
        s = out[c.source]
        s["clips"] += 1
        z = zero_mask(c.xy)
        s["zero_points"] += int(z.sum())
        if z.any():
            s["clips_with_zero"] += 1
            ids[c.source]["zero"].append(c.clip_id)
        s["tracks_all_zero"] += int(z.all(axis=0).sum())
        full = ~np.isnan(c.xy).any(axis=(0, 2))
        s["tracks_full"] += int(full.sum())
        fr = frozen_tracks(c.xy)
        s["tracks_frozen"] += int(fr.sum())
        s["tracks_frozen_nonzero"] += int((fr & ~z.all(axis=0)).sum())
        if fr.any():
            s["clips_with_frozen"] += 1
            ids[c.source]["frozen"].append(c.clip_id)
        if full.any() and fr[full].all():
            s["clips_all_frozen"] += 1
            ids[c.source]["all_frozen"].append(c.clip_id)
    return {src: {**dict(v), "ids_all_frozen": ids[src]["all_frozen"],
                  "ids_frozen_first20": ids[src]["frozen"][:20]} for src, v in sorted(out.items())}


def audit_items(items: pathlib.Path, pool: pathlib.Path | None) -> dict:
    cfg = json.loads((items / "config.json").read_text())
    ctl = dict(cfg["controls"])
    ctl.setdefault("drop_frozen", False)  # sets prepared before D18
    ccfg = ControlConfig(**ctl)
    seed = str(cfg.get("seed", 0))
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    bad: list[dict] = []
    for p in sorted((items / "clips").glob("*.npz")):
        c = Clip.load(p)
        s = out[c.source]
        s["clips"] += 1
        s["kept_players"] += c.n_players
        rng = np.ptp(c.xy, axis=0).max(axis=1)  # [N] largest range of x or y
        still = rng < POST_TOL
        s["kept_players_still_post"] += int(still.sum())
        if still.any():
            bad.append({"clip_id": c.clip_id, "check": "post", "n": int(still.sum())})
        if pool is None:
            continue
        raw = Clip.load(pool / p.name)
        r = np.random.default_rng(stable_seed(raw.clip_id, seed))
        k = select_players(raw, ccfg, r)
        if k is None or k.n_players != c.n_players:
            bad.append({"clip_id": c.clip_id, "check": "replay_mismatch"})
            s["replay_mismatch"] += 1
            continue
        win = k.xy[window_span(k.n_frames, ccfg.n_frames)]
        fr = frozen_tracks(win)
        zz = zero_mask(win).any(axis=0)
        s["kept_players_frozen_raw"] += int(fr.sum())
        s["kept_players_with_zero_raw"] += int(zz.sum())
        if fr.any() or zz.any():
            bad.append({"clip_id": c.clip_id, "check": "raw", "frozen": int(fr.sum()),
                        "zero": int(zz.sum())})
    return {"by_source": {k: dict(v) for k, v in sorted(out.items())}, "flagged": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "items"])
    ap.add_argument("path")
    ap.add_argument("--pool", help="items mode: replay the kept players from these raw clips")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.mode == "pool":
        res = audit_pool(pathlib.Path(a.path))
        for src, v in res.items():
            print(f"{src:15s} " + " ".join(f"{k}={v[k]}" for k in (
                "clips", "clips_with_zero", "zero_points", "tracks_all_zero", "tracks_full",
                "tracks_frozen", "tracks_frozen_nonzero", "clips_with_frozen",
                "clips_all_frozen") if k in v) + (f" all_frozen={v['ids_all_frozen']}"
                                                   if v["ids_all_frozen"] else ""))
    else:
        res = audit_items(pathlib.Path(a.path), pathlib.Path(a.pool) if a.pool else None)
        for src, v in res["by_source"].items():
            print(f"{src:15s} " + " ".join(f"{k}={n}" for k, n in v.items()))
        print(f"flagged clips: {len(res['flagged'])}", res["flagged"][:20])
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
