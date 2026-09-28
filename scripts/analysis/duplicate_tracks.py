"""Audit of duplicated tracks: one player under two ids (D19).

Some sources write the same player twice, with two ids that share the coordinates (e.g.
SportVU game 0021500368, segment 12: players 2755 and 2749 in every moment). Shown, that
is one dot drawn twice, and the clip has one real player fewer than it seems.

Pool mode (raw ingested clips, metres), per source and sport:
    python scripts/analysis/duplicate_tracks.py pool data/pool_final4 [--n-frames 20] [--json out.json]
  For every clip, the same steps as the controls before choosing the players
  (`fill_short_gaps`, `drop_frozen`), then, over the window the set would use, the largest
  distance each pair of fully observed tracks ever reaches (`controls.pair_max_distance`).
  Reports the pairs by bin of that distance, the per-clip minimum (the closest two tracks
  ever *stay* together in the window) by quantile, the closest pairs above the tolerance
  (with clip ids, to look at), the duplicate tracks the control would drop at
  `--tol` (default: `controls.DUPLICATE_TOL_M`), and, for every pair that is within the
  tolerance in at least one frame, in how many frames it is ("coloc_frames_hist": a
  crossing is 1-2 frames; a copy or a merge of two tracks covers most of the window).

Items mode (a prepared set), per source:
    python scripts/analysis/duplicate_tracks.py items runs/final --pool data/pool_final4
  On the prepared clips: kept pairs whose max distance in metres (post-control units x the
  clip's `space_scale`; rotation keeps distances, smoothing can only shrink them) is within
  the tolerance; kept players that never move; kept players with a step faster than
  `max_speed_ms`. With --pool, the kept players are replayed from the raw clips with
  `controls.select_players` (prepare's seed) and checked in metres over the window: pairs
  within the tolerance, exactly constant tracks, exact (0, 0) points, fastest step,
  tracks exactly linear for >= `linear_min_s` (controls.linear_tracks), and kept pairs
  within the tolerance in at least half the frames of the window (partial copies or
  merges, which the whole-window rule does not remove; reported, not a check).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from motion_sport.controls import (DUPLICATE_TOL_M, ControlConfig, drop_frozen,
                                   duplicate_tracks, fill_short_gaps, frozen_tracks,
                                   linear_tracks, max_step_speed, pair_max_distance,
                                   select_players, window_span)
from motion_sport.pipeline import interleave, stable_seed
from motion_sport.schema import Clip

BINS_M = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0]
QUANTILES = [0.0, 0.001, 0.01, 0.05, 0.10, 0.50]


def audit_pool(pool: pathlib.Path, n_frames: int, tol: float, n_closest: int = 15) -> dict:
    pairs: dict[str, list[float]] = collections.defaultdict(list)  # pair max < 2 m
    clip_min: dict[str, list[float]] = collections.defaultdict(list)
    closest: dict[str, list[tuple[float, str, int, int]]] = collections.defaultdict(list)
    cnt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    dup_ids: dict[str, list[str]] = collections.defaultdict(list)
    coloc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in sorted(pool.glob("*.npz")):
        c = Clip.load(p)
        key = f"{c.source}/{c.sport}"
        span = window_span(c.n_frames, n_frames)
        f = fill_short_gaps(c)
        f, _ = drop_frozen(f, span)
        win = f.xy[span]
        d = pair_max_distance(win)
        iu = np.triu_indices(d.shape[0], 1)
        v = d[iu]
        ok = np.isfinite(v)
        s = cnt[key]
        s["clips"] += 1
        s["pairs"] += int(ok.sum())
        if not ok.any():
            continue
        full = ~np.isnan(win).any(axis=(0, 2))
        g = win[:, full].astype(np.float64)
        kf = (np.linalg.norm(g[:, :, None] - g[:, None, :], axis=-1) <= tol).sum(axis=0)
        kf = kf[np.triu_indices(g.shape[1], 1)]
        coloc[key].update(int(x) for x in kf[kf > 0])
        pairs[key].extend(v[ok & (v < BINS_M[-1])].tolist())
        clip_min[key].append(float(v[ok].min()))
        above = ok & (v > tol)
        for k in np.flatnonzero(above):
            if v[k] < 0.5:
                closest[key].append((float(v[k]), c.clip_id, int(iu[0][k]), int(iu[1][k])))
        dup = duplicate_tracks(win, tol)
        if dup.any():
            s["duplicate_tracks"] += int(dup.sum())
            s["clips_with_duplicate"] += 1
            s["pairs_exactly_equal"] += int((v[ok] == 0).sum())
            dup_ids[key].append(c.clip_id)
    out = {}
    for key in sorted(cnt):
        pv = np.array(pairs[key])
        cm = np.array(clip_min[key])
        hist = {f"[{a:g},{b:g})": int(((pv >= a) & (pv < b)).sum())
                for a, b in zip(BINS_M[:-1], BINS_M[1:])}
        # the per-clip minimum over pairs that are NOT duplicates (the genuine closest pair)
        genuine = cm[cm > tol]
        out[key] = {
            **dict(cnt[key]),
            "pair_max_hist_m": hist,
            "clip_min_quantiles_m": {f"q{q:g}": float(np.quantile(cm, q)) for q in QUANTILES}
            if len(cm) else {},
            "genuine_clip_min_quantiles_m": {f"q{q:g}": float(np.quantile(genuine, q))
                                             for q in QUANTILES} if len(genuine) else {},
            "closest_above_tol": sorted(closest[key])[:n_closest],
            "ids_with_duplicate_first30": dup_ids[key][:30],
            "coloc_frames_hist": dict(sorted(coloc[key].items())),
        }
    return out


def audit_items(items: pathlib.Path, pool: pathlib.Path | None, tol: float) -> dict:
    cfg = json.loads((items / "config.json").read_text())
    ctl = dict(cfg["controls"])
    ctl.setdefault("drop_frozen", False)  # sets prepared before D18
    ctl.setdefault("drop_duplicates", False)  # sets prepared before D19
    ctl.setdefault("drop_linear", False)
    ccfg = ControlConfig(**ctl)
    ceiling = ccfg.max_speed_ms
    seed = str(cfg.get("seed", 0))
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    bad: list[dict] = []
    closest_post: list[tuple[float, str]] = []
    rows = [json.loads(line) for line in (items / "items.jsonl").open()]
    order = [it["clip_id"] for it in interleave(
        [it for it in rows if it["condition"] == "motion" and it["repr"] == "sheet"])]
    pos = {cid: i for i, cid in enumerate(order)}
    partial: list[tuple[int, str, int, float]] = []
    for p in sorted((items / "clips").glob("*.npz")):
        c = Clip.load(p)
        s = out[c.source]
        s["clips"] += 1
        s["kept_players"] += c.n_players
        scale = float(c.meta.get("space_scale", np.nan))
        dm = pair_max_distance(c.xy) * scale
        closest_post.append((float(dm.min()), c.clip_id))
        n_close = int((np.triu(dm, 1)[np.triu_indices(c.n_players, 1)] <= tol).sum())
        s["kept_pairs_within_tol_post"] += n_close
        still = np.ptp(c.xy, axis=0).max(axis=1) < 1e-6
        s["kept_players_still_post"] += int(still.sum())
        if n_close or still.any():
            bad.append({"clip_id": c.clip_id, "check": "post", "pairs": n_close,
                        "still": int(still.sum())})
        if pool is None:
            continue
        raw = Clip.load(pool / p.name)
        k = select_players(raw, ccfg, np.random.default_rng(stable_seed(raw.clip_id, seed)))
        if k is None or k.n_players != c.n_players:
            bad.append({"clip_id": c.clip_id, "check": "replay_mismatch"})
            s["replay_mismatch"] += 1
            continue
        win = k.xy[window_span(k.n_frames, ccfg.n_frames)]
        d = pair_max_distance(win)
        n_raw = int((d[np.triu_indices(k.n_players, 1)] <= tol).sum())
        fr = frozen_tracks(win)
        zz = ((win[..., 0] == 0) & (win[..., 1] == 0)).any(axis=0)
        top = max_step_speed(k)  # the controls' check: the kept players, whole clip
        lin = linear_tracks(win, int(round(ccfg.linear_min_s * k.fps)))
        s["kept_players_linear_raw"] += int(lin.sum())
        w64 = win.astype(np.float64)
        kf = (np.linalg.norm(w64[:, :, None] - w64[:, None, :], axis=-1) <= tol).sum(axis=0)
        iu = np.triu_indices(k.n_players, 1)
        if kf[iu].max() * 2 >= win.shape[0]:
            q = int(np.argmax(kf[iu]))
            partial.append((pos.get(c.clip_id, -1), c.clip_id, int(kf[iu][q]), float(d[iu][q])))
            s["clips_pair_within_tol_half_window_raw"] += 1
        s["kept_pairs_within_tol_raw"] += n_raw
        s["kept_players_frozen_raw"] += int(fr.sum())
        s["kept_players_with_zero_raw"] += int(zz.sum())
        s["clips_above_ceiling_raw"] += int(ceiling is not None and top > ceiling)
        s["max_step_speed_raw_x1000"] = max(s["max_step_speed_raw_x1000"], int(top * 1000))
        if n_raw or fr.any() or zz.any() or lin.any() or (ceiling is not None and top > ceiling):
            bad.append({"clip_id": c.clip_id, "check": "raw", "pairs": n_raw,
                        "frozen": int(fr.sum()), "zero": int(zz.sum()), "linear": int(lin.sum()),
                        "top_ms": top})
    return {"by_source": {k: dict(v) for k, v in sorted(out.items())},
            "closest_kept_pairs_post_m": sorted(closest_post)[:10], "flagged": bad,
            # (interleave position, clip, frames within tol, max distance m), not a check
            "partial_coloc_half_window": sorted(partial)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "items"])
    ap.add_argument("path")
    ap.add_argument("--pool", help="items mode: replay the kept players from these raw clips")
    ap.add_argument("--n-frames", type=int, default=20, help="pool mode: window length")
    ap.add_argument("--tol", type=float, default=DUPLICATE_TOL_M)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.mode == "pool":
        res = audit_pool(pathlib.Path(a.path), a.n_frames, a.tol)
        for key, v in res.items():
            print(f"== {key}: clips={v['clips']} pairs={v['pairs']} "
                  f"duplicate_tracks={v.get('duplicate_tracks', 0)} "
                  f"clips_with_duplicate={v.get('clips_with_duplicate', 0)} "
                  f"pairs_exactly_equal={v.get('pairs_exactly_equal', 0)}")
            print("   pair max-distance bins (m):", v["pair_max_hist_m"])
            print("   per-clip closest pair (m):",
                  {k: round(x, 3) for k, x in v["clip_min_quantiles_m"].items()})
            print(f"   same, clips whose closest pair is > {a.tol:g} m:",
                  {k: round(x, 3) for k, x in v["genuine_clip_min_quantiles_m"].items()})
            print("   frames within tol per pair (pairs within tol in >= 1 frame):",
                  v["coloc_frames_hist"])
            print("   closest pairs above tol:",
                  [(round(x, 3), cid, i, j) for x, cid, i, j in v["closest_above_tol"][:8]])
    else:
        res = audit_items(pathlib.Path(a.path), pathlib.Path(a.pool) if a.pool else None, a.tol)
        for src, v in res["by_source"].items():
            print(f"{src:15s} " + " ".join(f"{k}={n}" for k, n in v.items()))
        print("closest kept pairs (post, m):",
              [(round(x, 3), cid) for x, cid in res["closest_kept_pairs_post_m"][:5]])
        print(f"flagged clips: {len(res['flagged'])}", res["flagged"][:20])
        print("kept pairs within tol in >= half the window (position, clip, frames, max m):",
              [(p, c, k, round(m, 3)) for p, c, k, m in res["partial_coloc_half_window"][:20]])
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
