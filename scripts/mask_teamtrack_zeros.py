"""Already-ingested TeamTrack clips: exact (0, 0) points -> NaN, as copies (D18).

TeamTrack writes a missing detection as exactly (0.0, 0.0); clips ingested before
scripts/teamtrack_to_long_csv.py masked it carry those points as real positions (a player
parked in the corner of the court). This rewrites every `teamtrack-*` clip of a pool into
a new directory with those points set to NaN, which `fill_short_gaps` / `fix_player_count`
then treat as unobserved, and builds a new pool of symlinks identical to the old one except
that the TeamTrack entries point to the fixed copies. Clip ids, match ids, tags and every
other array are unchanged (so every seed derived from a clip id is too); `meta` gets
`missing_zero_masked` (number of points masked). Nothing under the old pool or its link
targets is written: re-ingesting would change the clip ids (the match id is in them).

    python scripts/mask_teamtrack_zeros.py data/pool_final3 data/clips_tt_fix0 data/pool_final4
    python scripts/mask_teamtrack_zeros.py data/pool_d8_final3 data/clips_d8_tt_fix0 \
        data/pool_d8_final4
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import pathlib

import numpy as np

PATTERN = "teamtrack-*.npz"


def mask_clip(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Write `src` (followed through symlinks) to `dst` with exact (0, 0) -> NaN.
    -> number of points masked."""
    with np.load(src.resolve(), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    xy = arrays["xy"].copy()
    miss = (xy[..., 0] == 0) & (xy[..., 1] == 0)
    xy[miss] = np.nan
    arrays["xy"] = xy
    meta = json.loads(str(arrays["meta_json"]))
    meta.setdefault("meta", {})["missing_zero_masked"] = int(miss.sum())
    arrays["meta_json"] = np.array(json.dumps(meta))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    tmp = dst.with_name(f".{dst.name}.tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, dst)
    return int(miss.sum())


def rebuild(old_pool: str | pathlib.Path, fixed: str | pathlib.Path,
            new_pool: str | pathlib.Path, pattern: str = PATTERN) -> dict:
    old_pool, fixed, new_pool = (pathlib.Path(p).resolve() for p in (old_pool, fixed, new_pool))
    if new_pool.exists() and any(new_pool.iterdir()):
        raise SystemExit(f"{new_pool} exists and is not empty")
    fixed.mkdir(parents=True, exist_ok=True)
    new_pool.mkdir(parents=True, exist_ok=True)
    counts = {"clips": 0, "teamtrack": 0, "teamtrack_with_zero": 0, "points_masked": 0,
              "by_sport": {}}
    for p in sorted(old_pool.glob("*.npz")):
        counts["clips"] += 1
        if fnmatch.fnmatch(p.name, pattern):
            n = mask_clip(p, fixed / p.name)
            with np.load(fixed / p.name, allow_pickle=False) as z:
                sport = json.loads(str(z["meta_json"]))["sport"]
            s = counts["by_sport"].setdefault(sport, {"clips": 0, "with_zero": 0, "points": 0})
            s["clips"] += 1
            s["with_zero"] += n > 0
            s["points"] += n
            counts["teamtrack"] += 1
            counts["teamtrack_with_zero"] += n > 0
            counts["points_masked"] += n
            target = fixed / p.name
        else:
            target = p.resolve()  # same clip as the old pool (a link or a real file there)
        os.symlink(target, new_pool / p.name)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_pool")
    ap.add_argument("fixed_dir", help="where the fixed TeamTrack copies are written")
    ap.add_argument("new_pool", help="new pool of symlinks (must not exist or be empty)")
    a = ap.parse_args()
    print(json.dumps(rebuild(a.old_pool, a.fixed_dir, a.new_pool), indent=2))


if __name__ == "__main__":
    main()
