"""TeamTrack (Kaggle atomscott/teamtrack) trajectories -> long CSV per sport.

`teamtrack-trajectory/<Sport>/<split>/<segment>_<k>.txt` are sliding windows of 240
rows x (2 * n_players) columns, pitch coordinates in metres, consecutive windows
shifted by 12 rows with a stable player order. Each 30 s segment is rebuilt by
downloading only the windows needed to cover it (0, 20, 40, ... and the last one),
not all ~56. Match ids: soccer `tt-soccer`; handball `tt-handball`; basketball P3.
Each sport is one recording, so it is one match for the grouped CV and the cluster
bootstrap; the part stays in the segment (e.g. train-1st_fisheye_0-30,
train-F_20200220_1_0000_0030).
- Handball: both halves are the same game.
- Soccer: F_20200220_1 (0-900 s) and F_20220220_1 (900-1980 s) are one continuous
  recording; the second date is a typo in the dataset. At the 900 s seam the same 22
  track indices are a median 0.6 m from where the first file left them, closer than
  between two segments of the same file (~4 m). Clips ingested before used the date as
  match id and counted the game as two.
Fix already-ingested clips without re-downloading with scripts/relabel_match.py.

Missing detections: TeamTrack writes a player it did not detect as exactly (0.0, 0.0)
(both coordinates; a single exact 0 never occurs). Left as is, a missing player becomes a
real position at the corner of the court: a whole-window (0, 0) track is "fully observed",
passes the speed ceiling and is drawn as a motionless dot (D18). `mask_missing` writes
them as NaN, which the loaders read as unobserved. Clips ingested before D18 are fixed,
as copies, with scripts/mask_teamtrack_zeros.py.

    python scripts/teamtrack_to_long_csv.py <file_list.txt> data/raw/teamtrack
    -> data/raw/teamtrack/{soccer,handball,basketball}.csv
    # rebuild from the downloaded windows only (no Kaggle call): file list = the cache
    python scripts/teamtrack_to_long_csv.py --from-cache data/raw/teamtrack/windows \
        data/raw/teamtrack_fix0
Needs the kaggle package and ~/.kaggle/access_token (only when a window must be
downloaded).
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

WIN, STEP = 240, 12
SPORT = {"Soccer": "soccer", "Handball": "handball", "Basketball": "basketball"}


HANDBALL_MATCH = "tt-handball"  # the one TeamTrack handball game, both halves
# The one TeamTrack soccer recording: F_20200220_1 and F_20220220_1 are its first 900 s and
# the rest (date typo), not two games.
SOCCER_MATCH = "tt-soccer"


def match_id(sport: str, seg: str) -> str:
    """Match id of a TeamTrack segment (`seg` = file stem without the window index)."""
    if sport == "Soccer":
        return SOCCER_MATCH  # the date in F_<date>_1 is not a match (see module docstring)
    if sport == "Handball":
        return HANDBALL_MATCH  # the half ("1st"/"2nd", seg.split("_")[0]) is not a match
    return seg.split("_")[0]


def mask_missing(xy: np.ndarray) -> np.ndarray:
    """[frames, 2 * players] (x0, y0, x1, y1, ...) -> copy with every exact (0.0, 0.0)
    point set to NaN: TeamTrack's code for a missing detection, not a position."""
    out = np.array(xy, dtype=float, copy=True)
    miss = (out[:, 0::2] == 0) & (out[:, 1::2] == 0)
    out[:, 0::2][miss] = np.nan
    out[:, 1::2][miss] = np.nan
    return out


def cache_file_list(cache: pathlib.Path) -> list[str]:
    """File names of the windows already in `cache` (<cache>/<Sport>/<split>/<file>.txt)."""
    return sorted(f"teamtrack-trajectory/{p.relative_to(cache).as_posix()}"
                  for p in cache.glob("*/*/*.txt"))


def main() -> None:
    if sys.argv[1] == "--from-cache":
        src_cache, out = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
        names = cache_file_list(src_cache)
        cache = src_cache
    else:
        names = [n.strip() for n in pathlib.Path(sys.argv[1]).read_text().split() if n.strip()]
        out = pathlib.Path(sys.argv[2])
        cache = out / "windows"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    segs: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    for n in names:
        parts = n.split("/")
        sport, split, fname = parts[-3], parts[-2], parts[-1]
        m = re.match(r"(.+)_(\d+)\.txt$", fname)
        segs[(sport, split, m.group(1))][int(m.group(2))] = n
    api = None
    writers, fhs = {}, []
    for (sport, split, seg), wins in sorted(segs.items()):
        last = max(wins)
        need = sorted(set(range(0, last + 1, WIN // STEP)) | {last})
        arrays = {}
        for k in need:
            local = cache / sport / split / pathlib.Path(wins[k]).name
            if not local.exists():
                if api is None:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                local.parent.mkdir(parents=True, exist_ok=True)
                api.dataset_download_file("atomscott/teamtrack", wins[k], path=str(local.parent),
                                          quiet=True)
            arrays[k] = np.loadtxt(local, delimiter=",", ndmin=2)
        n_rows = last * STEP + WIN
        xy = np.full((n_rows, arrays[0].shape[1]), np.nan)
        for k, a in arrays.items():
            xy[k * STEP:k * STEP + len(a)] = a
        if np.isnan(xy).any():
            print(f"skip {sport}/{seg}: gaps after stitching", file=sys.stderr)
            continue
        n_missing = int(((xy[:, 0::2] == 0) & (xy[:, 1::2] == 0)).sum())
        xy = mask_missing(xy)  # after the stitching check: these NaN are missing players
        match = match_id(sport, seg)
        sp = SPORT[sport]
        if sp not in writers:
            fh = (out / f"{sp}.csv").open("w", newline="")
            fhs.append(fh)
            writers[sp] = csv.writer(fh)
            writers[sp].writerow(["match_id", "segment", "frame", "track_id", "x", "y"])
        for f in range(n_rows):
            for j in range(xy.shape[1] // 2):
                writers[sp].writerow([match, f"{split}-{seg}", f, j, xy[f, 2 * j], xy[f, 2 * j + 1]])
        print(f"{sp} {split}/{seg}: {n_rows} frames, {xy.shape[1] // 2} players, {len(need)} files, "
              f"{n_missing} missing (0, 0) points -> NaN")
    for fh in fhs:
        fh.close()


if __name__ == "__main__":
    main()
