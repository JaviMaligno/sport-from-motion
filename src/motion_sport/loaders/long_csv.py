"""Generic loader for "long" tracking tables: one row per (frame, player).

This is the adapter of last resort and, in practice, the main one: TeamTrack's
pitch-coordinate exports, the output of our own video-extraction pipeline
(rugby), SoccerNet-GSR minimap exports and NFL Big Data Bowl files can all be
reduced to the same five columns. Column names are passed in, so a new source is
a mapping, not new code.
"""
from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from collections.abc import Iterator

import numpy as np

from motion_sport.loaders.common import window_stream
from motion_sport.schema import Clip

DEFAULT_COLUMNS = {"match": "match_id", "frame": "frame", "track": "track_id", "x": "x", "y": "y"}


def read_long_csv(
    path: str | pathlib.Path,
    columns: dict[str, str] | None = None,
    *,
    unit_scale: float = 1.0,
    exclude_tracks: set[str] | None = None,
    group_extra: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """Read a long table into {group_key: (xy[F, N, 2], track_ids, frame_numbers)}.

    `group_extra` adds columns to the grouping key (e.g. NFL `playId`), so each
    group is one continuous stream. `unit_scale` converts to metres
    (0.9144 for yards, 0.3048 for feet).
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    exclude = exclude_tracks or set()
    rows: dict[str, dict[int, dict[str, tuple[float, float]]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            tid = r[cols["track"]]
            if tid in ("", "NA", "nan") or tid in exclude:
                continue
            try:
                x, y = float(r[cols["x"]]), float(r[cols["y"]])
            except ValueError:
                continue
            key = r.get(cols["match"], "0") or "0"
            if group_extra:
                key = "-".join([key] + [r[c] for c in group_extra])
            rows[key][int(float(r[cols["frame"]]))][tid] = (x * unit_scale, y * unit_scale)

    out = {}
    for key, frames in rows.items():
        frame_ids = np.array(sorted(frames))
        tracks = sorted({t for f in frames.values() for t in f})
        index = {t: i for i, t in enumerate(tracks)}
        # Dense frame axis so gaps become NaN rather than silently shortened time.
        full = np.arange(frame_ids[0], frame_ids[-1] + 1)
        pos = {f: i for i, f in enumerate(full)}
        xy = np.full((len(full), len(tracks), 2), np.nan, dtype=np.float32)
        for f, players in frames.items():
            for t, p in players.items():
                xy[pos[f], index[t]] = p
        out[key] = (xy, tracks, full)
    return out


def load_long_csv(
    path: str | pathlib.Path,
    *,
    sport: str,
    source: str,
    fps: float,
    clip_seconds: float = 4.0,
    stride_seconds: float = 4.0,
    target_fps: float | None = None,
    columns: dict[str, str] | None = None,
    unit_scale: float = 1.0,
    exclude_tracks: set[str] | None = None,
    group_extra: list[str] | None = None,
    min_valid_players: int = 8,
    tags: list[str] | None = None,
) -> Iterator[Clip]:
    streams = read_long_csv(path, columns, unit_scale=unit_scale,
                            exclude_tracks=exclude_tracks, group_extra=group_extra)
    for key, (xy, _tracks, _frames) in streams.items():
        # With group_extra (e.g. one NFL play per group) the match is the first part.
        match_id = key.split("-")[0] if group_extra else key
        yield from window_stream(
            xy, sport=sport, source=source, match_id=match_id, fps=fps,
            clip_seconds=clip_seconds, stride_seconds=stride_seconds, target_fps=target_fps,
            min_valid_players=min_valid_players, id_prefix=f"{source}-{key}", tags=tags,
        )
