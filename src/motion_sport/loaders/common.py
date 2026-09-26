"""Shared helpers for turning a continuous tracking stream into fixed-length clips."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from motion_sport.schema import FIELD_SIZE_M, Clip


def window_stream(
    xy: np.ndarray,
    *,
    sport: str,
    source: str,
    match_id: str,
    fps: float,
    clip_seconds: float,
    stride_seconds: float,
    target_fps: float | None = None,
    team: np.ndarray | None = None,
    field_size: tuple[float, float] | None = None,
    min_valid_players: int = 8,
    id_prefix: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[Clip]:
    """Cut a [F, N, 2] stream (metres) into clips of `clip_seconds`.

    Frames are optionally decimated to `target_fps` (all sources should end up at
    the same rate, otherwise fps itself becomes a sport cue). A window is kept only
    if at least `min_valid_players` players are observed in every frame of it.
    """
    if target_fps and target_fps < fps:
        step = fps / target_fps
        if abs(step - round(step)) > 1e-6:
            raise ValueError(f"cannot decimate {fps} Hz to {target_fps} Hz by an integer step")
        xy = xy[:: int(round(step))]
        fps = target_fps
    length = int(round(clip_seconds * fps))
    stride = max(1, int(round(stride_seconds * fps)))
    field_size = field_size or FIELD_SIZE_M.get(sport)
    prefix = id_prefix or f"{source}-{match_id}"
    for k, start in enumerate(range(0, xy.shape[0] - length + 1, stride)):
        win = xy[start : start + length]
        observed = ~np.isnan(win).any(axis=2)  # [T, N]
        if observed.sum(axis=1).min() < min_valid_players:
            continue
        keep = observed.any(axis=0)  # drop players never seen in this window
        yield Clip(
            clip_id=f"{prefix}-{k:05d}",
            sport=sport,
            source=source,
            match_id=match_id,
            fps=fps,
            xy=win[:, keep],
            team=team[keep] if team is not None else None,
            field_size=field_size,
            tags=list(tags or []),
            meta={"start_frame": int(start)},
        )
