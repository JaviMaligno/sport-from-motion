"""Experimental conditions: which slice of a clip a model is allowed to see.

The design separates three sources of signal that "watching a clip" mixes up:

  formation         one frame. Only the *shape* of the group. If this already
                    identifies the sport, motion is not needed (scrums, line-outs,
                    a line of scrimmage are formations, not movements).
  motion            K frames in order. Formation + how it changes.
  motion_shuffled   the same K frames in a random order. If accuracy does not
                    drop vs. `motion`, the model is reading K formations, not
                    motion (the control that decided Part 1 of wheres-the-ball).
  kinematics        each player's path re-anchored on a neutral grid slot, so the
                    formation is destroyed. Individual movement survives (speeds,
                    turns, stop-start rhythm) AND so does coordination: if the whole
                    team runs the same way, the re-anchored paths still point the
                    same way.
  kinematics_solo   as `kinematics`, but each path is also rotated independently,
                    so coordination is destroyed too and only each individual's
                    movement rhythm is left. kinematics - kinematics_solo measures
                    the value of collective motion.

Formation is also studied as a sub-case: `formation` evaluated only on clips
tagged as set pieces / static phases (e.g. NFL pre-snap) vs. open play.

Representations: "sheet" (all frames tiled in one image — works with every API),
"frames" (one image per frame), "trails" (one image, fading trails), "text"
(coordinates as text), "gif" (for humans and video-capable models).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from motion_sport.schema import Clip

CONDITIONS = ("formation", "motion", "motion_shuffled", "kinematics", "kinematics_solo")
REPRESENTATIONS = ("sheet", "frames", "trails", "text", "gif")


@dataclass
class View:
    """What the model gets: a sequence of [N, 2] point sets, in display order."""

    condition: str
    frames: list[np.ndarray]
    frame_times: list[float]  # seconds, in *display* order (shuffled for motion_shuffled)
    order: list[int] = field(default_factory=list)  # original index of each shown frame
    ordered: bool = True  # whether display order == time order (trails/text rely on it)


def sample_indices(n_frames: int, k: int) -> list[int]:
    return np.linspace(0, n_frames - 1, min(k, n_frames)).round().astype(int).tolist()


def kinematics_layout(xy: np.ndarray, rng: np.random.Generator, spacing: float = 1.0,
                      solo: bool = False) -> np.ndarray:
    """Re-anchor each trajectory on a shuffled grid slot centred on the origin.

    solo=True also rotates each displacement path by its own random angle."""
    t, n, _ = xy.shape
    disp = xy - xy[0:1]
    if solo:
        a = rng.uniform(0, 2 * np.pi, n)
        rot = np.stack([np.stack([np.cos(a), -np.sin(a)], -1),
                        np.stack([np.sin(a), np.cos(a)], -1)], -2)  # [N, 2, 2]
        disp = np.einsum("nij,tnj->tni", rot, disp)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    slots = np.stack([gx.ravel(), gy.ravel()], 1)[:n].astype(np.float32)
    slots = (slots - slots.mean(0)) * spacing
    slots = slots[rng.permutation(n)]
    out = disp + slots[None]
    return out - np.nanmean(out, axis=(0, 1))  # centre the whole layout, not just t=0


def build_view(clip: Clip, condition: str, rng: np.random.Generator, k: int = 8,
               kin_spacing: float = 1.0) -> View:
    idx = sample_indices(clip.n_frames, k)
    times = [i / clip.fps for i in idx]
    if condition == "formation":
        mid = clip.n_frames // 2
        return View(condition, [clip.xy[mid]], [mid / clip.fps], [mid])
    if condition == "motion":
        return View(condition, [clip.xy[i] for i in idx], times, idx)
    if condition == "motion_shuffled":
        perm = rng.permutation(len(idx))
        while len(idx) > 1 and (perm == np.arange(len(idx))).all():
            perm = rng.permutation(len(idx))
        return View(condition, [clip.xy[idx[p]] for p in perm], [times[p] for p in perm],
                    [idx[p] for p in perm], ordered=False)
    if condition in ("kinematics", "kinematics_solo"):
        kin = kinematics_layout(clip.xy, rng, kin_spacing, solo=condition == "kinematics_solo")
        return View(condition, [kin[i] for i in idx], times, idx)
    raise ValueError(f"unknown condition {condition!r}; known: {CONDITIONS}")
