"""Toy trajectories for smoke-testing the pipeline end to end. NOT DATA.

Nothing produced here may appear in a result, a figure or an article. It exists
so the harness (controls -> rendering -> prompts -> backends -> metrics) can be
exercised without downloading anything. The two "sports" differ in exactly the
ways the leak controls are supposed to neutralise (player count, field size,
speed) plus one genuine motion difference (a collective defensive line that
advances together vs. loosely coupled players), which makes it a good test of
whether the controls remove the shortcuts while leaving the motion signal.
"""
from __future__ import annotations

import numpy as np

from motion_sport.schema import FIELD_SIZE_M, Clip


def _loose(rng, n, t, fps, size, speed):
    """Loosely coupled players: each follows its own smooth random walk."""
    pos = rng.uniform([0.2, 0.2], [0.8, 0.8], (n, 2)) * size
    vel = rng.normal(0, speed, (n, 2))
    out = np.empty((t, n, 2), np.float32)
    for k in range(t):
        vel = 0.9 * vel + rng.normal(0, speed * 0.3, (n, 2))
        pos = np.clip(pos + vel / fps, 0, size)
        out[k] = pos
    return out


def _lines(rng, n, t, fps, size, speed):
    """Two facing lines that advance/retreat together, plus a contact cluster."""
    half = n // 2
    ys = np.linspace(0.1, 0.9, half) * size[1]
    x0 = rng.uniform(0.4, 0.6) * size[0]
    drift = rng.choice([-1, 1]) * speed
    out = np.empty((t, n, 2), np.float32)
    for k in range(t):
        front = x0 + drift * k / fps
        a = np.stack([np.full(half, front - 2.0), ys], 1)
        b = np.stack([np.full(n - half, front + 2.0), np.linspace(0.1, 0.9, n - half) * size[1]], 1)
        pts = np.concatenate([a, b]) + rng.normal(0, 0.4, (n, 2))
        pts[:4] = [front, size[1] / 2] + rng.normal(0, 0.8, (4, 2))  # ruck-like cluster
        out[k] = pts
    return out


def make_toy_clips(n_per_sport: int = 20, *, seed: int = 0, fps: float = 5.0,
                   seconds: float = 4.0, matches_per_sport: int = 4) -> list[Clip]:
    rng = np.random.default_rng(seed)
    t = int(seconds * fps)
    specs = {
        "soccer": dict(n=22, gen=_loose, speed=3.0),
        "rugby_union": dict(n=30, gen=_lines, speed=2.0),
    }
    clips = []
    for sport, spec in specs.items():
        size = np.array(FIELD_SIZE_M[sport], np.float32)
        for i in range(n_per_sport):
            xy = spec["gen"](rng, spec["n"], t, fps, size, spec["speed"])
            clips.append(Clip(
                clip_id=f"toy-{sport}-{i:04d}", sport=sport, source="toy",
                match_id=f"toy-{sport}-m{i % matches_per_sport}", fps=fps, xy=xy,
                field_size=tuple(size.tolist()), meta={"synthetic": True},
            ))
    return clips
