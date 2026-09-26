"""Leak controls: remove every sport cue that is not *how the players move*.

Each control targets one named shortcut (see docs/design.md, "Fugas"):

| control              | shortcut it removes                                         |
|----------------------|-------------------------------------------------------------|
| fix_player_count     | counting dots (30 in rugby union vs 22 in soccer vs 10 ...) |
| normalize_space      | absolute scale / field size (basketball court vs pitch)     |
| random_rigid         | field orientation, attack direction, pitch aspect ratio     |
| fit_window(tempo=)   | raw speed, which in field units encodes field size          |
| (team dropped)       | team structure given for free instead of inferred           |

Controls are pure functions Clip -> Clip (or None when a clip cannot satisfy the
control, e.g. has fewer fully observed players than N). Whatever they do is
recorded in `clip.meta["controls"]` so a result can always be traced back.

The nuisance baseline (baselines.py) is the check that they work: after the
controls, a classifier on count/scale/speed features alone should sit at chance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from motion_sport.schema import Clip


def _log(clip: Clip, name: str, **params) -> Clip:
    clip.meta.setdefault("controls", []).append({"name": name, **params})
    return clip


def fill_short_gaps(clip: Clip, max_gap: int = 3) -> Clip:
    """Linearly interpolate gaps of <= max_gap frames; longer gaps stay NaN."""
    xy = clip.xy.copy()
    t = np.arange(clip.n_frames)
    for j in range(clip.n_players):
        for d in range(2):
            v = xy[:, j, d]
            bad = np.isnan(v)
            if not bad.any() or bad.all():
                continue
            # run-length of each gap
            filled = np.interp(t, t[~bad], v[~bad])
            run = 0
            for k in range(len(v) + 1):
                if k < len(v) and bad[k]:
                    run += 1
                    continue
                if 0 < run <= max_gap and k - run > 0 and k < len(v):  # interior gaps only
                    v[k - run:k] = filled[k - run:k]
                run = 0
    return _log(clip.replace(xy=xy), "fill_short_gaps", max_gap=max_gap)


def fix_player_count(clip: Clip, n: int, rng: np.random.Generator,
                     mode: str = "central") -> Clip | None:
    """Keep exactly `n` fully observed players, or reject the clip.

    mode="central": the n players closest to the clip's mean position — mimics a
    camera window and keeps the interacting core of the play.
    mode="random": a uniform random subset.
    """
    full = ~np.isnan(clip.xy).any(axis=(0, 2))
    cand = np.flatnonzero(full)
    if len(cand) < n:
        return None
    if mode == "central":
        centre = np.nanmean(clip.xy[:, cand], axis=(0, 1))
        dist = np.linalg.norm(clip.xy[:, cand].mean(axis=0) - centre, axis=1)
        keep = cand[np.argsort(dist)[:n]]
    elif mode == "random":
        keep = rng.choice(cand, size=n, replace=False)
    else:
        raise ValueError(mode)
    keep = rng.permutation(keep)  # track order must carry no information either
    out = clip.replace(xy=clip.xy[:, keep],
                       team=clip.team[keep] if clip.team is not None else None)
    return _log(out, "fix_player_count", n=n, mode=mode)


def normalize_space(clip: Clip, mode: str = "spread") -> Clip:
    """Centre on the clip's mean position and rescale.

    mode="spread": divide by the RMS distance of players to that centre, so every
      clip has unit spread — removes field size AND how spread out the sport is.
    mode="field":  divide by the field length — keeps relative spread (a sport cue
      that is arguably part of "how they move"), removes absolute size.
    mode="none":   metres, only centred.
    """
    xy = clip.xy - np.nanmean(clip.xy, axis=(0, 1))
    if mode == "spread":
        scale = float(np.sqrt(np.nanmean(np.sum(xy ** 2, axis=2))))
    elif mode == "field":
        if not clip.field_size:
            raise ValueError(f"{clip.clip_id}: field mode needs field_size")
        scale = float(clip.field_size[0])
    elif mode == "none":
        scale = 1.0
    else:
        raise ValueError(mode)
    out = clip.replace(xy=xy / max(scale, 1e-6))
    out.meta["space_scale"] = scale
    return _log(out, "normalize_space", mode=mode)


def random_rigid(clip: Clip, rng: np.random.Generator) -> Clip:
    """Random rotation (any angle) plus a random reflection, about the origin."""
    a = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    if rng.random() < 0.5:
        rot = rot @ np.array([[1, 0], [0, -1]], np.float32)
    return _log(clip.replace(xy=clip.xy @ rot.T), "random_rigid", angle=float(a))


def median_speed(clip: Clip) -> float:
    """Median per-player speed in clip units per second."""
    step = np.linalg.norm(np.diff(clip.xy, axis=0), axis=2) * clip.fps
    return float(np.nanmedian(step)) if step.size else 0.0


def resample_centered(clip: Clip, n: int, factor: float = 1.0) -> Clip | None:
    """`n` frames around the clip centre, source time advancing `factor` frames per
    output frame (factor > 1 plays faster). None if the clip is too short."""
    centre = (clip.n_frames - 1) / 2
    t = centre + (np.arange(n) - (n - 1) / 2) * factor
    if t[0] < -1e-6 or t[-1] > clip.n_frames - 1 + 1e-6:
        return None
    src = np.arange(clip.n_frames)
    xy = np.empty((n, clip.n_players, 2), np.float32)
    for j in range(clip.n_players):
        for d in range(2):
            xy[:, j, d] = np.interp(t, src, clip.xy[:, j, d])
    return clip.replace(xy=xy)


def fit_window(clip: Clip, n: int, *, space: str, tempo: float | None = None,
               iters: int = 4) -> Clip | None:
    """Choose the final `n`-frame window, before any spatial normalisation.

    Order matters: spread must be normalised on the window the model will see.
    Normalising the whole clip and cropping afterwards leaves the window's own
    spread free to vary by sport — a leak the nuisance baseline caught in the
    smoke test. With `tempo`, the window is also time-resampled so that, *after*
    normalising its spread, its median speed is `tempo`; since resampling changes
    the spread, the factor is refined a few times (it converges quickly).
    """
    if tempo is None:
        return resample_centered(clip, n) if clip.n_frames >= n else None
    factor = 1.0
    for _ in range(iters):
        win = resample_centered(clip, n, factor)
        if win is None:
            return None
        speed = median_speed(normalize_space(win, space))
        if speed <= 0:
            return None
        factor *= tempo / speed
    win = resample_centered(clip, n, factor)
    return _log(win, "normalize_tempo", factor=float(factor), target=tempo) if win else None


@dataclass(frozen=True)
class ControlConfig:
    """One named bundle of controls. Ablations are just different configs."""

    n_players: int | None = 12
    player_mode: str = "central"
    space: str = "spread"
    rotate: bool = True
    # target median speed in normalised units/s; "auto" = a low percentile of the
    # dataset, resolved by pipeline.prepare. Speeding a clip up consumes source frames, so
    # ingest clips ~2-3x longer than n_frames / fps when tempo is on.
    tempo: float | str | None = None
    n_frames: int | None = 20
    drop_team: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


# The configs the design refers to by name.
PRESETS: dict[str, ControlConfig] = {
    # nothing removed except the ball: the "how much leaks" reference point
    "raw": ControlConfig(n_players=None, space="none", rotate=False, tempo=None, drop_team=False),
    # the main condition: count, scale, orientation and team removed; speed kept
    "strict": ControlConfig(),
    # additionally equalise tempo: only the *shape* of motion is left
    "strict_tempo": ControlConfig(tempo="auto"),
    # keeps relative spread (field-normalised), for the multi-size comparison
    "field_scaled": ControlConfig(space="field"),
}


def apply_controls(clip: Clip, cfg: ControlConfig, rng: np.random.Generator) -> Clip | None:
    c: Clip | None = fill_short_gaps(clip)
    if cfg.n_players:
        c = fix_player_count(c, cfg.n_players, rng, cfg.player_mode)
        if c is None:
            return None
    else:  # still drop players that are not fully observed
        full = ~np.isnan(c.xy).any(axis=(0, 2))
        c = c.replace(xy=c.xy[:, full], team=c.team[full] if c.team is not None else None)
    if cfg.tempo is not None and not isinstance(cfg.tempo, (int, float)):
        raise ValueError("resolve tempo='auto' to a number first (pipeline.prepare does)")
    if cfg.n_frames:
        c = fit_window(c, cfg.n_frames, space=cfg.space,
                       tempo=float(cfg.tempo) if cfg.tempo else None)
        if c is None:
            return None
    elif cfg.tempo:
        raise ValueError("tempo normalisation needs n_frames")
    c = normalize_space(c, cfg.space)
    if cfg.rotate:
        c = random_rigid(c, rng)
    if cfg.drop_team:
        c = c.replace(team=None)
    c.meta["control_config"] = cfg.as_dict()
    return c
