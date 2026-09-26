"""Two feature-based classifiers that bracket what the models do.

nuisance   Only the cues the experiment is NOT about: how many players, how big
           the field/spread is, frame rate, duration. On `raw` clips it shows how
           easy the task is by cheating. After any preset other than `raw` it must
           sit at chance — that is the test that the controls work.
tempo      Median player speed in the clip's own units. On `strict` clips
           speed is relative to spread and is a mix of genuine motion and scale;
           `strict_tempo` equalises it, so there it must sit at chance too.
kinematic  Hand-crafted motion statistics (speed/acceleration/turning
           distributions, velocity alignment, spacing, stop-start). A cheap,
           transparent specialist: if it succeeds, the signal is in the motion;
           the question for the VLMs is then whether they can *see* it.

Both are evaluated with out-of-fold predictions grouped by match (probe.fit_probe_oof).
"""
from __future__ import annotations

import numpy as np

from motion_sport.schema import Clip


def _speeds(xy: np.ndarray, fps: float) -> np.ndarray:
    return np.linalg.norm(np.diff(xy, axis=0), axis=2) * fps  # [T-1, N]


def nuisance_features(clip: Clip) -> np.ndarray:
    """Pure shortcuts. Shape of the formation is deliberately NOT here: it is signal."""
    c = clip.xy - np.nanmean(clip.xy, axis=(0, 1))
    spread = np.sqrt(np.nanmean(np.sum(c ** 2, axis=2)))
    return np.array([clip.n_players, clip.fps, clip.duration_s, spread], dtype=np.float64)


def tempo_features(clip: Clip) -> np.ndarray:
    """Overall pace only. The *shape* of the speed distribution (sprints vs. jogging,
    p90/median) is motion signal and lives in `kinematic`, not here."""
    return np.array([np.nanmedian(_speeds(clip.xy, clip.fps))], dtype=np.float64)


def _mean_nn(pts: np.ndarray) -> float:
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def kinematic_features(clip: Clip) -> np.ndarray:
    """Scale-free where possible: speeds are divided by the clip's median speed."""
    xy = clip.xy
    v = np.diff(xy, axis=0) * clip.fps  # [T-1, N, 2]
    sp = np.linalg.norm(v, axis=2)
    med = max(float(np.nanmedian(sp)), 1e-6)
    spn = sp / med
    acc = np.linalg.norm(np.diff(v, axis=0), axis=2) * clip.fps / med
    # turning: angle between consecutive velocity vectors, only when moving
    a, b = v[:-1], v[1:]
    cos = np.sum(a * b, axis=2) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-9)
    moving = (sp[:-1] > 0.2 * med) & (sp[1:] > 0.2 * med)
    turn = np.arccos(np.clip(cos[moving], -1, 1)) if moving.any() else np.array([0.0])
    # polarisation: |mean unit velocity| — 1 when everyone runs the same way
    unit = v / np.maximum(sp[..., None], 1e-9)
    polar = np.linalg.norm(np.nanmean(unit, axis=1), axis=1)
    # velocity correlation with nearest neighbour (collective vs. individual motion)
    nn_corr = []
    for k in range(0, v.shape[0], max(1, v.shape[0] // 5)):
        p = xy[k]
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        np.fill_diagonal(d, np.inf)
        j = d.argmin(axis=1)
        nn_corr.append(np.nanmean(np.sum(unit[k] * unit[k][j], axis=1)))
    # spacing relative to spread, and how it changes (compression / expansion)
    c = xy - np.nanmean(xy, axis=1, keepdims=True)
    spread_t = np.sqrt(np.nanmean(np.sum(c ** 2, axis=2), axis=1))
    nn_t = np.array([_mean_nn(xy[k]) for k in range(0, clip.n_frames, max(1, clip.n_frames // 5))])
    q = lambda arr, p: float(np.nanpercentile(arr, p))  # noqa: E731
    return np.array([
        q(spn, 10), q(spn, 50), q(spn, 90), float(np.nanstd(spn)),
        float(np.mean(spn < 0.2)),  # fraction of near-stationary player-frames
        q(acc, 50), q(acc, 90),
        float(np.mean(turn)), float(np.percentile(turn, 90)),
        float(np.nanmean(polar)), float(np.nanstd(polar)),
        float(np.nanmean(nn_corr)),
        float(np.nanmean(nn_t) / max(np.nanmean(spread_t), 1e-6)),
        float((spread_t[-1] - spread_t[0]) / max(spread_t.mean(), 1e-6)),
    ], dtype=np.float64)


FEATURE_SETS = {"nuisance": nuisance_features, "tempo": tempo_features,
                "kinematic": kinematic_features}


def featurize(clips: list[Clip], kind: str) -> np.ndarray:
    X = np.stack([FEATURE_SETS[kind](c) for c in clips])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
