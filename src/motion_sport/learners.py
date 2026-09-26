"""Classifiers fitted on our clips, all behind one grouped cross-validation loop.

The ladder they complete (see docs/design.md, "Modelos"):

  logistic    hand-crafted features -> standardised logistic regression
  minirocket  permutation- and rotation-invariant time series per clip -> MiniRocket
              (thousands of *random, untrained* convolution kernels) -> logistic.
              The "fixed representation + linear fit" rung, native to time series.
  deepsets    raw per-player trajectories -> small permutation-invariant network,
              the architecture of wheres-the-ball's Level-2 specialist (per-player
              MLP -> mean+max pool -> head), with a classification head.

Rule for all of them: anything that looks at data is fitted inside the fold.
MiniRocket's kernel biases are quantiles of the training data, and the scaler of
the logistic probe has means and variances — fitting either on all clips would
leak the test matches into training.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from motion_sport.conditions import build_view
from motion_sport.schema import Clip

# fit_predict(train_idx, test_idx) -> (classes, probs[len(test_idx), len(classes)])
FitPredict = Callable[[np.ndarray, np.ndarray], tuple[list[str], np.ndarray]]


def grouped_oof(y: list[str], groups: list[str], fit_predict: FitPredict,
                n_splits: int = 5) -> tuple[list[str], list[dict[str, float]]]:
    """Out-of-fold predictions with GroupKFold by match.

    Grouping by match is not optional: clips from one match share players and
    conditions, and a random split would let a model recognise the match instead
    of the sport.
    """
    from sklearn.model_selection import GroupKFold

    y_arr = np.asarray(y)
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError(f"need at least 2 sports to classify, got {classes}")
    splits = min(n_splits, len(set(groups)))
    if splits < 2:
        raise ValueError("need at least 2 matches for grouped cross-validation")
    pred = [""] * len(y)
    probs: list[dict[str, float]] = [{} for _ in y]
    for train, test in GroupKFold(n_splits=splits).split(np.zeros(len(y)), y_arr, groups):
        if len(set(y_arr[train])) < 2:
            continue
        fold_classes, p = fit_predict(train, test)
        for i, row in zip(test, p):
            d = {c: 0.0 for c in classes}
            d.update({c: float(v) for c, v in zip(fold_classes, row)})
            probs[i] = d
            pred[i] = max(d, key=d.get)
    return pred, probs


def _logistic(X_train, y_train, X_test, seed: int = 0, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C, random_state=seed))
    clf.fit(X_train, y_train)
    return list(clf.classes_), clf.predict_proba(X_test)


def logistic_fit_predict(X: np.ndarray, y: list[str], seed: int = 0, C: float = 1.0) -> FitPredict:
    y_arr = np.asarray(y)
    return lambda train, test: _logistic(X[train], y_arr[train], X[test], seed, C)


# ------------------------------------------------------------------ MiniRocket

SERIES_CHANNELS = ("speed_q10", "speed_q50", "speed_q90", "accel_q50", "turn_mean",
                   "polarisation", "nn_velocity_alignment", "spread", "nn_distance")


def invariant_series(xy: np.ndarray, fps: float) -> np.ndarray:
    """[T, N, 2] -> [C, T-2] time series that do not depend on player order or on
    rotation/reflection: each channel summarises all players at one instant."""
    v = np.diff(xy, axis=0) * fps  # [T-1, N, 2]
    sp = np.linalg.norm(v, axis=2)
    acc = np.linalg.norm(np.diff(v, axis=0), axis=2) * fps  # [T-2, N]
    a, b = v[:-1], v[1:]
    cos = np.sum(a * b, axis=2) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-9)
    turn = np.arccos(np.clip(cos, -1, 1))  # [T-2, N]
    unit = v / np.maximum(sp[..., None], 1e-9)
    polar = np.linalg.norm(unit.mean(axis=1), axis=1)  # [T-1]
    pos = xy[1:]  # align positions with velocities
    d = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=3)
    idx = np.arange(d.shape[1])
    d[:, idx, idx] = np.inf
    nn = d.argmin(axis=2)  # [T-1, N]
    nn_align = np.sum(unit * np.take_along_axis(unit, nn[..., None], axis=1), axis=2).mean(axis=1)
    c = pos - pos.mean(axis=1, keepdims=True)
    spread = np.sqrt(np.mean(np.sum(c ** 2, axis=2), axis=1))
    nn_dist = d.min(axis=2).mean(axis=1)
    per_step = [np.quantile(sp, q, axis=1)[1:] for q in (0.1, 0.5, 0.9)]
    series = per_step + [np.median(acc, axis=1), turn.mean(axis=1),
                         polar[1:], nn_align[1:], spread[1:], nn_dist[1:]]
    return np.nan_to_num(np.stack(series).astype(np.float32))


def clip_series(clip: Clip, condition: str, seed: int) -> np.ndarray:
    """Series from the same view a VLM gets, but at full temporal resolution."""
    rng = np.random.default_rng(seed)
    view = build_view(clip, condition, rng, k=clip.n_frames)
    if condition == "formation":
        raise ValueError("MiniRocket needs a sequence; formation is a single frame")
    return invariant_series(np.stack(view.frames), clip.fps)


def minirocket_fit_predict(X: np.ndarray, y: list[str], seed: int = 0,
                           n_kernels: int = 10_000) -> FitPredict:
    from aeon.transformations.collection.convolution_based import MiniRocket

    if X.shape[2] < 9:
        raise ValueError(f"MiniRocket needs >= 9 time points, got {X.shape[2]} (use longer clips)")
    y_arr = np.asarray(y)

    def fit_predict(train, test):
        mr = MiniRocket(n_kernels=n_kernels, random_state=seed)
        f_train = mr.fit_transform(X[train])  # kernel biases from the training fold only
        # ~10k features for a few hundred clips: stronger regularisation than the default
        return _logistic(f_train, y_arr[train], mr.transform(X[test]), seed, C=0.1)

    return fit_predict


# -------------------------------------------------------------------- DeepSets


def player_tokens(clip: Clip, condition: str, seed: int, k: int = 8) -> np.ndarray:
    """[N, 4k] per-player features from the same view a VLM gets: (x, y) at each
    shown snapshot plus the step to the next one, in display order. Shuffled views
    therefore give shuffled steps, which is the point of that control."""
    rng = np.random.default_rng(seed)
    view = build_view(clip, condition, rng, k=k)
    f = np.stack(view.frames)  # [K, N, 2]
    steps = np.concatenate([np.diff(f, axis=0), np.zeros_like(f[:1])], axis=0)
    return np.concatenate([f, steps], axis=2).transpose(1, 0, 2).reshape(f.shape[1], -1)


def _rotate_tokens(tok, rng):
    """Random rotation + reflection of every (x, y) pair: augmentation, since the
    controls rotate each clip once and the net should not learn an orientation."""
    import torch

    a = float(rng.uniform(0, 2 * np.pi))
    rot = torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=tok.dtype)
    if rng.random() < 0.5:
        rot = rot @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=tok.dtype)
    shp = tok.shape
    return (tok.reshape(*shp[:-1], -1, 2) @ rot.T).reshape(shp)


def deepsets_fit_predict(tokens: list[np.ndarray], y: list[str], seed: int = 0,
                         epochs: int = 150, d_h: int = 64, lr: float = 1e-3,
                         weight_decay: float = 1e-4) -> FitPredict:
    import torch
    import torch.nn as nn

    class DeepSets(nn.Module):
        """Same shape as wheres-the-ball's Level-2 specialist, classification head."""

        def __init__(self, d_in, n_out):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, n_out))

        def forward(self, p):  # [B, N, D]; N is fixed by the controls, so no mask
            h = self.phi(p)
            return self.rho(torch.cat([h.mean(1), h.max(1).values], -1))

    X = torch.tensor(np.stack(tokens), dtype=torch.float32)  # [n, N, D]
    y_arr = np.asarray(y)

    def fit_predict(train, test):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        classes = sorted(set(y_arr[train]))
        yt = torch.tensor([classes.index(c) for c in y_arr[train]])
        model = DeepSets(X.shape[2], len(classes))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # fixed schedule, no early stopping: there is no validation data to peek at
        for _ in range(epochs):
            order = rng.permutation(len(train))
            for i in range(0, len(order), 64):
                b = order[i:i + 64]
                xb = _rotate_tokens(X[train[b]], rng)
                loss = nn.functional.cross_entropy(model(xb), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return classes, torch.softmax(model(X[test]), -1).numpy()

    return fit_predict
