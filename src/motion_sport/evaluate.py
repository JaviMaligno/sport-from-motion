"""Metrics with match-clustered bootstrap CIs.

Lesson carried over from wheres-the-ball Part 1 (a 64% on 14 items whose CI
spanned 36-86%): every number gets a CI from day one, and the resampling unit is
the *match*, not the clip — clips of one match are not independent.

Accuracy alone mixes "sees the signal" with "has a favourite answer" (pilot 2:
one model said "American football" on 64% of the clips). So every summary also
carries metrics that a constant answer cannot inflate: per-class recall, balanced
accuracy, macro-F1, Cohen's kappa, and a prior-corrected accuracy (contextual
calibration: each item's probabilities divided by the model's mean predicted
distribution, estimated on the *other* matches only).

All bootstrapped statistics are computed as weighted statistics: resampling
matches with replacement is the same as giving every clip of match m the weight
"number of times m was drawn", which lets one draw serve every metric at once.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

NONE = "__none__"  # unparsed / errored / out-of-set answers; always counts as wrong
_EPS = 1e-6


def _acc(rows) -> float:
    return float(np.mean([r["label"] == r["sport"] for r in rows])) if rows else float("nan")


def _balanced_acc(rows, classes) -> float:
    per = [np.mean([r["label"] == c for r in rows if r["sport"] == c])
           for c in classes if any(r["sport"] == c for r in rows)]
    return float(np.mean(per)) if per else float("nan")


def _log_loss(rows, eps=1e-3) -> float:
    vals = [-np.log(max(r["probs"].get(r["sport"], 0.0), eps)) for r in rows if r.get("probs")]
    return float(np.mean(vals)) if vals else float("nan")


def cluster_bootstrap(rows: list[dict], stat, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Generic percentile CI of `stat(rows)` resampling matches (slow; any statistic)."""
    by_match = defaultdict(list)
    for r in rows:
        by_match[r["match_id"]].append(r)
    keys = list(by_match)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for k in rng.choice(keys, size=len(keys), replace=True) for r in by_match[k]]
        vals.append(stat(sample))
    return _ci(np.array(vals, dtype=float))


def _ci(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_counts(match_ids: list, n_boot: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """-> (row -> match index [n], draws per match [n_boot, M]). Matches in first-seen order."""
    pos: dict = {}
    idx = np.array([pos.setdefault(m, len(pos)) for m in match_ids], dtype=int)
    n_m = len(pos)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, n_m))
    if n_m:
        draws = rng.integers(0, n_m, size=(n_boot, n_m))
        np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
    return idx, counts


def _wmean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of `values` [n] under each weight row [B, n] -> [B]."""
    tot = weights.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, weights @ values / np.where(tot > 0, tot, 1), np.nan)


def _prob_matrix(rows: list[dict], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Probability vector per row over `classes`; one-hot label when a model gave no
    probabilities; invalid (all zero) when it gave neither."""
    P = np.zeros((len(rows), len(classes)))
    for i, r in enumerate(rows):
        probs = r.get("probs") or {}
        if probs:
            P[i] = [float(probs.get(c, 0.0)) for c in classes]
        elif r.get("label") in classes:
            P[i, classes.index(r["label"])] = 1.0
        s = P[i].sum()
        if s > 0:
            P[i] /= s
    return P, P.sum(1) > 0


def _prior_corrected_pred(P, valid, idx, counts, chunk: int = 250) -> np.ndarray:
    """Argmax of p / prior per row and bootstrap draw -> [B, n] (-1 = no prediction).

    The prior of a row in match m is the mean predicted distribution over the rows of
    every *other* match in that draw (leave-one-match-out), so an item never helps
    correct itself. Without any other match the row is left uncorrected.
    """
    n, k = P.shape
    n_m = counts.shape[1]
    Pv = P * valid[:, None]
    S_m = np.zeros((n_m, k))
    np.add.at(S_m, idx, Pv)
    c_m = np.bincount(idx, weights=valid.astype(float), minlength=n_m)
    out = np.empty((counts.shape[0], n), dtype=int)
    logP = np.log(np.maximum(P, 1e-12))
    for b0 in range(0, counts.shape[0], chunk):
        cb = counts[b0:b0 + chunk]                                  # [b, M]
        S_all = cb @ S_m                                            # [b, K]
        N_all = cb @ c_m                                            # [b]
        other = S_all[:, None, :] - cb[:, :, None] * S_m[None]      # [b, M, K]
        n_other = N_all[:, None] - cb * c_m[None]                   # [b, M]
        with np.errstate(invalid="ignore", divide="ignore"):
            prior = other / n_other[:, :, None]
        prior = np.where(n_other[:, :, None] > 0, prior, 1.0)
        logprior = np.log(np.maximum(prior, _EPS))[:, idx, :]       # [b, n, K]
        pred = np.argmax(logP[None] - logprior, axis=2)
        out[b0:b0 + chunk] = np.where(valid[None], pred, -1)
    return out


def _metrics(counts, idx, y, yhat, pc_pred, n_cls, support_cls) -> dict[str, np.ndarray]:
    """Every bootstrapped metric for each draw (a row of `counts`). Shapes [B] or [B, C]."""
    W = counts[:, idx]                                              # [B, n]
    n_p = n_cls + 1                                                 # + NONE
    onehot = np.zeros((len(y), n_cls * n_p))
    onehot[np.arange(len(y)), y * n_p + yhat] = 1
    C = (W @ onehot).reshape(-1, n_cls, n_p)                        # [B, true, pred]
    tot = C.sum((1, 2))
    diag = np.stack([C[:, c, c] for c in range(n_cls)], 1)          # [B, C]
    sup = C.sum(2)                                                  # [B, C]
    col = C.sum(1)[:, :n_cls]                                       # [B, C] predicted as c
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(sup > 0, diag / sup, np.nan)
        f1_den = sup + col                                          # 2tp + fp + fn
        f1 = np.where(f1_den > 0, 2 * diag / f1_den, np.nan)[:, support_cls]
        po = diag.sum(1) / tot
        pe = (sup * col).sum(1) / tot ** 2
        kappa = np.where(pe < 1, (po - pe) / (1 - pe), np.nan)
    correct_pc = (pc_pred == y[None]).astype(float)                 # [B, n]
    with np.errstate(invalid="ignore", divide="ignore"):
        pc = (W * correct_pc).sum(1) / W.sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN, expected
        bal = np.nanmean(recall, axis=1)
        macro_f1 = np.nanmean(f1, axis=1) if f1.shape[1] else np.full(len(tot), np.nan)
    return {"accuracy": po, "balanced_accuracy": bal, "macro_f1": macro_f1, "kappa": kappa,
            "prior_corrected_accuracy": pc, "recall": recall}


def summarize(rows: list[dict], classes: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """rows: dicts with sport, label, probs, match_id. Errors count as wrong."""
    classes = list(classes) + sorted({r["sport"] for r in rows} - set(classes))
    rows = [{**r, "label": r.get("label") or NONE, "probs": r.get("probs") or {}} for r in rows]
    n_cls = len(classes)
    y = np.array([classes.index(r["sport"]) for r in rows], dtype=int)
    yhat = np.array([classes.index(r["label"]) if r["label"] in classes else n_cls
                     for r in rows], dtype=int)
    support_cls = sorted(set(y.tolist()))
    P, valid = _prob_matrix(rows, classes)
    idx, counts = boot_counts([r["match_id"] for r in rows], n_boot, seed)
    ones = np.ones((1, counts.shape[1]))
    point = _metrics(ones, idx, y, yhat, _prior_corrected_pred(P, valid, idx, ones),
                     n_cls, support_cls)
    boot = _metrics(counts, idx, y, yhat, _prior_corrected_pred(P, valid, idx, counts),
                    n_cls, support_cls) if n_boot else None
    out = {
        "n": len(rows),
        "n_matches": len({r["match_id"] for r in rows}),
        "n_unparsed": int(sum(r["label"] == NONE for r in rows)),
        "chance": 1 / len(classes),
        "log_loss": _log_loss(rows), "log_loss_uniform": float(np.log(len(classes))),
        # decision models report their input size: flags truncation by small contexts
        "max_input_tokens": max((r["input_tokens"] for r in rows if r.get("input_tokens")),
                                default=None),
        "confusion": {t: {p: sum(r["sport"] == t and (r["label"] if r["label"] in classes
                                                      else NONE) == p for r in rows)
                          for p in classes + [NONE]} for t in classes},
        # the response bias itself: how often each option is the answer
        "predicted_share": {p: float(np.mean(yhat == i)) if len(rows) else float("nan")
                            for i, p in enumerate(classes + [NONE])},
    }
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "prior_corrected_accuracy"):
        out[k] = float(point[k][0])
        out[f"{k}_ci"] = _ci(boot[k]) if boot else (float("nan"), float("nan"))
    present = [classes[c] for c in support_cls]
    out["per_class_recall"] = {c: float(point["recall"][0, classes.index(c)]) for c in present}
    out["per_class_recall_ci"] = {c: _ci(boot["recall"][:, classes.index(c)]) if boot
                                  else (float("nan"), float("nan")) for c in present}
    return out


def _correct(r: dict) -> float:
    """Per-item correctness; `correct` (e.g. averaged over replicates) wins if present."""
    if "correct" in r:
        return float(r["correct"])
    return float(r.get("label") == r["sport"])


def bootstrap_p(vals: np.ndarray) -> float:
    """Two-sided p-value of H0: statistic = 0, from its bootstrap distribution.

    Consistent with the percentile CI: p < 0.05 roughly iff the 95% CI excludes 0.
    The +1 terms keep p > 0 with a finite number of draws.
    """
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return float("nan")
    b = len(vals)
    lo = (np.sum(vals <= 0) + 1) / (b + 1)
    hi = (np.sum(vals >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def paired_difference(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy(a) - accuracy(b) on the clips both conditions share, match-clustered.

    Used for motion vs. motion_shuffled (does order matter?) and motion vs.
    formation (does motion add anything over a single frame?). Returns the
    difference, its percentile CI and a two-sided bootstrap p-value (same draws).
    """
    ia = {r["clip_id"]: r for r in a}
    ib = {r["clip_id"]: r for r in b}
    common = sorted(set(ia) & set(ib))
    d = np.array([_correct(ia[c]) - _correct(ib[c]) for c in common])
    diff = float(d.mean()) if len(d) else float("nan")
    if not len(d):
        return {"n": 0, "diff": diff, "ci": (float("nan"), float("nan")), "p": float("nan")}
    idx, counts = boot_counts([ia[c]["match_id"] for c in common], n_boot, seed)
    vals = _wmean(d, counts[:, idx])
    return {"n": len(d), "diff": diff, "ci": _ci(vals), "p": bootstrap_p(vals)}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (one family). NaNs stay NaN and are not counted."""
    ok = [i for i, p in enumerate(pvals) if p == p]
    order = sorted(ok, key=lambda i: pvals[i])
    m = len(order)
    out = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out
