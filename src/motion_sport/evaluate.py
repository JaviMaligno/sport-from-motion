"""Metrics with match-clustered bootstrap CIs.

Lesson carried over from wheres-the-ball Part 1 (a 64% on 14 items whose CI
spanned 36-86%): every number gets a CI from day one, and the resampling unit is
the *match*, not the clip — clips of one match are not independent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


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
    by_match = defaultdict(list)
    for r in rows:
        by_match[r["match_id"]].append(r)
    keys = list(by_match)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for k in rng.choice(keys, size=len(keys), replace=True) for r in by_match[k]]
        vals.append(stat(sample))
    vals = np.array([v for v in vals if not np.isnan(v)])
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(rows: list[dict], classes: list[str], n_boot: int = 2000) -> dict:
    """rows: dicts with sport, label, probs, match_id. Errors count as wrong."""
    rows = [{**r, "label": r.get("label") or "__none__", "probs": r.get("probs") or {}} for r in rows]
    bal = lambda s: _balanced_acc(s, classes)  # noqa: E731
    out = {
        "n": len(rows),
        "n_matches": len({r["match_id"] for r in rows}),
        "n_unparsed": sum(r["label"] == "__none__" for r in rows),
        "chance": 1 / len(classes),
        "accuracy": _acc(rows), "accuracy_ci": cluster_bootstrap(rows, _acc, n_boot),
        "balanced_accuracy": bal(rows), "balanced_accuracy_ci": cluster_bootstrap(rows, bal, n_boot),
        "log_loss": _log_loss(rows), "log_loss_uniform": float(np.log(len(classes))),
        "confusion": {t: {p: sum(r["sport"] == t and r["label"] == p for r in rows)
                          for p in classes + ["__none__"]} for t in classes},
    }
    return out


def paired_difference(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy(a) - accuracy(b) on the clips both conditions share, match-clustered.

    Used for motion vs. motion_shuffled (does order matter?) and motion vs.
    formation (does motion add anything over a single frame?).
    """
    ia = {r["clip_id"]: r for r in a}
    ib = {r["clip_id"]: r for r in b}
    common = sorted(set(ia) & set(ib))
    pairs = [{"match_id": ia[c]["match_id"],
              "d": float(ia[c].get("label") == ia[c]["sport"]) - float(ib[c].get("label") == ib[c]["sport"])}
             for c in common]
    stat = lambda s: float(np.mean([p["d"] for p in s])) if s else float("nan")  # noqa: E731
    return {"n": len(pairs), "diff": stat(pairs), "ci": cluster_bootstrap(pairs, stat, n_boot, seed)}
