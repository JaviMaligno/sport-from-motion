"""Within-sport source identification: can a classifier tell the capture source of a clip
when the sport is held fixed? A non-degenerate leak check (unlike `nuisance`).

For each sport, keep the sources with >= 2 matches in that sport (grouped CV by match
cannot test a one-match source: its clips are never both in train and test), predict the
source with grouped out-of-fold CV by match (same loop as every specialist), from
  - kinematic: the 14 kinematic baseline features -> balanced logistic
  - minirocket: the motion-condition invariant series -> MiniRocket -> logistic
Balanced accuracy (chance = 1 / n_sources) with a 95% CI from a match bootstrap stratified
by source (matches resampled within each source), and an exact match-level permutation
test when the number of distinct source assignments is small (<= MAX_PERM), else a
random one: the source labels are shuffled between matches, the CV rerun, and
p = (#{null >= observed} + 1) / (n_null + 1)  [exact enumeration includes the observed].

    PYTHONPATH=src .venv/bin/python scripts/analysis/source_id.py runs/final
    -> runs/final/source_id.json
"""
import collections
import itertools
import json
import pathlib
import sys

import numpy as np

from motion_sport.baselines import featurize
from motion_sport.learners import (clip_series, grouped_oof, logistic_fit_predict,
                                   minirocket_fit_predict)
from motion_sport.pipeline import stable_seed
from motion_sport.schema import load_clips

MAX_PERM = 200
N_BOOT = 2000


def bal_acc(y, pred, classes):
    return float(np.mean([np.mean([p == c for t, p in zip(y, pred) if t == c]) for c in classes]))


def strat_boot(y, pred, groups, classes, seed=0):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, (t, g) in enumerate(zip(y, groups)):
        by[t][g].append(i)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        idx = []
        for c in classes:
            gs = list(by[c])
            for k in rng.integers(0, len(gs), len(gs)):
                idx += by[c][gs[k]]
        vals.append(bal_acc([y[i] for i in idx], [pred[i] for i in idx], classes))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def assignments(match_src, rng):
    """Match-level relabellings keeping each source's number of matches."""
    ms = sorted(match_src)
    counts = collections.Counter(match_src.values())
    srcs = sorted(counts)
    # number of distinct assignments = multinomial(len(ms); counts)
    from math import factorial
    total = factorial(len(ms))
    for c in counts.values():
        total //= factorial(c)
    if total <= MAX_PERM and len(srcs) == 2:
        small = min(srcs, key=lambda s: counts[s])
        other = [s for s in srcs if s != small][0]
        for combo in itertools.combinations(ms, counts[small]):
            yield {m: (small if m in combo else other) for m in ms}, True
        return
    labels = [match_src[m] for m in ms]
    for _ in range(MAX_PERM):
        yield dict(zip(ms, rng.permutation(labels))), False


def run(name, make_fp, y, groups, classes, match_src, perm=True):
    fp = make_fp(y)
    pred, _ = grouped_oof(y, groups, fp)
    obs = bal_acc(y, pred, classes)
    ci = strat_boot(y, pred, groups, classes)
    rec = {c: float(np.mean([p == c for t, p in zip(y, pred) if t == c])) for c in classes}
    share = {c: float(np.mean([p == c for p in pred])) for c in classes}
    per_match = {}
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        per_match[g] = {"source": y[idx[0]], "n": len(idx),
                        "recall": float(np.mean([pred[i] == y[i] for i in idx]))}
    out = {"balanced_accuracy": obs, "ci95": ci, "chance": 1 / len(classes), "recall": rec,
           "predicted_share": share, "per_match": per_match}
    if perm:
        rng = np.random.default_rng(0)
        null, exact = [], False
        for assign, exact in assignments(match_src, rng):
            yp = [assign[g] for g in groups]
            if yp == y:
                continue  # the observed assignment; counted once below
            pp, _ = grouped_oof(yp, groups, make_fp(yp))
            null.append(bal_acc(yp, pp, classes))
        null = np.array(null)
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        out["permutation"] = {"exact": exact, "n_null": int(len(null)), "p": float(p),
                              "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                              "null_max": float(null.max())}
    print(f"  {name:10s} balanced acc {obs:.3f} [{ci[0]:.2f}; {ci[1]:.2f}] (chance {1 / len(classes):.2f}) "
          f"recall {', '.join(f'{c} {v:.2f}' for c, v in rec.items())} | predicted share "
          f"{', '.join(f'{c} {v:.2f}' for c, v in share.items())}"
          + (f" | permutation ({'exact' if out['permutation']['exact'] else 'random'}, "
             f"{out['permutation']['n_null']} null) p={out['permutation']['p']:.3f}, null mean "
             f"{out['permutation']['null_mean']:.3f}, null 95th pct {out['permutation']['null_p95']:.3f}"
             if perm else ""))
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    clips = load_clips(root / "clips")
    report = {}
    for sport in sorted({c.sport for c in clips}):
        sc = [c for c in clips if c.sport == sport]
        mbys = collections.defaultdict(set)
        for c in sc:
            mbys[c.source].add(c.match_id)
        summary = {s: {"clips": sum(c.source == s for c in sc), "matches": len(m)} for s, m in sorted(mbys.items())}
        keep = sorted(s for s, m in mbys.items() if len(m) >= 2)
        print(f"== {sport}: sources {summary}")
        if len(keep) < 2:
            print(f"  not testable: {len(keep)} source(s) with >= 2 matches ({keep})")
            report[sport] = {"sources": summary, "testable": False, "kept": keep}
            continue
        sub = [c for c in sc if c.source in keep]
        y = [c.source for c in sub]
        groups = [f"{c.source}:{c.match_id}" for c in sub]
        match_src = {g: s for g, s in zip(groups, y)}
        print(f"  kept {keep}: {len(sub)} clips, {len(match_src)} matches")
        Xk = featurize(sub, "kinematic")
        Xs = np.stack([clip_series(c, "motion", stable_seed(c.clip_id, "0")) for c in sub])
        report[sport] = {"sources": summary, "testable": True, "kept": keep, "n": len(sub),
                         "kinematic": run("kinematic", lambda yy: logistic_fit_predict(Xk, yy),
                                          y, groups, keep, match_src),
                         "minirocket": run("minirocket", lambda yy: minirocket_fit_predict(Xs, yy),
                                           y, groups, keep, match_src)}
    (root / "source_id.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
