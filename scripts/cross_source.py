"""Cross-source test: does a trained learner recognise the sport or the capture source?

Train on some sources, test on a source never seen in training (same sports).
If accuracy collapses on the held-out source, the learner was partly reading the
capture pipeline. Compare with grouped out-of-fold accuracy inside the training
sources.

    python scripts/cross_source.py runs/xsource --test-source teamtrack
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from motion_sport.baselines import featurize
from motion_sport.learners import (clip_series, grouped_oof, logistic_fit_predict,
                                   minirocket_fit_predict)
from motion_sport.schema import load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--test-source", default="teamtrack")
    ap.add_argument("--condition", default="motion")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian sigma in frames, applied to every clip alike (tracker jitter)")
    a = ap.parse_args()
    clips = load_clips(pathlib.Path(a.items) / "clips")
    if a.smooth:
        from scipy.ndimage import gaussian_filter1d
        clips = [c.replace(xy=gaussian_filter1d(c.xy, a.smooth, axis=0, mode="nearest"))
                 for c in clips]
    y = [c.sport for c in clips]
    groups = [f"{c.source}:{c.match_id}" for c in clips]
    held = np.array([c.source == a.test_source for c in clips])
    tr, te = np.flatnonzero(~held), np.flatnonzero(held)
    print("train:", dict(collections.Counter((clips[i].source, y[i]) for i in tr)))
    print("test: ", dict(collections.Counter((clips[i].source, y[i]) for i in te)))
    feats = {
        "kinematic": lambda: logistic_fit_predict(featurize(clips, "kinematic"), y),
        "minirocket": lambda: minirocket_fit_predict(
            np.stack([clip_series(c, a.condition, 0) for c in clips]), y),
    }
    out = {}
    for name, make in feats.items():
        fp = make()
        classes, p = fp(tr, te)
        pred = [classes[k] for k in p.argmax(axis=1)]
        acc = float(np.mean([pred[j] == y[i] for j, i in enumerate(te)]))
        conf = collections.Counter((y[i], pred[j]) for j, i in enumerate(te))
        # reference: grouped OOF inside the training sources only
        sub_y = [y[i] for i in tr]
        sub_g = [groups[i] for i in tr]
        fp_sub = lambda a_, b_: fp(tr[a_], tr[b_])  # noqa: E731
        oof, _ = grouped_oof(sub_y, sub_g, fp_sub)
        ref = float(np.mean([p_ == t for p_, t in zip(oof, sub_y) if p_]))
        out[name] = {"in_source_oof": ref, "held_out_source": acc,
                     "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items())}}
        print(f"{name:11s} dentro de fuentes (OOF) {ref:.2f} | fuente nunca vista ({a.test_source}) {acc:.2f}"
              f" | {out[name]['confusion']}")
    (pathlib.Path(a.items) / f"cross_source_{a.test_source}_s{a.smooth:g}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
