"""Specialist breakdown: all clips and the first N interleaved clips (the models' sample),
accuracy with a match-cluster CI, balanced accuracy, recall per class, per source and the
confusion, in the format of runs/final-v2/specialists_breakdown.txt.

    PYTHONPATH=src .venv/bin/python scripts/analysis/breakdown.py runs/final 400 \
        > runs/final/specialists_breakdown.txt
"""
import collections
import json
import pathlib
import sys

import numpy as np

from motion_sport.evaluate import _acc, cluster_bootstrap
from motion_sport.pipeline import interleave

AB = {"american_football": "AF", "basketball": "BB", "handball": "HB", "soccer": "SO"}


def main():
    root = pathlib.Path(sys.argv[1])
    n_first = int(sys.argv[2])
    items = [json.loads(l) for l in (root / "items.jsonl").open()]
    src = {it["clip_id"]: it["source"] for it in items}
    ms = [it for it in items if it["condition"] == "motion" and it["repr"] == "sheet"]
    first = [it["clip_id"] for it in interleave(ms)[:n_first]]
    fset = set(first)
    fi = [it for it in ms if it["clip_id"] in fset]
    print(f"first {n_first} interleaved (motion/sheet): by sport "
          f"{dict(sorted(collections.Counter(it['sport'] for it in fi).items()))} by source "
          f"{dict(sorted(collections.Counter((it['source'], it['sport']) for it in fi).items()))} "
          f"matches {len({it['match_id'] for it in fi})}")
    for f in sorted((root / "predictions").glob("*.jsonl")):
        rows = [json.loads(l) for l in f.open()]
        classes = sorted({r["sport"] for r in rows})
        print(f"== {f.stem}")
        for name, sub in (("all", rows), (f"first{n_first}", [r for r in rows if r["clip_id"] in fset])):
            if not sub:
                continue
            acc = _acc(sub)
            lo, hi = cluster_bootstrap(sub, _acc, n_boot=2000)
            rec = {c: np.mean([r["label"] == c for r in sub if r["sport"] == c]) for c in classes}
            bal = float(np.mean(list(rec.values())))
            nm = len({r["match_id"] for r in sub})
            print(f"  {name:8s} n={len(sub):5d} m={nm:3d} acc {acc:.3f} [{lo:.2f},{hi:.2f}] bal {bal:.3f} | "
                  "recall " + " ".join(f"{AB[c]} {rec[c]:.2f}" for c in classes))
            per = collections.defaultdict(list)
            for r in sub:
                s = src[r["clip_id"]]
                key = f"{s}/{AB[r['sport']]}" if s == "teamtrack" else s
                per[key].append(r)
            print("           per source: " + "; ".join(
                f"{k} {np.mean([r['label'] == r['sport'] for r in v]):.2f} (n={len(v)},m={len({r['match_id'] for r in v})})"
                for k, v in sorted(per.items())))
            conf = collections.Counter((r["sport"], r["label"]) for r in sub)
            print("           confusion (true -> pred): " + "; ".join(
                f"{AB[t]}: " + ",".join(f"{AB.get(p, p)}{conf[(t, p)]}" for p in classes if conf[(t, p)])
                for t in classes))


if __name__ == "__main__":
    main()
