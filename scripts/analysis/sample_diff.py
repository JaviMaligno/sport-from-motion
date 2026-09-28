"""Which clips a rebuilt items directory changed: the first N interleaved clips (the models'
sample) in and out, with their interleave positions, and, over every clip in both sets,
whether the prepared clip is identical (xy arrays equal).

    PYTHONPATH=src .venv/bin/python scripts/analysis/sample_diff.py runs/final-v4 runs/final 400
"""
import json
import pathlib
import sys

import numpy as np

from motion_sport.pipeline import interleave
from motion_sport.schema import Clip


def order(root: pathlib.Path) -> list[str]:
    items = [json.loads(line) for line in (root / "items.jsonl").open()]
    return [it["clip_id"] for it in interleave(
        [it for it in items if it["condition"] == "motion" and it["repr"] == "sheet"])]


def main():
    old, new, n = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    oo, no = order(old), order(new)
    fo, fn = set(oo[:n]), set(no[:n])
    out = {"first": n,
           "out": [(oo.index(c), c) for c in oo[:n] if c not in fn],
           "in": [(no.index(c), c) for c in no[:n] if c not in fo]}
    common = sorted(set(oo) & set(no))
    changed = [c for c in common if not np.array_equal(
        Clip.load(old / "clips" / f"{c}.npz").xy, Clip.load(new / "clips" / f"{c}.npz").xy)]
    out.update(n_old=len(oo), n_new=len(no), removed=sorted(set(oo) - set(no)),
               added=sorted(set(no) - set(oo)), common=len(common), changed=changed,
               changed_in_first=[c for c in changed if c in fn])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
