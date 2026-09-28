"""Paired specialist contrasts on all clips and on the first N interleaved clips (the
models' sample), with `evaluate.paired_difference` (match-clustered bootstrap, 2,000
draws, seed 0), as cited in D17 / D18.

    PYTHONPATH=src .venv/bin/python scripts/analysis/contrasts.py runs/final 400
    PYTHONPATH=src .venv/bin/python scripts/analysis/contrasts.py runs/final-d8 300
"""
import json
import pathlib
import sys

from motion_sport.evaluate import paired_difference
from motion_sport.pipeline import interleave

PAIRS = [("minirocket__motion__series", "minirocket__motion_shuffled__series"),
         ("minirocket__kinematics__series", "minirocket__kinematics_solo__series"),
         ("deepsets__motion__tracks", "deepsets__formation__tracks")]


def main():
    root = pathlib.Path(sys.argv[1])
    n_first = int(sys.argv[2])
    items = [json.loads(line) for line in (root / "items.jsonl").open()]
    ms = [it for it in items if it["condition"] == "motion" and it["repr"] == "sheet"]
    first = {it["clip_id"] for it in interleave(ms)[:n_first]}
    out = {}
    for a, b in PAIRS:
        pa, pb = root / "predictions" / f"{a}.jsonl", root / "predictions" / f"{b}.jsonl"
        if not (pa.exists() and pb.exists()):
            continue
        ra = [json.loads(line) for line in pa.open()]
        rb = [json.loads(line) for line in pb.open()]
        for name, keep in (("all", None), (f"first{n_first}", first)):
            sa = ra if keep is None else [r for r in ra if r["clip_id"] in keep]
            sb = rb if keep is None else [r for r in rb if r["clip_id"] in keep]
            d = paired_difference(sa, sb)
            out[f"{a} - {b} ({name})"] = d
            print(f"{a} - {b} [{name}]: {d['diff']:+.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] "
                  f"p={d['p']:.4f} n={d['n']} matches={d['n_matches']}")
    (root / "contrasts.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
