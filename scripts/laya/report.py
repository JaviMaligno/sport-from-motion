"""Analysis of the Laya arm, as pre-registered in docs/preregistration-laya.md.

    PYTHONPATH=src .venv/bin/python scripts/laya/report.py runs/laya/all runs/final [laya-ft-e8] > runs/laya/report.json

Reads the per-fold prediction files downloaded from Kaggle, joins the folds of each
(model, condition, seed), and reports:
  - primary: the 4 contrasts on the 400 evaluation clips, own Holm family, B = 10,000;
  - secondary: per-condition summaries (400 and all 1,526), seed spread, sign per seed,
    kinematics contrasts and comparisons with Jev / the chat models / the specialists
    (raw p, no Holm).
"""
import collections
import glob
import json
import pathlib
import re
import sys

import numpy as np

from motion_sport.evaluate import holm, paired_difference, replicate_summary, summarize

B = 10_000
CONDS = ["motion", "motion_shuffled", "formation", "kinematics", "kinematics_solo"]
CLASSES = ["american_football", "basketball", "handball", "soccer"]


def load_laya(out_dir):
    cells = collections.defaultdict(list)  # (model, cond, seed) -> rows
    folds = collections.defaultdict(set)
    for p in glob.glob(f"{out_dir}/*__fold*.jsonl"):
        m = re.match(r"laya-(zs|ft(?:-([a-z0-9]+))?-s(\d+))__(.+)__fold(\d)\.jsonl$", pathlib.Path(p).name)
        if not m:  # positive controls and sweeps are not part of the analysis
            continue
        model = "laya-zs" if m.group(1) == "zs" else "laya-ft" + (f"-{m.group(2)}" if m.group(2) else "")
        seed = int(m.group(3) or 0)
        key = (model, m.group(4), seed)
        cells[key] += [json.loads(l) for l in open(p)]
        folds[key].add(int(m.group(5)))
    incomplete = {f"{k[0]}/{k[1]}/s{k[2]}": sorted(v) for k, v in folds.items() if len(v) < 5}
    return cells, incomplete


def merged(cells, model, cond, eval_only=True):
    """Per-clip correctness averaged over seeds (clips answered by every seed)."""
    reps = [rows for (m, c, s), rows in sorted(cells.items()) if m == model and c == cond]
    if not reps:
        return None, None
    if eval_only:
        reps = [[r for r in rows if r["eval400"]] for rows in reps]
    return replicate_summary(reps, n_boot=B), reps


def load_final(final_dir, pattern, eval_ids):
    p = pathlib.Path(final_dir) / "predictions" / pattern
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    return [r for r in rows if r["clip_id"] in eval_ids] if eval_ids is not None else rows


def contrast(a, b):
    r = paired_difference(a, b, n_boot=B)
    return {k: r[k] for k in ("n", "n_matches", "diff", "ci", "p")}


def main(out_dir, final_dir, ft_model="laya-ft-e8"):
    """`ft_model`: the fine-tuned configuration under test (DL1: laya-ft-e8; laya-ft is the
    original section-4 configuration). `out_dir` may hold several batches' files."""
    cells, incomplete = load_laya(out_dir)
    ft = {c: merged(cells, ft_model, c) for c in CONDS}
    zs = {c: merged(cells, "laya-zs", c) for c in CONDS}
    eval_ids = {r["clip_id"] for (m, c, s), rows in cells.items() for r in rows if r["eval400"]}
    mr = load_final(final_dir, "minirocket__motion__series.jsonl", eval_ids)

    def rows(x):
        return x[0]["rows"] if x and x[0] else None

    primary = {
        "ft_gain": (rows(ft["motion"]), rows(zs["motion"])),
        "order": (rows(ft["motion"]), rows(ft["motion_shuffled"])),
        "motion_over_shape": (rows(ft["motion"]), rows(ft["formation"])),
        "vs_minirocket": (rows(ft["motion"]), mr),
    }
    prim = {k: (contrast(a, b) if a and b else None) for k, (a, b) in primary.items()}
    names = [k for k, v in prim.items() if v]
    for k, ph in zip(names, holm([prim[k]["p"] for k in names])):
        prim[k]["p_holm"] = ph
        prim[k]["significant"] = ph < 0.05

    # sign of each primary contrast per seed
    per_seed = {}
    seeds = sorted({s for (m, c, s) in cells if m == ft_model})
    for s in seeds:
        one = {c: [r for r in cells.get((ft_model, c, s), []) if r["eval400"]] for c in CONDS}
        z = rows(zs["motion"])
        per_seed[s] = {
            "ft_gain": contrast(one["motion"], z)["diff"] if one["motion"] and z else None,
            "order": contrast(one["motion"], one["motion_shuffled"])["diff"] if one["motion_shuffled"] else None,
            "motion_over_shape": contrast(one["motion"], one["formation"])["diff"] if one["formation"] else None,
            "vs_minirocket": contrast(one["motion"], mr)["diff"] if one["motion"] and mr else None,
        }

    summaries = {}
    for model in sorted({m for (m, c, s) in cells}):
        for c in CONDS:
            reps = [rows_ for (m, cc, s), rows_ in sorted(cells.items()) if m == model and cc == c]
            if not reps:
                continue
            summaries[f"{model}/{c}"] = {
                "eval400_per_seed": [summarize([r for r in rep if r["eval400"]], CLASSES, n_boot=B) for rep in reps],
                "all1526_per_seed_accuracy": [float(np.mean([r["label"] == r["sport"] for r in rep])) for rep in reps],
                "replicates_eval400": {k: v for k, v in merged(cells, model, c)[0].items() if k != "rows"},
            }
            for s in summaries[f"{model}/{c}"]["eval400_per_seed"]:
                s.pop("confusion", None)

    secondary = {}
    if rows(ft["kinematics"]) and rows(ft["kinematics_solo"]):
        secondary["kinematics_vs_solo"] = contrast(rows(ft["kinematics"]), rows(ft["kinematics_solo"]))
        secondary["motion_vs_kinematics"] = contrast(rows(ft["motion"]), rows(ft["kinematics"]))
    others = {
        "vs_jev_text": "jev-openrouter__~typesafe_jev-latest__motion__text.jsonl",
        "vs_gpt56sol_text": "azure-openai__gpt-5.6-sol__motion__text.jsonl",
        "vs_gpt56terra_text": "azure-openai__gpt-5.6-terra__motion__text.jsonl",
        "vs_deepsets_motion": "deepsets__motion__tracks.jsonl",
    }
    for k, f in others.items():
        b = load_final(final_dir, f, eval_ids)
        if b and rows(ft["motion"]):
            secondary[k] = contrast(rows(ft["motion"]), b)
    for c in ("motion_shuffled", "kinematics", "kinematics_solo"):
        b = load_final(final_dir, f"minirocket__{c}__series.jsonl", eval_ids)
        if b and rows(ft[c]):
            secondary[f"vs_minirocket_{c}"] = contrast(rows(ft[c]), b)

    json.dump({"ft_model": ft_model, "incomplete_cells": incomplete, "primary": prim, "primary_per_seed": per_seed,
               "summaries": summaries, "secondary_raw_p": secondary}, sys.stdout, indent=2, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:4])
