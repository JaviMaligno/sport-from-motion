"""Secondary analyses of the final run that `report` does not compute
(docs/preregistration.md section 7). Exploratory: raw p, never in the Holm family.

Reads a primary view (scripts/primary_view.py: replicate 1 of the planned models, the
specialists on the same clips) and writes one JSON:

- `manual_contrasts`: motion/trails - motion/sheet (every chat model) and, for the
  models with video, motion/video - motion/sheet, with `evaluate.paired_difference`
  (match-clustered bootstrap, seed 0).
- `jev_text_vs_json`: per Jev cell, accuracy in plain-text and JSON state format, the
  share of clips with the same label, and the paired difference (descriptive).
- `slices`: per chat model, by sport and by source (TeamTrack split by sport), the
  accuracy of motion/sheet with its match-clustered CI and the paired contrasts
  `order` and `motion_over_shape` inside the slice. With fewer than 5 matches there is
  no CI or p (`too_few_matches`).
- `nfl_phase`: recall of American football in motion/sheet and motion_shuffled/sheet by
  the clip's offset from the snap (`meta.snap_offset_s`, D1/D16), descriptive.

    PYTHONPATH=src .venv/bin/python scripts/analysis/final_extras.py runs/final-primary \
        --n-boot 10000 --out runs/analysis-final/extras.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from motion_sport.evaluate import _acc, cluster_bootstrap, paired_difference

SPECIALIST_PREFIXES = ("baseline:", "minirocket", "deepsets", "probe:")
PHASE_BINS = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, float("inf"))]


def load(view: pathlib.Path) -> dict[tuple, list[dict]]:
    """(model, condition, repr, prompt_style) -> rows (replicate 1 only in a primary view)."""
    cells = {}
    for p in sorted((view / "predictions").glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        r = rows[0]
        style = r.get("prompt_style") or "neutral"
        cells[(r["model"], r.get("condition"), r.get("repr"), style)] = rows
    return cells


def slim(d: dict) -> dict:
    return {k: d[k] for k in ("diff", "ci", "p", "n", "n_matches", "too_few_matches")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("view")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    view = pathlib.Path(a.view)
    items = [json.loads(line) for line in (view / "items.jsonl").open()]
    source = {it["clip_id"]: it["source"] for it in items}
    cells = load(view)
    chat = sorted({k[0] for k in cells if not k[0].startswith(SPECIALIST_PREFIXES)
                   and not k[0].startswith("jev")})
    out: dict = {"n_boot": a.n_boot, "manual_contrasts": [], "jev_text_vs_json": [],
                 "slices": [], "nfl_phase": []}

    # 1. representation contrasts done by hand (section 7)
    for m in chat:
        sheet = cells.get((m, "motion", "sheet", "neutral"))
        for rep in ("trails", "video"):
            other = cells.get((m, "motion", rep, "neutral"))
            if sheet and other:
                out["manual_contrasts"].append(
                    {"model": m, "a": f"motion/{rep}", "b": "motion/sheet",
                     **slim(paired_difference(other, sheet, a.n_boot))})
        vs, vss = cells.get((m, "motion", "video", "neutral")), cells.get((m, "motion_shuffled", "video", "neutral"))
        ss = cells.get((m, "motion_shuffled", "sheet", "neutral"))
        if vss and ss:
            out["manual_contrasts"].append(
                {"model": m, "a": "motion_shuffled/video", "b": "motion_shuffled/sheet",
                 **slim(paired_difference(vss, ss, a.n_boot))})

    # 2. Jev: plain-text state vs JSON state (descriptive)
    for (m, cond, rep, style), rows in sorted(cells.items()):
        if not m.startswith("jev") or m.endswith("+json"):
            continue
        js = cells.get((m + "+json", cond, rep, style))
        if not js:
            continue
        a_, b_ = {r["clip_id"]: r for r in rows}, {r["clip_id"]: r for r in js}
        common = sorted(set(a_) & set(b_))
        out["jev_text_vs_json"].append({
            "cell": f"{cond}/{rep}" + ("" if style == "neutral" else f"[{style}]"),
            "acc_text": _acc(rows), "acc_json": _acc(js),
            "same_label": float(np.mean([a_[c].get("label") == b_[c].get("label") for c in common])),
            "label_share_text": dict(collections.Counter(r.get("label") for r in rows)),
            "label_share_json": dict(collections.Counter(r.get("label") for r in js)),
            "json_minus_text": slim(paired_difference(js, rows, a.n_boot))})

    # 3. slices by sport and by source
    def slice_key(kind: str, r: dict) -> str:
        if kind == "sport":
            return r["sport"]
        s = source[r["clip_id"]]
        return f"{s}/{r['sport']}" if s == "teamtrack" else s

    for m in chat:
        ms = cells.get((m, "motion", "sheet", "neutral"))
        sh = cells.get((m, "motion_shuffled", "sheet", "neutral"))
        fo = cells.get((m, "formation", "sheet", "neutral"))
        if not ms:
            continue
        for kind in ("sport", "source"):
            groups = collections.defaultdict(list)
            for r in ms:
                groups[slice_key(kind, r)].append(r)
            for g, rows in sorted(groups.items()):
                keep = {r["clip_id"] for r in rows}
                n_m = len({r["match_id"] for r in rows})
                ci = cluster_bootstrap(rows, _acc, n_boot=2000) if n_m >= 5 else (float("nan"),) * 2
                rec = {"model": m, "kind": kind, "slice": g, "n": len(rows), "n_matches": n_m,
                       "acc_motion_sheet": _acc(rows), "acc_ci": ci}
                if sh:
                    sub = [r for r in sh if r["clip_id"] in keep]
                    rec["acc_shuffled_sheet"] = _acc(sub)
                    rec["order"] = slim(paired_difference(rows, sub, a.n_boot))
                if fo:
                    sub = [r for r in fo if r["clip_id"] in keep]
                    rec["acc_formation_sheet"] = _acc(sub)
                    rec["motion_over_shape"] = slim(paired_difference(rows, sub, a.n_boot))
                out["slices"].append(rec)

    # 4. NFL: recall of American football by offset from the snap
    offset = {}
    for cid in {r["clip_id"] for rows in cells.values() for r in rows
                if r["sport"] == "american_football"}:
        z = np.load(view / "clips" / f"{cid}.npz", allow_pickle=True)
        offset[cid] = json.loads(str(z["meta_json"]))["meta"].get("snap_offset_s")
    for m in chat:
        for cond in ("motion", "motion_shuffled"):
            rows = [r for r in cells.get((m, cond, "sheet", "neutral"), [])
                    if r["sport"] == "american_football"]
            for lo, hi in PHASE_BINS:
                sub = [r for r in rows if offset.get(r["clip_id"]) is not None
                       and lo <= offset[r["clip_id"]] < hi]
                if sub:
                    out["nfl_phase"].append({"model": m, "cell": f"{cond}/sheet", "bin": [lo, hi],
                                             "n": len(sub), "recall_af": _acc(sub)})
    out["nfl_offsets"] = {"n": len(offset), "quartiles": np.percentile(
        [v for v in offset.values() if v is not None], [0, 25, 50, 75, 100]).tolist()}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}: {len(out['manual_contrasts'])} manual contrasts, "
          f"{len(out['jev_text_vs_json'])} Jev cells, {len(out['slices'])} slices, "
          f"{len(out['nfl_phase'])} NFL phase bins")


if __name__ == "__main__":
    main()
