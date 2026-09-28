"""Primary-analysis view of the final run (docs/preregistration.md, section 5).

`report` merges the replicates of a cell into one cell averaged over the clips that
*every* replicate answered. In the final run replicates 2-3 cover only the first 200
clips, so on the full directory every primary contrast that touches motion/sheet or
motion_shuffled/sheet would be computed on 200 clips instead of 400. The pre-registered
primary analysis therefore reads replicate 1 only, of the pre-registered models only:

    .venv/bin/python scripts/primary_view.py runs/final runs/final-primary
    .venv/bin/motion-sport report --items runs/final-primary --n-boot 10000

The view is symlinks (config.json, items.jsonl, clips/ and the selected prediction
files); nothing in the source directory is modified. If the source has a plan.json
(written by final_run.sh), the view gets its replicate-1 cells of the planned models,
so `report` flags planned cells with missing rows there too.

`--clips-from-models`: the specialists (trained on every clip) are restricted to the
clip ids the models were asked about (union over the selected model files), so a
per-clip comparison with a model is like for like. Those specialist files are written
as filtered copies instead of symlinks.

    .venv/bin/python scripts/primary_view.py runs/final runs/final-primary --clips-from-models
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

# the pre-registered models (chat models form the Holm family; Jev has no primary cell)
PLANNED = {
    "azure-openai:gpt-5.6-sol", "azure-openai:gpt-5.6-terra",
    "vertex-anthropic:claude-sonnet-5", "vertex-anthropic:claude-opus-5-5",
    "vertex:gemini-3.1-pro-preview",
    "jev-openrouter:~typesafe/jev-latest", "jev-openrouter:~typesafe/jev-latest+json",
}
# trained on our clips, no API: kept as reference (never a primary cell: repr != sheet)
SPECIALIST_PREFIXES = ("baseline:", "minirocket", "deepsets", "probe:")


def _safe(model: str) -> str:
    """Model id as it appears in a prediction file name (pipeline._pred_path)."""
    return model.replace(":", "__").replace("/", "_")


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_view(items: str | pathlib.Path, out: str | pathlib.Path,
               clips_from_models: bool = False) -> dict:
    """-> {"kept": [...], "skipped": [...], "restricted": {file: (n_before, n_after)},
    "model_clips": n, "model_clip_sets_equal": bool}"""
    src, out = pathlib.Path(items).resolve(), pathlib.Path(out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "items.jsonl", "clips"):
        link = out / name
        if not link.exists():
            os.symlink(src / name, link)
    kept, skipped, specialists, model_sets = [], [], [], []
    for p in sorted((src / "predictions").glob("*.jsonl")):
        first = next((json.loads(line) for line in p.open() if line.strip()), {})
        model = first.get("model", "")
        replicate = int(first.get("replicate") or 1)
        if re.search(r"__r\d+$", p.stem):
            replicate = max(replicate, int(p.stem.rsplit("__r", 1)[1]))
        specialist = model.startswith(SPECIALIST_PREFIXES)
        ok = replicate == 1 and (model in PLANNED or specialist)
        (kept if ok else skipped).append(p.name)
        if not ok:
            continue
        if specialist and clips_from_models:
            specialists.append(p)
            continue
        if not specialist:
            model_sets.append({r["clip_id"] for r in _rows(p)})
        link = out / "predictions" / p.name
        if not link.exists():
            os.symlink(p, link)
    info = {"kept": kept, "skipped": skipped, "restricted": {}}
    if clips_from_models:
        if not model_sets:
            raise SystemExit("--clips-from-models: no model prediction files in the view")
        clips = set().union(*model_sets)
        info["model_clips"] = len(clips)
        info["model_clip_sets_equal"] = all(s == model_sets[0] for s in model_sets)
        for p in specialists:
            rows = _rows(p)
            sel = [r for r in rows if r["clip_id"] in clips]
            dst = out / "predictions" / p.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()  # a symlink from an earlier view: never write through it
            dst.write_text("".join(json.dumps(r) + "\n" for r in sel))
            info["restricted"][p.name] = (len(rows), len(sel))
    plan = src / "plan.json"
    if plan.exists():
        planned = tuple(_safe(m) + "__" for m in PLANNED)
        cells = {k: v for k, v in json.loads(plan.read_text())["cells"].items()
                 if not re.search(r"__r\d+$", k) and k.startswith(planned)}
        (out / "plan.json").unlink(missing_ok=True)
        (out / "plan.json").write_text(json.dumps({"cells": cells}, indent=2, sort_keys=True) + "\n")
        info["planned_cells"] = len(cells)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--clips-from-models", action="store_true",
                    help="restrict the specialists to the clip ids the models answered")
    a = ap.parse_args()
    info = build_view(a.items, a.out, a.clips_from_models)
    print(f"{a.out}: {len(info['kept'])} prediction files linked, {len(info['skipped'])} left out")
    for s in info["skipped"]:
        print(f"  left out: {s}")
    if a.clips_from_models:
        print(f"specialists restricted to the {info['model_clips']} clips the models answered"
              + ("" if info["model_clip_sets_equal"] else
                 " (union: the model files do NOT all cover the same clips)"))
        for name, (before, after) in info["restricted"].items():
            print(f"  {name}: {before} -> {after} rows")
    if "planned_cells" in info:
        print(f"plan.json: {info['planned_cells']} planned replicate-1 cells of the planned models")


if __name__ == "__main__":
    main()
