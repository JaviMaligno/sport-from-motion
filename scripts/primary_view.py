"""Primary-analysis view of the final run (docs/preregistration.md, section 5).

`report` merges the replicates of a cell into one cell averaged over the clips that
*every* replicate answered. In the final run replicates 2-3 cover only the first 200
clips, so on the full directory every primary contrast that touches motion/sheet or
motion_shuffled/sheet would be computed on 200 clips instead of 400. The pre-registered
primary analysis therefore reads replicate 1 only, of the pre-registered models only:

    .venv/bin/python scripts/primary_view.py runs/final runs/final-primary
    .venv/bin/motion-sport report --items runs/final-primary --n-boot 10000

The view is symlinks (config.json, items.jsonl, clips/ and the selected prediction
files); nothing is copied or modified.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    a = ap.parse_args()
    src, out = pathlib.Path(a.items).resolve(), pathlib.Path(a.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "items.jsonl", "clips"):
        link = out / name
        if not link.exists():
            os.symlink(src / name, link)
    kept, skipped = [], []
    for p in sorted((src / "predictions").glob("*.jsonl")):
        first = next((json.loads(line) for line in p.open() if line.strip()), {})
        model = first.get("model", "")
        replicate = int(first.get("replicate") or 1)
        if re.search(r"__r\d+$", p.stem):
            replicate = max(replicate, int(p.stem.rsplit("__r", 1)[1]))
        ok = replicate == 1 and (model in PLANNED or model.startswith(SPECIALIST_PREFIXES))
        (kept if ok else skipped).append(p.name)
        link = out / "predictions" / p.name
        if ok and not link.exists():
            os.symlink(p, link)
    print(f"{out}: {len(kept)} prediction files linked, {len(skipped)} left out")
    for s in skipped:
        print(f"  left out: {s}")


if __name__ == "__main__":
    main()
