"""Export the text items of runs/final for the Laya fine-tuning arm.

One row per (clip, condition), with the fold that clip belongs to in the
specialists' cross-validation (StratifiedGroupKFold by match, random_state=0,
over the clip order of load_clips) and whether it is one of the 400 clips the
frontier models saw. Laya trains on four folds and predicts the fifth, exactly
like MiniRocket and DeepSets.

    .venv/bin/python scripts/laya/export.py runs/final runs/laya/export
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from motion_sport.pipeline import interleave, load_clips


def main(items_dir: str, out_dir: str, n_eval: int = 400) -> None:
    root, out = pathlib.Path(items_dir), pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips = load_clips(root / "clips")
    y = np.asarray([c.sport for c in clips])
    groups = [c.match_id for c in clips]
    fold = {}
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)
    for k, (_, test) in enumerate(splits):
        for i in test:
            fold[clips[i].clip_id] = k

    rows = [json.loads(l) for l in (root / "items.jsonl").open()]
    text = [r for r in rows if r["repr"] == "text"]
    motion = [r for r in text if r["condition"] == "motion"]
    eval_ids = {r["clip_id"] for r in interleave(motion)[:n_eval]}

    n = 0
    with (out / "items.jsonl").open("w") as f:
        for r in text:
            f.write(json.dumps({
                "item_id": r["item_id"], "clip_id": r["clip_id"], "sport": r["sport"],
                "source": r["source"], "match_id": r["match_id"], "condition": r["condition"],
                "fold": fold[r["clip_id"]], "eval400": r["clip_id"] in eval_ids,
                "state": r["state_text"], "instructions": r["decision_instructions"],
                "criteria": r["decision_criteria"],
            }) + "\n")
            n += 1
    print(f"{n} rows, {len(fold)} clips, {len(eval_ids)} eval clips -> {out / 'items.jsonl'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
