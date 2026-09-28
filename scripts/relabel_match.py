"""Rewrite the match id of already-ingested clips, without re-downloading the source.

Used for TeamTrack handball: clips ingested before scripts/teamtrack_to_long_csv.py
grouped both halves as one game carry the half ("1st" / "2nd") as match_id, which
makes one game look like two independent matches to the grouped CV and the cluster
bootstrap. Clip ids (and so every seed derived from them) are left unchanged.

    # symlink copy of data/clips first: never relabel data/clips itself
    python scripts/relabel_match.py data/clips_tt_regroup --match tt-handball \
        --glob 'teamtrack-1st-*' --glob 'teamtrack-2nd-*' --sport handball

Each matching `.npz` is rewritten through a temporary file and `os.replace`, so a
symlink is *replaced* by a real file and its target (the original clip in data/clips)
is never written. Arrays other than meta_json are copied byte for byte.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import pathlib

import numpy as np


def relabel(directory: str | pathlib.Path, patterns: list[str], match: str,
            sport: str | None = None, dry_run: bool = False) -> dict:
    """-> counts: relabelled, unchanged (already `match`), wrong_sport (skipped),
    symlinks_replaced (of the relabelled)."""
    d = pathlib.Path(directory)
    counts = {"relabelled": 0, "unchanged": 0, "wrong_sport": 0, "symlinks_replaced": 0,
              "old_match_ids": {}}
    for path in sorted(d.glob("*.npz")):
        if not any(fnmatch.fnmatch(path.name, p) for p in patterns):
            continue
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays["meta_json"]))
        if sport and meta["sport"] != sport:
            counts["wrong_sport"] += 1
            continue
        if meta["match_id"] == match:
            counts["unchanged"] += 1
            continue
        old = meta["match_id"]
        counts["old_match_ids"][old] = counts["old_match_ids"].get(old, 0) + 1
        counts["relabelled"] += 1
        counts["symlinks_replaced"] += path.is_symlink()
        if dry_run:
            continue
        meta["match_id"] = match
        arrays["meta_json"] = np.array(json.dumps(meta))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)  # swaps the directory entry: a symlink's target is untouched
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", help="clips directory to rewrite (a symlink copy, not data/clips)")
    ap.add_argument("--match", required=True, help="new match_id")
    ap.add_argument("--glob", action="append", required=True,
                    help="file-name pattern of the clips to relabel (repeatable)")
    ap.add_argument("--sport", help="only relabel clips of this sport; others are counted")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = relabel(a.dir, a.glob, a.match, a.sport, a.dry_run)
    print(f"{a.dir}: {'would relabel' if a.dry_run else 'relabelled'} {c['relabelled']} clips "
          f"-> match {a.match!r} (from {c['old_match_ids']}; {c['symlinks_replaced']} symlinks "
          f"replaced by real files), {c['unchanged']} already {a.match!r}, "
          f"{c['wrong_sport']} skipped (sport != {a.sport})")


if __name__ == "__main__":
    main()
