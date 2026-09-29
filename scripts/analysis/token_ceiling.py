"""Output tokens against the 1024 `--max-tokens` ceiling on the Anthropic route (D21).

Before commit d6a3833 the Anthropic payload sent `max_tokens = 1024` with no reasoning
headroom, and Claude 5.x counts its adaptive thinking against it. A row whose
`usage.output_tokens` equals the ceiling was cut; the rest stopped on their own. For each
Anthropic cell this prints the rows, the maximum output tokens, how many rows came within
100 tokens of the ceiling or reached it, and the errors. With `--sens` it also compares a
sensitivity copy that re-ran the errored rows with the fix: how many were recovered and how
many of those are correct, against the accuracy of the rows that were valid in the
original. Nothing is sent to any API.

    .venv/bin/python scripts/analysis/token_ceiling.py runs/final runs/final-d8 \
        --sens runs/final-sens --json runs/analysis-final/token_ceiling.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

CEILING = 1024


def rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def out_tokens(r: dict) -> int | None:
    return (r.get("usage") or {}).get("output_tokens")


def is_error(r: dict) -> bool:
    return bool(r.get("error")) and not r.get("label")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="items dirs with predictions/")
    ap.add_argument("--sens", help="sensitivity copy of the first dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {"ceiling": CEILING, "cells": [], "sensitivity": []}
    for d in map(pathlib.Path, a.dirs):
        for p in sorted((d / "predictions").glob("*anthropic*.jsonl")):
            rs = rows(p)
            toks = [t for t in map(out_tokens, rs) if t is not None]
            c = {"dir": d.name, "file": p.stem, "rows": len(rs),
                 "max_output_tokens": max(toks) if toks else None,
                 "within_100": sum(t >= CEILING - 100 for t in toks),
                 "at_ceiling": sum(t >= CEILING for t in toks),
                 "valid_at_ceiling": sum(1 for r in rs if r.get("label")
                                         and (out_tokens(r) or 0) >= CEILING),
                 "errors": sum(map(is_error, rs))}
            res["cells"].append(c)
            print(f"{c['dir']:12} {c['file']:62} n={c['rows']:3d} max={c['max_output_tokens']} "
                  f">={CEILING - 100}:{c['within_100']:3d} at {CEILING}:{c['at_ceiling']:3d} "
                  f"(valid {c['valid_at_ceiling']}) errors={c['errors']}")
    if a.sens:
        base, sens = pathlib.Path(a.dirs[0]), pathlib.Path(a.sens)
        for p in sorted((sens / "predictions").glob("*anthropic*.jsonl")):
            ref = base / "predictions" / p.name
            if not ref.exists() or ref.read_bytes() == p.read_bytes():
                continue
            o = {r["clip_id"]: r for r in rows(ref)}
            s = {r["clip_id"]: r for r in rows(p)}
            err = [k for k, r in o.items() if is_error(r)]
            changed = [k for k in o if json.dumps(o[k], sort_keys=True) != json.dumps(s[k], sort_keys=True)]
            rec = [s[k] for k in err if s[k].get("label")]
            valid = [r for r in o.values() if r.get("label")]
            v = {"file": p.stem, "errors_orig": len(err), "rows_changed": len(changed),
                 "changed_all_errors": set(changed) <= set(err),
                 "errors_orig_at_ceiling": sum((out_tokens(o[k]) or 0) >= CEILING for k in err),
                 "errors_by_sport": dict(collections.Counter(o[k]["sport"] for k in err)),
                 "recovered": len(rec),
                 "recovered_correct": sum(r["label"] == r["sport"] for r in rec),
                 "recovered_correct_by_sport": dict(collections.Counter(
                     r["sport"] for r in rec if r["label"] == r["sport"])),
                 "recovered_output_tokens_median": sorted(map(out_tokens, rec))[len(rec) // 2] if rec else None,
                 "recovered_output_tokens_max": max(map(out_tokens, rec)) if rec else None,
                 "errors_left": sum(is_error(s[k]) for k in err),
                 "orig_valid_rows": len(valid),
                 "orig_valid_correct": sum(r["label"] == r["sport"] for r in valid)}
            res["sensitivity"].append(v)
            print(f"sens {v['file']}: {v['errors_orig']} errors ({v['errors_orig_at_ceiling']} at the "
                  f"ceiling; {v['errors_by_sport']}); {v['rows_changed']} rows changed, all errors: "
                  f"{v['changed_all_errors']}; recovered {v['recovered']}, correct "
                  f"{v['recovered_correct']} ({v['recovered_correct_by_sport']}); left "
                  f"{v['errors_left']}; original valid rows correct {v['orig_valid_correct']}/"
                  f"{v['orig_valid_rows']}; recovered output tokens median "
                  f"{v['recovered_output_tokens_median']}, max {v['recovered_output_tokens_max']}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
