"""The run plan of scripts/final_run.sh, and the checks against it.

    # plan.json from the launcher's own command lines (one `motion-sport run` per line)
    bash scripts/final_run.sh ... | python scripts/run_plan.py write runs/final
    # error summary: unrecovered errors per file, and planned cells with missing rows
    python scripts/run_plan.py check runs/final

`<items>/plan.json` is {"cells": {<prediction file stem>: planned rows}}. The stem is
the file `motion-sport run` writes for that cell (pipeline._pred_path), so a planned
cell that never started shows up as missing, not as silently absent. The planned rows
come from the `--limit` of each command (the first N of the interleaved order).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys

from motion_sport import pipeline
from motion_sport.backends.decision import is_decision_model


def plan_from_commands(items_dir: str | pathlib.Path, lines: list[str]) -> dict:
    """`motion-sport run ...` lines -> {"cells": {stem: limit}}. Lines that are not a
    run command (comments, blank) are ignored."""
    ap = argparse.ArgumentParser(add_help=False)
    for flag in ("--items", "--model", "--condition", "--repr", "--prompt-style",
                 "--state-format"):
        ap.add_argument(flag)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--workers")
    cells = {}
    for line in lines:
        words = shlex.split(line)
        if len(words) < 2 or not words[0].endswith("motion-sport") or words[1] != "run":
            continue
        a = ap.parse_args(words[2:])
        if a.limit is None:
            raise ValueError(f"run command without --limit, no planned size: {line}")
        name = a.model + ("+json" if is_decision_model(a.model) and a.state_format == "json"
                          else "")
        stem = pipeline._pred_path(items_dir, name, a.condition, a.repr,
                                   prompt_style=a.prompt_style or "neutral",
                                   replicate=a.replicate).stem
        cells[stem] = a.limit
    return {"cells": cells}


def write_plan(items_dir: str | pathlib.Path, lines: list[str]) -> pathlib.Path:
    """Merge the cells of `lines` into <items>/plan.json (a relaunch with MODELS=<one
    model> must not drop the other models' cells)."""
    path = pathlib.Path(items_dir) / "plan.json"
    cells = json.loads(path.read_text())["cells"] if path.exists() else {}
    cells.update(plan_from_commands(items_dir, lines)["cells"])
    path.write_text(json.dumps({"cells": cells}, indent=2, sort_keys=True) + "\n")
    return path


def check(items_dir: str | pathlib.Path) -> list[str]:
    """One line per prediction file with unrecovered errors, and one per planned cell
    with missing rows. Flag: missing rows, or errors above 2 % of the planned (else
    the present) rows — pre-registration section 8."""
    root = pathlib.Path(items_dir)
    status = {s["file"]: s for s in pipeline.plan_status(root)}
    out = []
    files = sorted((root / "predictions").glob("*.jsonl")) if (root / "predictions").exists() else []
    for p in files:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        err = sum(pipeline._unrecovered(r) for r in rows)
        base = status.get(p.stem, {}).get("planned") or len(rows)
        if rows and err:
            flag = "  > 2 %: REPORT AS SUCH" if err > pipeline.ERROR_FLAG_SHARE * base else ""
            out.append(f"{p.name}: {err}/{len(rows)} unrecovered errors ({err / len(rows):.1%})"
                       f"{flag}")
    for s in status.values():
        if s["missing"]:
            what = "no file" if not s["rows"] else f"{s['rows']}/{s['planned']} rows"
            out.append(f"{s['file']}.jsonl: {what}, {s['missing']} planned rows MISSING")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="plan.json from run commands on stdin")
    w.add_argument("items")
    c = sub.add_parser("check", help="unrecovered errors and missing planned rows")
    c.add_argument("items")
    a = ap.parse_args()
    if a.cmd == "write":
        path = write_plan(a.items, sys.stdin.read().splitlines())
        print(f"{path}: {len(json.loads(path.read_text())['cells'])} planned cells")
    else:
        lines = check(a.items)
        print("\n".join(lines) if lines else "no unrecovered errors, no missing planned rows")


if __name__ == "__main__":
    main()
