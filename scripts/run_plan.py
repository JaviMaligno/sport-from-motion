"""The run plan of scripts/final_run.sh, and the checks against it.

    # plan.json from the launcher's own command lines (one `motion-sport run` per line)
    bash scripts/final_run.sh ... | python scripts/run_plan.py write runs/final
    # error summary: unrecovered errors per file, and planned cells with missing rows
    python scripts/run_plan.py check runs/final
    # the items must be the pre-registered set (final_run.sh refuses to start otherwise)
    python scripts/run_plan.py preflight runs/final 400 "<models>"

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


# pre-registered item set (docs/preregistration.md section 3 and deviations D1-D2, 2026-09-28)
PREFLIGHT_CONTROLS = {"player_mode": "random", "n_players": 10}


def preflight(items_dir: str | pathlib.Path, n: int, models: list[str]) -> list[str]:
    """-> the reasons the items are not the pre-registered set ([] = ok)."""
    root = pathlib.Path(items_dir)
    cfg = json.loads((root / "config.json").read_text())
    bad = []
    if cfg.get("preset") != "strict_smooth":
        bad.append(f"preset is {cfg.get('preset')!r}, pre-registered strict_smooth")
    controls = cfg.get("controls") or cfg.get("control_config") or {}
    for key, want in PREFLIGHT_CONTROLS.items():
        if controls.get(key) != want:
            bad.append(f"controls.{key} is {controls.get(key)!r}, pre-registered {want!r}")
    if not controls.get("max_speed_ms"):
        bad.append(f"controls.max_speed_ms is {controls.get('max_speed_ms')!r}: the teleport "
                   "ceiling must be set (D2)")
    per = n // len(cfg["candidates"])
    short = {s: k for s, k in cfg["clips_kept_by_sport"].items() if k < per}
    if short:
        bad.append(f"sports with fewer than {per} clips: {short}")
    items = [json.loads(line) for line in (root / "items.jsonl").read_text().splitlines()
             if line.strip()]
    if not all("prompt_informed" in i for i in items):
        bad.append("items without prompt_informed: re-run prepare")
    reprs = {i["repr"] for i in items}
    need = {"sheet", "text", "trails"} | ({"video"} if any(m.startswith("vertex:") for m in models)
                                          else set())
    if need - reprs:
        bad.append(f"missing representations {sorted(need - reprs)}")
    af = {i["clip_id"] for i in items if i["sport"] == "american_football"}
    aligned = {i["clip_id"] for i in items
               if i["sport"] == "american_football" and "random_phase" not in i.get("tags", [])}
    if not af:
        bad.append("no american_football clips")
    elif aligned:
        bad.append(f"{len(aligned)}/{len(af)} american_football clips not tagged random_phase: "
                   "ingest NFL with --nfl-phase random (D1)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="plan.json from run commands on stdin")
    w.add_argument("items")
    c = sub.add_parser("check", help="unrecovered errors and missing planned rows")
    c.add_argument("items")
    f = sub.add_parser("preflight", help="the items must be the pre-registered set")
    f.add_argument("items")
    f.add_argument("n", type=int, help="clips per cell (N of final_run.sh)")
    f.add_argument("models", help="space-separated model ids (MODELS of final_run.sh)")
    a = ap.parse_args()
    if a.cmd == "preflight":
        bad = preflight(a.items, a.n, a.models.split())
        if bad:
            sys.exit("preflight failed:\n  " + "\n  ".join(bad))
        cfg = json.loads((pathlib.Path(a.items) / "config.json").read_text())
        print(f"preflight ok: {a.items} ({cfg['preset']}, {cfg['clips_kept_by_sport']})")
    elif a.cmd == "write":
        path = write_plan(a.items, sys.stdin.read().splitlines())
        print(f"{path}: {len(json.loads(path.read_text())['cells'])} planned cells")
    else:
        lines = check(a.items)
        print("\n".join(lines) if lines else "no unrecovered errors, no missing planned rows")


if __name__ == "__main__":
    main()
