"""The run plan of scripts/final_run.sh, and the checks against it.

    # plan.json from the launcher's own command lines (one `motion-sport run` per line)
    bash scripts/final_run.sh ... | python scripts/run_plan.py write runs/final
    # the A7b directory is exploratory: report prints no primary contrast and no Holm
    bash scripts/final_run.sh ... | python scripts/run_plan.py write runs/final-d8 --exploratory
    # error summary: unrecovered errors per file, and planned cells with missing rows
    python scripts/run_plan.py check runs/final
    # the items must be the pre-registered set (final_run.sh refuses to start otherwise)
    python scripts/run_plan.py preflight runs/final 400 "<models>"
    # the exploratory 8 s cells (A7b, deviation D7): 3 sports, 40 frames, sheet only
    python scripts/run_plan.py preflight-a7b runs/final-d8 300

`<items>/plan.json` is {"cells": {<prediction file stem>: planned rows}} (plus
"exploratory": true for an exploratory directory). The stem is
the file `motion-sport run` writes for that cell (pipeline._pred_path), so a planned
cell that never started shows up as missing, not as silently absent. The planned rows
come from the `--limit` of each command (the first N of the interleaved order). A
command whose `--items` is another directory (the A7b cells on runs/final-d8) belongs to
that directory's plan and is skipped here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import os
import shlex
import sys

from motion_sport import pipeline
from motion_sport.backends.decision import is_decision_model


def plan_from_commands(items_dir: str | pathlib.Path, lines: list[str]) -> dict:
    """`motion-sport run ...` lines -> {"cells": {stem: limit}}. Lines that are not a
    run command (comments, blank), or that run on another items directory, are ignored."""
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
        if a.items and not _same_dir(a.items, items_dir):
            continue
        if a.limit is None:
            raise ValueError(f"run command without --limit, no planned size: {line}")
        name = a.model + ("+json" if is_decision_model(a.model) and a.state_format == "json"
                          else "")
        stem = pipeline._pred_path(items_dir, name, a.condition, a.repr,
                                   prompt_style=a.prompt_style or "neutral",
                                   replicate=a.replicate).stem
        cells[stem] = a.limit
    return {"cells": cells}


def _same_dir(a: str | pathlib.Path, b: str | pathlib.Path) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def write_plan(items_dir: str | pathlib.Path, lines: list[str],
               exploratory: bool = False) -> pathlib.Path:
    """Merge the cells of `lines` into <items>/plan.json (a relaunch with MODELS=<one
    model> must not drop the other models' cells). `exploratory` marks the whole
    directory as such ("exploratory": true, sticky): `report` then has no primary
    contrast and no Holm family (A7b, deviation D7)."""
    path = pathlib.Path(items_dir) / "plan.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    cells = old.get("cells", {})
    cells.update(plan_from_commands(items_dir, lines)["cells"])
    plan = {"cells": cells, **({"exploratory": True}
                               if exploratory or old.get("exploratory") else {})}
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def check(items_dir: str | pathlib.Path) -> list[str]:
    """One line per prediction file with unrecovered errors, and one per planned cell
    with missing rows. Flag: missing rows, or errors above 2 % of the planned (else
    the present) rows; above 50 % of the present rows the cell is a systemic failure
    that report leaves out of every contrast — pre-registration section 8."""
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
            if err > pipeline.SYSTEMIC_FAILURE_SHARE * len(rows):  # same rule as report
                flag = "  > 50 %: SYSTEMIC FAILURE, cell excluded from every contrast"
            out.append(f"{p.name}: {err}/{len(rows)} unrecovered errors ({err / len(rows):.1%})"
                       f"{flag}")
    for s in status.values():
        if s["missing"]:
            what = "no file" if not s["rows"] else f"{s['rows']}/{s['planned']} rows"
            out.append(f"{s['file']}.jsonl: {what}, {s['missing']} planned rows MISSING")
    return out


# pre-registered item set (docs/preregistration.md section 3 and deviations D1-D2, 2026-09-28):
# N = 10 random players, the 12 m/s teleport ceiling (D2) and the strict_smooth smoothing
# (sigma = 2 frames), pinned exactly: a preset edit must not slip through
PREFLIGHT_CONTROLS = {"player_mode": "random", "n_players": 10, "max_speed_ms": 12.0,
                      "smooth": 2.0}
N_FRAMES = 20  # 4 s at 5 Hz
CANDIDATES = ["american_football", "basketball", "handball", "soccer"]


def _controls(cfg: dict, pinned: dict) -> list[str]:
    controls = cfg.get("controls") or cfg.get("control_config") or {}
    return [f"controls.{key} is {controls.get(key)!r}, pre-registered {want!r}"
            for key, want in pinned.items()
            # exact: 12 == 12.0 is fine, None / 0 / 15.0 / True are not
            if type(controls.get(key)) is bool or controls.get(key) != want]


def preflight(items_dir: str | pathlib.Path, n: int, models: list[str]) -> list[str]:
    """-> the reasons the items are not the pre-registered set ([] = ok)."""
    root = pathlib.Path(items_dir)
    cfg = json.loads((root / "config.json").read_text())
    bad = []
    if cfg.get("preset") != "strict_smooth":
        bad.append(f"preset is {cfg.get('preset')!r}, pre-registered strict_smooth")
    bad += _controls(cfg, {**PREFLIGHT_CONTROLS, "n_frames": N_FRAMES})
    if sorted(cfg.get("candidates") or []) != CANDIDATES:
        bad.append(f"candidates are {cfg.get('candidates')}, pre-registered {CANDIDATES}")
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
    # every one of the 5 conditions in each planned representation that can show it
    # (pipeline.VALID; e.g. trails has no formation or motion_shuffled)
    cells = {(i.get("condition"), i["repr"]) for i in items}
    missing = sorted((c, r) for r in need & reprs for c, ok in pipeline.VALID.items()
                     if r in ok and (c, r) not in cells)
    if missing:
        bad.append(f"missing cells {missing}")
    af = {i["clip_id"] for i in items if i["sport"] == "american_football"}
    aligned = {i["clip_id"] for i in items
               if i["sport"] == "american_football" and "random_phase" not in i.get("tags", [])}
    if not af:
        bad.append("no american_football clips")
    elif aligned:
        bad.append(f"{len(aligned)}/{len(af)} american_football clips not tagged random_phase: "
                   "ingest NFL with --nfl-phase random (D1)")
    return bad


# exploratory 8 s cells (A7b, docs/preregistration.md deviation D7): same controls as the
# final set, 40 frames (8 s at 5 Hz), 3 sports (no mid-play NFL clip lasts 8 s)
A7B_CANDIDATES = ["basketball", "handball", "soccer"]
A7B_N_FRAMES = 40
A7B_CELLS = {("motion", "sheet"), ("motion_shuffled", "sheet")}


def preflight_a7b(items_dir: str | pathlib.Path, n: int) -> list[str]:
    """-> the reasons the items are not the A7b set ([] = ok)."""
    root = pathlib.Path(items_dir)
    cfg = json.loads((root / "config.json").read_text())
    bad = []
    if cfg.get("preset") != "strict_smooth":
        bad.append(f"preset is {cfg.get('preset')!r}, pre-registered strict_smooth")
    bad += _controls(cfg, {**PREFLIGHT_CONTROLS, "n_frames": A7B_N_FRAMES})
    if sorted(cfg.get("candidates") or []) != A7B_CANDIDATES:
        bad.append(f"candidates are {cfg.get('candidates')}, pre-registered {A7B_CANDIDATES}")
    per = n // len(A7B_CANDIDATES)
    short = {s: k for s, k in cfg.get("clips_kept_by_sport", {}).items() if k < per}
    if short:
        bad.append(f"sports with fewer than {per} clips: {short}")
    items = [json.loads(line) for line in (root / "items.jsonl").read_text().splitlines()
             if line.strip()]
    missing = A7B_CELLS - {(i["condition"], i["repr"]) for i in items}
    if missing:
        bad.append(f"missing cells {sorted(missing)}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="plan.json from run commands on stdin")
    w.add_argument("items")
    w.add_argument("--exploratory", action="store_true",
                   help="mark the directory exploratory (report: no primary contrast, no Holm)")
    c = sub.add_parser("check", help="unrecovered errors and missing planned rows")
    c.add_argument("items")
    f = sub.add_parser("preflight", help="the items must be the pre-registered set")
    f.add_argument("items")
    f.add_argument("n", type=int, help="clips per cell (N of final_run.sh)")
    f.add_argument("models", help="space-separated model ids (MODELS of final_run.sh)")
    f8 = sub.add_parser("preflight-a7b", help="the 8 s items must be the A7b set (D7)")
    f8.add_argument("items")
    f8.add_argument("n", type=int, help="clips per cell (ND8 of final_run.sh)")
    a = ap.parse_args()
    if a.cmd in ("preflight", "preflight-a7b"):
        bad = (preflight(a.items, a.n, a.models.split()) if a.cmd == "preflight"
               else preflight_a7b(a.items, a.n))
        if bad:
            sys.exit("preflight failed:\n  " + "\n  ".join(bad))
        cfg = json.loads((pathlib.Path(a.items) / "config.json").read_text())
        print(f"preflight ok: {a.items} ({cfg['preset']}, {cfg['clips_kept_by_sport']})")
    elif a.cmd == "write":
        path = write_plan(a.items, sys.stdin.read().splitlines(), a.exploratory)
        plan = json.loads(path.read_text())
        print(f"{path}: {len(plan['cells'])} planned cells"
              + (" (exploratory)" if plan.get("exploratory") else ""))
    else:
        lines = check(a.items)
        print("\n".join(lines) if lines else "no unrecovered errors, no missing planned rows")


if __name__ == "__main__":
    main()
