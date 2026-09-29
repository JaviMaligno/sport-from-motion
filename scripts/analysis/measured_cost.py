"""Measured cost and wall-clock of a model run, priced with the list-price table of
scripts/estimate_run.py (docs/preregistration.md, annex B). Nothing is sent to any API.

Token usage is read from the `usage` field of every prediction row (chat models) or
`input_tokens` (Jev), with the same field rules as `estimate_run.tokens`, except that
rows with an unrecovered error that still carry a usage (e.g. an answer with no JSON)
are counted too: those calls were billed.

What the files cannot show: when `run` retries an item (pass 2 of final_run.sh), the
first attempt's row is dropped from the file, and its usage with it. Those attempts are
counted from the per-model logs (the `[k/N] ... pred=<error>` lines of the pass that
failed) and priced, as an approximation, at the mean usage of the same cell's rows
(`lost_attempts`, `lost_usd_approx`). API exceptions carry no usage and are likely not
billed, so that figure is an upper bound.

    PYTHONPATH=src .venv/bin/python scripts/analysis/measured_cost.py \
        --dir final=runs/final --dir a7b=runs/final-d8 \
        --dir sens=runs/final-sens --only-changed sens=runs/final \
        --logs runs/final/logs --json runs/analysis-final/cost.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from estimate_run import PRICES  # noqa: E402  (list-price assumptions, USD / M tokens)

SPECIALIST_PREFIXES = ("baseline:", "minirocket", "deepsets", "probe:")


def price_key(model: str) -> str:
    """Jev in JSON state format is the same model at the same price."""
    return model.removesuffix("+json")


def usage_tokens(row: dict) -> tuple[int, int, int] | None:
    """(input, output, of which thinking/reasoning) of one row; None without a usage."""
    u = row.get("usage") or {}
    if "prompt_tokens" in u:  # OpenAI / Azure OpenAI: completion includes reasoning
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        return int(u["prompt_tokens"]), int(u.get("completion_tokens") or 0), int(rt)
    if "input_tokens" in u:  # Anthropic: output includes thinking
        inp = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens"))
        th = (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
        return inp, int(u.get("output_tokens") or 0), int(th)
    if "promptTokenCount" in u:  # Gemini: thinking billed as output
        th = int(u.get("thoughtsTokenCount") or 0)
        return int(u["promptTokenCount"]), int(u.get("candidatesTokenCount") or 0) + th, th
    if row.get("input_tokens"):  # Jev: input only
        return int(row["input_tokens"]), 0, 0
    return None


def usd(model: str, inp: float, out: float) -> float:
    pin, pout = PRICES[price_key(model)]
    return (inp * pin + out * pout) / 1e6


def read_rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lost_attempts(log_dir: pathlib.Path) -> dict[str, int]:
    """prediction-file stem -> failed attempts that a later pass retried (their rows are
    no longer in the file). Parsed from `+ ... motion-sport run ...` headers and the
    `[k/N] clip true=.. pred=..` lines of each pass."""
    out: dict[str, int] = collections.Counter()
    for log in sorted(log_dir.glob("*.log")):
        cell, passno = None, 0
        errs: dict[tuple, set] = collections.defaultdict(set)   # (stem) -> clips failed so far
        for line in log.read_text().splitlines():
            if line.startswith("=== pass"):
                passno = int(line.split()[-1])
                continue
            if line.startswith("+ ") and " run " in line:
                a = line.split()
                arg = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
                model = arg["--model"] + ("+json" if arg.get("--state-format") == "json" else "")
                safe = model.replace(":", "__").replace("/", "_")
                style = arg.get("--prompt-style", "neutral")
                rep = int(arg.get("--replicate", 1))
                stem = (f"{pathlib.Path(arg['--items']).name}/{safe}__{arg['--condition']}__"
                        f"{arg['--repr']}" + ("" if style == "neutral" else f"__{style}")
                        + ("" if rep == 1 else f"__r{rep}"))
                cell = stem
                continue
            m = re.match(r"\[\d+/\d+\] (\S+) true=\S+ pred=(.*)$", line)
            if m and cell:
                clip, pred = m.groups()
                if clip in errs[cell]:  # a retry of an earlier failed attempt: that one is lost
                    out[cell] += 1
                    errs[cell].discard(clip)
                if not re.fullmatch(r"[a-z_]+", pred):  # an error string, not a label
                    errs[cell].add(clip)
    return dict(out)


def wall_clock(log_dir: pathlib.Path) -> dict[str, dict]:
    """Per model log: first write (birth time on macOS) and last modification."""
    out = {}
    for log in sorted(log_dir.glob("*.log")):
        st = os.stat(log)
        start = getattr(st, "st_birthtime", st.st_ctime)
        out[log.stem] = {"start": dt.datetime.fromtimestamp(start).isoformat(timespec="seconds"),
                         "end": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                         "hours": round((st.st_mtime - start) / 3600, 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="label=path of an items dir")
    ap.add_argument("--only-changed", action="append", default=[],
                    help="label=reference dir: count only rows that differ from the reference "
                         "(a sensitivity copy re-running a few items)")
    ap.add_argument("--logs", help="log dir of the launcher (lost attempts, wall-clock)")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = dict(s.split("=", 1) for s in a.only_changed)
    lost = lost_attempts(pathlib.Path(a.logs)) if a.logs else {}
    cells, by_model = [], collections.defaultdict(lambda: collections.Counter())
    for spec in a.dir:
        label, path = spec.split("=", 1)
        root = pathlib.Path(path)
        for p in sorted((root / "predictions").glob("*.jsonl")):
            rows = read_rows(p)
            if not rows or rows[0].get("model", "").startswith(SPECIALIST_PREFIXES):
                continue
            model = rows[0]["model"]
            if label in ref:
                rp = pathlib.Path(ref[label]) / "predictions" / p.name
                old = {json.dumps(r, sort_keys=True) for r in read_rows(rp)} if rp.exists() else set()
                rows = [r for r in rows if json.dumps(r, sort_keys=True) not in old]
                if not rows:
                    continue
            toks = [t for t in map(usage_tokens, rows) if t]
            inp, out, th = (sum(t[i] for t in toks) for i in range(3))
            c = {"set": label, "file": p.stem, "model": model, "rows": len(rows),
                 "rows_with_usage": len(toks),
                 "errors": sum(bool(r.get("error")) and not r.get("label") for r in rows),
                 "input_tokens": inp, "output_tokens": out, "thinking_tokens": th,
                 "usd": usd(model, inp, out)}
            n_lost = lost.get(f"{root.name}/{p.stem}", 0) if label not in ref else 0
            if n_lost and toks:
                c["lost_attempts"] = n_lost
                c["lost_usd_approx"] = n_lost * c["usd"] / len(toks)
            cells.append(c)
            k = by_model[(label, model)]
            for f in ("rows", "rows_with_usage", "errors", "input_tokens", "output_tokens",
                      "thinking_tokens", "usd", "lost_attempts", "lost_usd_approx"):
                k[f] += c.get(f, 0)
    summary = [{"set": s, "model": m, **dict(v)} for (s, m), v in sorted(by_model.items())]
    # A sensitivity copy (--only-changed) is not part of the run the estimate covers:
    # the run total is every other set, and each set also gets its own subtotal.
    by_set = collections.Counter()
    for v in summary:
        by_set[v["set"]] += v["usd"]
    run_total = sum(u for s, u in by_set.items() if s not in ref)
    total = sum(by_set.values())
    total_lost = sum(v.get("lost_usd_approx", 0) for v in summary)
    print(f"{'set':6} {'model':42} {'rows':>6} {'in Mtok':>8} {'out Mtok':>8} {'think':>7} "
          f"{'USD':>8} {'lost':>5} {'~USD':>6}")
    for v in summary:
        print(f"{v['set']:6} {v['model']:42} {v['rows']:6d} {v['input_tokens'] / 1e6:8.2f} "
              f"{v['output_tokens'] / 1e6:8.2f} {v['thinking_tokens'] / 1e6:7.2f} {v['usd']:8.2f} "
              f"{v.get('lost_attempts', 0):5d} {v.get('lost_usd_approx', 0):6.2f}")
    for s_, u in sorted(by_set.items()):
        print(f"SUBTOTAL {s_}: USD {u:.2f}" + (" (sensitivity, not in the run)" if s_ in ref else ""))
    print(f"RUN TOTAL USD {run_total:.2f} (sets: {', '.join(s_ for s_ in sorted(by_set) if s_ not in ref)}; "
          f"+ ~{total_lost:.2f} for retried attempts not in the files)")
    if ref:
        print(f"ALL SETS incl. sensitivity USD {total:.2f}")
    res = {"prices_usd_per_mtok": PRICES, "cells": cells, "by_model": summary,
           "by_set_usd": dict(by_set), "run_total_usd": run_total,
           "sensitivity_sets": sorted(ref), "total_usd_incl_sensitivity": total,
           "lost_usd_approx": total_lost}
    if a.logs:
        res["wall_clock"] = wall_clock(pathlib.Path(a.logs))
        for k, v in res["wall_clock"].items():
            print(f"wall-clock {k}: {v['start']} -> {v['end']} ({v['hours']} h)")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
