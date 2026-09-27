"""Cost / time estimate of the final model run (docs/preregistration.md, scripts/final_run.sh).

Reads the token usage the pilots actually recorded (one `usage` per prediction row)
and prices the planned cells. Nothing is sent to any API.

    .venv/bin/python scripts/estimate_run.py              # per model
    .venv/bin/python scripts/estimate_run.py --detail     # per cell, with the usage source

Sources, in order: runs/pilot4-strict (4 sports, the prompt of the final run), then
runs/pilot-strict (pilot 1, 2 sports: the only Gemini 3.1 usage), then a proxy (the
closest measured model / representation, always named in the output).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

# ---------------------------------------------------------------------------
# PRICES — LIST-PRICE ASSUMPTIONS, USD per million tokens (input, output).
# Not invoices: discounts, caching, batch pricing and regional surcharges are ignored.
# Gemini output includes its thinking tokens; gpt-5.6 completion tokens include
# reasoning. Jev (OpenRouter) is billed on input only.
# ---------------------------------------------------------------------------
PRICES: dict[str, tuple[float, float]] = {
    "azure-openai:gpt-5.6-sol": (2.5, 15.0),
    "azure-openai:gpt-5.6-terra": (0.75, 4.5),
    "vertex-anthropic:claude-sonnet-5": (3.0, 15.0),
    "vertex-anthropic:claude-opus-5-5": (5.0, 25.0),
    "vertex:gemini-3.1-pro-preview": (2.0, 12.0),
    "jev-openrouter:~typesafe/jev-latest": (0.042, 0.0),
}

# Models with no measured usage: price their own tokens with this model's usage.
PROXY_MODEL = {"vertex-anthropic:claude-opus-5-5": "vertex-anthropic:claude-sonnet-5"}
# ASSUMPTION: video tokens per frame for Gemini (the documented default-resolution
# figure of Gemini 2.x; Gemini 3 at low/medium resolution uses fewer). A 4 s clip at
# 5 fps is 20 frames, sent with videoMetadata.fps = 5.
VIDEO_TOKENS_PER_FRAME = 258
VIDEO_FRAMES = 20
PILOT_WORKERS = 6        # workers the pilots ran with (pilot_models.sh default)
CHARS_PER_TOKEN = 4.0    # only for the extra text of the informed prompt

# ---------------------------------------------------------------------------
# The plan. Keep in sync with scripts/final_run.sh (same cells, same order).
# (condition, repr, prompt_style, replicate, n_clips)
# ---------------------------------------------------------------------------
N, N_REP = 400, 200
CHAT_CELLS = [
    ("motion", "sheet", "neutral", 1, N), ("motion_shuffled", "sheet", "neutral", 1, N),
    ("formation", "sheet", "neutral", 1, N), ("motion", "text", "neutral", 1, N),
    ("motion", "sheet", "informed", 1, N), ("kinematics", "sheet", "neutral", 1, N),
    ("kinematics_solo", "sheet", "neutral", 1, N), ("motion", "trails", "neutral", 1, N),
]
VIDEO_CELLS = [("motion", "video", "neutral", 1, N), ("motion_shuffled", "video", "neutral", 1, N)]
REPLICATE_CELLS = [(c, "sheet", "neutral", k, N_REP) for k in (2, 3)
                   for c in ("motion", "motion_shuffled")]
JEV_CELLS = [
    ("motion", "text", "neutral", 1, N), ("motion_shuffled", "text", "neutral", 1, N),
    ("formation", "text", "neutral", 1, N), ("motion", "text", "informed", 1, N),
    ("kinematics", "text", "neutral", 1, N), ("kinematics_solo", "text", "neutral", 1, N),
]
FINAL_WORKERS = {"vertex:gemini-3.1-pro-preview": 12}  # others: 6


def plan() -> dict[str, list[tuple]]:
    out = {}
    for m in PRICES:
        if m.startswith("jev"):
            out[m] = [(*c, fmt) for c in JEV_CELLS for fmt in ("text", "json")]
            continue
        cells = CHAT_CELLS + (VIDEO_CELLS if m.startswith("vertex:") else []) + REPLICATE_CELLS
        out[m] = [(*c, None) for c in cells]
    return out


def tokens(row: dict) -> tuple[int, int] | None:
    """(input, output) tokens of one prediction row, whatever the provider's field names."""
    if row.get("error") and not row.get("label"):
        return None
    u = row.get("usage") or {}
    if "prompt_tokens" in u:                                   # OpenAI / Azure OpenAI
        return int(u["prompt_tokens"]), int(u.get("completion_tokens") or 0)
    if "input_tokens" in u:                                    # Anthropic
        inp = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens"))
        return inp, int(u.get("output_tokens") or 0)
    if "promptTokenCount" in u:                                # Gemini (thinking billed as output)
        return (int(u["promptTokenCount"]),
                int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0))
    if row.get("input_tokens"):                                # Jev: input only
        return int(row["input_tokens"]), 0
    return None


def text_tokens(row: dict) -> int | None:
    """Gemini: the text part of the prompt (to rebuild a video prompt from a sheet one)."""
    for d in (row.get("usage") or {}).get("promptTokensDetails") or []:
        if d.get("modality") == "TEXT":
            return int(d["tokenCount"])
    return None


def load_measured(dirs: list[pathlib.Path]) -> dict:
    """(model, condition, repr) -> {in, out, text_in, sec_per_call, n, source}; first dir wins."""
    out: dict = {}
    for d in dirs:
        for p in sorted((d / "predictions").glob("*.jsonl")):
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            if not rows or rows[0].get("prompt_style", "neutral") != "neutral" \
                    or rows[0].get("replicate", 1) != 1:
                continue
            key = (rows[0].get("model"), rows[0].get("condition"), rows[0].get("repr"))
            if key in out:
                continue
            toks = [t for t in map(tokens, rows) if t]
            if not toks:
                continue
            st = os.stat(p)
            span = st.st_mtime - getattr(st, "st_birthtime", st.st_ctime)
            txt = [t for t in map(text_tokens, rows) if t]
            out[key] = {"in": statistics.mean(t[0] for t in toks),
                        "out": statistics.mean(t[1] for t in toks),
                        "text_in": statistics.mean(txt) if txt else None,
                        # file span x workers / rows: seconds of one call on one worker
                        "sec_per_call": span * PILOT_WORKERS / len(rows) if span > 0 else None,
                        "n": len(toks), "source": d.name}
    return out


def informed_extra_tokens(candidates: list[str]) -> float:
    from motion_sport.prompts import MOVEMENT
    return sum(len(MOVEMENT[c]) + len(c) + 4 for c in candidates) / CHARS_PER_TOKEN


def estimate_cell(model: str, cond: str, rep: str, style: str, fmt, measured: dict,
                  extra_informed: float) -> tuple[dict, str]:
    """Per-call usage of one planned cell and a note saying where it came from."""
    notes = []
    src_model = PROXY_MODEL.get(model, model)
    if src_model != model:
        notes.append(f"proxy {src_model.split(':')[-1]}")
    name = src_model + ("+json" if fmt == "json" else "")
    look_rep = "sheet" if rep in ("trails", "video") else rep
    if rep == "trails":
        notes.append("trails<-sheet (input overstated: 1 image vs a 4x2 sheet)")
    m = measured.get((name, cond, look_rep))
    if m is None:  # same model and repr, any measured condition
        alt = [v for (mm, c, r), v in measured.items() if mm == name and r == look_rep]
        if not alt:
            raise SystemExit(f"no usage for {name} {look_rep}: run the pilots first")
        m = {k: statistics.mean(a[k] for a in alt if a[k] is not None)
             if any(a[k] is not None for a in alt) else None
             for k in ("in", "out", "text_in", "sec_per_call")} | {"source": alt[0]["source"]}
        notes.append(f"{cond} not measured: mean of {len(alt)} {look_rep} cells")
    u = dict(m)
    if rep == "video":
        text_in = u["text_in"] if u.get("text_in") else u["in"] * 0.2
        u["in"] = text_in + VIDEO_TOKENS_PER_FRAME * VIDEO_FRAMES
        notes.append(f"video<-sheet text + {VIDEO_FRAMES}x{VIDEO_TOKENS_PER_FRAME} tok/frame")
    if style == "informed":
        u["in"] += extra_informed
        notes.append(f"informed: +{extra_informed:.0f} input tok")
    if u["source"] != "pilot4-strict":
        notes.append(f"from {u['source']}")
    return u, "; ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--detail", action="store_true", help="one line per planned cell")
    a = ap.parse_args()
    root = pathlib.Path(a.runs)
    measured = load_measured([root / "pilot4-strict", root / "pilot-strict"])
    cands = json.loads((root / "pilot4-strict" / "config.json").read_text())["candidates"]
    extra = informed_extra_tokens(cands)

    print("LIST-PRICE ASSUMPTIONS (USD / M tokens, in / out): "
          + ", ".join(f"{m.split(':', 1)[1]} {p[0]:g}/{p[1]:g}" for m, p in PRICES.items()))
    print(f"usage measured in {root}/pilot4-strict (fallback {root}/pilot-strict); time = "
          f"pilot seconds/call x calls / workers, assuming pilots ran with {PILOT_WORKERS} "
          "workers and throughput scales with workers (rate limits may not)\n")
    tot_usd, tot_calls, walls = 0.0, 0, []
    hdr = f"{'model':38} {'calls':>6} {'in Mtok':>8} {'out Mtok':>8} {'USD':>8} {'hours':>6}  notes"
    print(hdr)
    print("-" * len(hdr))
    for model, cells in plan().items():
        pin, pout = PRICES[model]
        workers = FINAL_WORKERS.get(model, 6)
        calls = tin = tout = secs = 0.0
        notes, lines = set(), []
        for cond, rep, style, k, n, fmt in cells:
            u, note = estimate_cell(model, cond, rep, style, fmt, measured, extra)
            ci, co = n * u["in"], n * u["out"]
            usd = (ci * pin + co * pout) / 1e6
            calls += n
            tin += ci
            tout += co
            secs += n * (u["sec_per_call"] or 0) / workers
            for part in note.split("; "):
                if part and not part.startswith("informed") and "not measured" not in part:
                    notes.add(part)
            cell = f"{cond}/{rep}" + ("" if style == "neutral" else f"[{style}]") + \
                ("" if k == 1 else f" r{k}") + (f" ({fmt})" if fmt else "")
            lines.append(f"    {cell:38} {n:6d} {ci / 1e6:8.3f} {co / 1e6:8.3f} {usd:8.2f}"
                         f"  in {u['in']:.0f} out {u['out']:.0f} tok/call"
                         + (f"  [{note}]" if note else ""))
        usd = (tin * pin + tout * pout) / 1e6
        tot_usd += usd
        tot_calls += calls
        walls.append(secs / 3600)
        print(f"{model:38} {int(calls):6d} {tin / 1e6:8.2f} {tout / 1e6:8.2f} {usd:8.2f} "
              f"{secs / 3600:6.1f}  {'; '.join(sorted(notes))}")
        if a.detail:
            print("\n".join(lines))
    print("-" * len(hdr))
    print(f"{'TOTAL':38} {int(tot_calls):6d} {'':8} {'':8} {tot_usd:8.2f} {max(walls):6.1f}  "
          "(hours = slowest model; models run in parallel)")


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    main()
