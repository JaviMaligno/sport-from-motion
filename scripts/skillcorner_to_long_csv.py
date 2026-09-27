"""SkillCorner Open Data -> long CSV (match_id, frame, track_id, x, y) for `ingest --source long-csv`.

Input: the `data/matches/<id>/` folders of github.com/SkillCorner/opendata
(`<id>_tracking_extrapolated.jsonl`, 10 fps, metres centred on the pitch).
The ball is dropped. By default every player row is kept, including the
positions SkillCorner extrapolates for players off camera; `--detected-only`
keeps only detected positions (partial rosters, closer to what video extraction
gives, and without the extrapolation's own smoothing signature).

    python scripts/skillcorner_to_long_csv.py <opendata>/data/matches data/raw/skillcorner.csv
    motion-sport ingest --source long-csv --input data/raw/skillcorner.csv --out data/clips \
        --sport soccer --fps 10 --name skillcorner
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("matches_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--detected-only", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "frame", "track_id", "x", "y"])
        for match in sorted(p for p in pathlib.Path(a.matches_dir).iterdir() if p.is_dir()):
            src = match / f"{match.name}_tracking_extrapolated.jsonl"
            if not src.exists():
                continue
            with src.open() as f:
                for line in f:
                    r = json.loads(line)
                    for p in r.get("player_data") or []:
                        if not p or p.get("x") is None:
                            continue
                        if a.detected_only and not p.get("is_detected"):
                            continue
                        w.writerow([match.name, r["frame"], p["player_id"], p["x"], p["y"]])
                        rows += 1
    print(f"wrote {rows} rows to {out}")


if __name__ == "__main__":
    main()
