"""EIGD-H (Handball-Bundesliga 2019/20, Kinexon LPS) positions -> long CSV.

Source: https://data.uni-hannover.de/dataset/eigd (`eigd-h_pos.zip`, CC BY-NC-SA 4.0;
cite Biermann et al. 2021 and credit the Handball-Bundesliga). Each `<match>_<hh-mm-ss>.h5`
is one continuous 5 min sequence with `team_a` / `team_b` arrays [frames, players, 2] in
metres on the 40 x 20 m court, resampled to the 30 fps video. The ball is dropped.

    python scripts/eigd_to_long_csv.py data/raw/eigd_h data/raw/eigd_h/handball.csv
    motion-sport ingest --source long-csv --input data/raw/eigd_h/handball.csv --out data/clips \
        --sport handball --fps 30 --name eigd --group-extra segment
"""
from __future__ import annotations

import csv
import pathlib
import sys

import h5py
import numpy as np


def main() -> None:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    rows = 0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "segment", "frame", "track_id", "x", "y"])
        for f in sorted(src.glob("*.h5")):
            match, seg = f.stem.split("_", 1)
            with h5py.File(f) as h:
                xy = np.concatenate([h["team_a"][:], h["team_b"][:]], axis=1)
            for t in range(xy.shape[0]):
                for j in range(xy.shape[1]):
                    x, y = xy[t, j]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    w.writerow([match, seg, t, j, round(float(x), 3), round(float(y), 3)])
                    rows += 1
    print(f"wrote {rows} rows to {out}")


if __name__ == "__main__":
    main()
