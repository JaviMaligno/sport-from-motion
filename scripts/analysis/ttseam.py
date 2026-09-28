"""Continuity check at the 900 s seam between F_20200220_1 and F_20220220_1 (D10).

    .venv/bin/python scripts/analysis/ttseam.py [data/raw/teamtrack/soccer.csv]

The three segments it reads (0840-0870, 0870-0900, 0900-0930) have no missing (0, 0)
point, so the CSV rebuilt with them as NaN (data/raw/teamtrack_fix0, D18) gives the
same numbers.
"""
import csv, collections, sys
import numpy as np
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/raw/teamtrack/soccer.csv"
rows = collections.defaultdict(dict)  # segment -> frame -> {track: (x,y)}
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        seg = r["segment"]
        if "0870_0900" in seg or "0900_0930" in seg or "0840_0870" in seg:
            rows[seg].setdefault(int(r["frame"]), {})[int(r["track_id"])] = (float(r["x"]), float(r["y"]))
for seg in sorted(rows):
    fr = rows[seg]
    print(seg, "frames", min(fr), max(fr), "players", len(fr[min(fr)]))


def step(a, b):
    # best-matching distance between two player sets (greedy nearest)
    A = np.array(list(a.values())); B = np.array(list(b.values()))
    d = np.linalg.norm(A[:, None] - B[None], axis=2)
    return np.median(d.min(axis=1)), np.max(d.min(axis=1))


segs = sorted(rows)
by = {s.split("-", 1)[1]: s for s in segs}
prev = [s for s in segs if "0870_0900" in s][0]
nxt = [s for s in segs if "0900_0930" in s][0]
base = [s for s in segs if "0840_0870" in s][0]
pf, nf, bf = rows[prev], rows[nxt], rows[base]
print("within-recording seam 0840-0870 -> 0870-0900: median/max nearest dist (m):",
      step(bf[max(bf)], pf[min(pf)]))
print("cross-id seam 0870-0900 (20200220) -> 0900-0930 (20220220):",
      step(pf[max(pf)], nf[min(nf)]))
print("consecutive frames inside a segment:", step(pf[100], pf[101]))
# same track index, same position?
same = [np.hypot(*(np.array(pf[max(pf)][k]) - np.array(nf[min(nf)][k]))) for k in pf[max(pf)] if k in nf[min(nf)]]
print("same track index across seam: median dist", np.median(same), "n", len(same))
