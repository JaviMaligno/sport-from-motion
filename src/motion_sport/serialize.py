"""Text serialisation of a View: the same information as the images, as numbers.

Part 2 of wheres-the-ball showed frontier models doing *worse* with raw
coordinates than with pixels. Keeping both representations on identical items
lets that comparison be repeated here.
"""
from __future__ import annotations

from motion_sport.conditions import View


def view_to_text(view: View, decimals: int = 2) -> str:
    lines = []
    for k, (pts, t) in enumerate(zip(view.frames, view.frame_times)):
        # For shuffled views the time stamp is withheld: it would give the order away.
        head = f"snapshot {k + 1}" + (f" (t={t:.1f}s)" if view.ordered else "")
        coords = " ".join(f"({x:.{decimals}f},{y:.{decimals}f})" for x, y in pts)
        lines.append(f"{head}: {coords}")
    note = ("Players are listed in the same order in every snapshot."
            if view.condition != "formation" else "")
    return "\n".join(filter(None, [note, *lines]))
