"""Text serialisation of a View: the same information as the images, as numbers.

Part 2 of wheres-the-ball showed frontier models doing *worse* with raw
coordinates than with pixels. Keeping both representations on identical items
lets that comparison be repeated here.
"""
from __future__ import annotations

from motion_sport.conditions import View


def view_to_text(view: View, scale: float = 100.0) -> str:
    """Integer coordinates (x100): short numbers tokenise into few tokens, which
    matters for Laya's 512-8,192-token context, at a resolution (0.01 of the spread)
    well below the per-snapshot displacements."""
    lines = []
    for k, (pts, t) in enumerate(zip(view.frames, view.frame_times)):
        # For shuffled views the time stamp is withheld: it would give the order away.
        head = f"snapshot {k + 1}" + (f" (t={t:.1f}s)" if view.ordered else "")
        coords = " ".join(f"({round(x * scale)},{round(y * scale)})" for x, y in pts)
        lines.append(f"{head}: {coords}")
    note = "Coordinates are (x,y) in arbitrary units." + (
        " Players are listed in the same order in every snapshot."
        if view.condition != "formation" else "")
    return "\n".join([note, *lines])


def view_to_state(view: View, scale: float = 100.0) -> dict:
    """The same content as JSON, for decision models that accept a JSON state."""
    snaps = []
    for pts, t in zip(view.frames, view.frame_times):
        snap = {"players": [[round(x * scale), round(y * scale)] for x, y in pts]}
        if view.ordered:
            snap["t"] = round(t, 1)
        snaps.append(snap)
    return {"snapshots": snaps}
