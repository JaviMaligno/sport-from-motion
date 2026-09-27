"""Point-light rendering: grey dots on a white canvas, nothing else.

No axes, no field outline, no colours per team, square canvas (the aspect ratio of
the canvas must not echo the aspect ratio of the pitch). This is the Johansson
point-light display applied to a whole team.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

BG = (255, 255, 255)
DOT = (70, 70, 70)
TRAIL_NEW, TRAIL_OLD = 110, 215  # grey level of the newest / oldest trail segment


def _to_px(points: np.ndarray, extent: float, size: int) -> np.ndarray:
    # y is flipped so "up" is positive y, as in a top-view pitch plot
    px = (points[:, 0] / extent + 1) * 0.5 * size
    py = (1 - (points[:, 1] / extent + 1) * 0.5) * size
    return np.stack([px, py], axis=1)


def render_points(points: np.ndarray, *, extent: float, size: int = 384,
                  radius: float | None = None, history: list[np.ndarray] | None = None) -> Image.Image:
    """Draw one frame. `history` (oldest first) adds fading trails behind each dot."""
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    r = radius or max(2.0, size / 110)
    if history:
        seq = [_to_px(h, extent, size) for h in history] + [_to_px(points, extent, size)]
        for k in range(1, len(seq)):
            # older segments are lighter: the direction of time is visible in one image
            shade = int(TRAIL_NEW + (TRAIL_OLD - TRAIL_NEW) * (1 - k / (len(seq) - 1)))
            for a, b in zip(seq[k - 1], seq[k]):
                d.line([tuple(a), tuple(b)], fill=(shade, shade, shade), width=max(1, int(r / 2)))
    for x, y in _to_px(points, extent, size):
        d.ellipse([x - r, y - r, x + r, y + r], fill=DOT)
    return img


def contact_sheet(frames: list[Image.Image], cols: int = 4, label: bool = True,
                  gap: int = 6) -> Image.Image:
    """Tile frames left-to-right, top-to-bottom, with a small panel number."""
    w, h = frames[0].size
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (200, 200, 200))
    for i, f in enumerate(frames):
        x, y = (i % cols) * (w + gap), (i // cols) * (h + gap)
        sheet.paste(f, (x, y))
        if label:
            ImageDraw.Draw(sheet).text((x + 6, y + 4), str(i + 1), fill=(0, 0, 0))
    return sheet


def save_gif(frames: list[Image.Image], path: str | pathlib.Path, fps: float) -> pathlib.Path:
    path = pathlib.Path(path)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path


def ffmpeg_exe() -> str:
    """FFMPEG_BIN, else imageio-ffmpeg's bundled binary, else an ffmpeg on PATH."""
    if os.environ.get("FFMPEG_BIN"):
        return os.environ["FFMPEG_BIN"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 — not installed, or no binary for this platform
        pass
    for exe in (shutil.which("ffmpeg"), "/usr/local/bin/ffmpeg"):
        if exe and pathlib.Path(exe).exists():
            return exe
    raise RuntimeError("no ffmpeg found: pip install imageio-ffmpeg (or .[video]) or set FFMPEG_BIN")


def save_mp4(frames: list[Image.Image], path: str | pathlib.Path, fps: float) -> pathlib.Path:
    """H.264 mp4 of the frames at `fps`, one video frame per frame (for video models)."""
    path = pathlib.Path(path)
    w, h = frames[0].size
    raw = b"".join(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes() for f in frames)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
           # yuv420p needs even sides; pad with the background colour if they are odd
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", f"{fps:g}",
           "-movflags", "+faststart", "-an", str(path)]
    res = subprocess.run(cmd, input=raw, capture_output=True, timeout=120)
    if res.returncode:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.decode(errors='replace')[:500]}")
    return path


def auto_extent(xy: np.ndarray, margin: float = 1.1) -> float:
    return float(np.nanmax(np.abs(xy))) * margin
