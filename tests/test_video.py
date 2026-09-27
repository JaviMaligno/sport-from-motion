import json
import subprocess

import numpy as np
import pytest

from motion_sport import pipeline
from motion_sport.backends import chat
from motion_sport.loaders import make_toy_clips
from motion_sport.render import ffmpeg_exe, render_points, save_mp4

try:
    ffmpeg_exe()
    HAVE_FFMPEG = True
except RuntimeError:
    HAVE_FFMPEG = False
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="no ffmpeg (pip install imageio-ffmpeg)")


def _decoded_frames(path, size):
    out = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return len(out) // (size * size * 3)


@needs_ffmpeg
def test_save_mp4_keeps_every_frame(tmp_path):
    pts = np.random.default_rng(0).normal(size=(7, 5, 2))
    frames = [render_points(p, extent=3.0, size=64) for p in pts]
    p = save_mp4(frames, tmp_path / "x.mp4", fps=5)
    data = p.read_bytes()
    assert data[4:8] == b"ftyp"
    assert _decoded_frames(p, 64) == 7


@pytest.fixture()
def video_items(tmp_path):
    for c in make_toy_clips(4, seconds=6):
        c.save(tmp_path / "clips")
    return pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict",
                            conditions=["formation", "motion", "motion_shuffled", "kinematics"],
                            reprs=["sheet", "video"], image_size=64)


@needs_ffmpeg
def test_prepare_writes_full_rate_videos(video_items):
    items = pipeline.load_items(video_items)
    vids = [i for i in items if i["repr"] == "video"]
    assert {i["condition"] for i in vids} == {"motion", "motion_shuffled", "kinematics"}
    for it in vids:
        (f,) = it["files"]
        assert f.endswith(".mp4") and it["video_fps"] == 5.0
        assert "video" in it["prompt"] and "video" in it["prompt_informed"]
        assert len(it["frame_order"]) == 20  # every frame of the clip, not the 8 of the sheet
    m = next(i for i in vids if i["condition"] == "motion")
    assert _decoded_frames(video_items / m["files"][0], 64) == 20
    assert "20 frames over 3.8 s (5 frames per second)" in m["prompt"]
    s = next(i for i in vids if i["condition"] == "motion_shuffled")
    assert s["frame_order"] != sorted(s["frame_order"])
    # the sheet items are unchanged by the video pass: same options, same frames
    sheet = {i["item_id"]: i for i in items if i["repr"] == "sheet"}
    assert all(len(i["frame_order"]) in (1, 8) for i in sheet.values())


@needs_ffmpeg
def test_run_sends_videos_not_images(video_items):
    seen = []

    def fake(model_id, req):
        seen.append(req)
        return chat.complete("dummy:first", req)

    p = pipeline.run_model(str(video_items), "fake:m", condition="motion", rep="video", complete=fake)
    assert seen and all(len(r.videos) == 1 and not r.images and r.video_fps == 5.0 for r in seen)
    assert all(r.videos[0][4:8] == b"ftyp" for r in seen)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert all(r["label"] and not r["error"] for r in rows)


def test_vertex_payload_carries_inline_video(monkeypatch):
    seen = {}

    def fake_post_gcloud(url, payload, post=None):
        seen.update(url=url, payload=payload)
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}

    monkeypatch.setattr(chat, "_post_gcloud", fake_post_gcloud)
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    chat.complete("vertex:gemini-3.1-pro-preview",
                  chat.Request("which sport?", videos=[b"\x00\x00\x00\x18ftypmp42"], video_fps=5.0))
    assert seen["url"].endswith("/publishers/google/models/gemini-3.1-pro-preview:generateContent")
    parts = seen["payload"]["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "video/mp4"
    assert parts[0]["inlineData"]["data"] == "AAAAGGZ0eXBtcDQy"
    assert parts[0]["videoMetadata"] == {"fps": 5.0}
    assert parts[-1] == {"text": "which sport?"}


@pytest.mark.parametrize("model", ["azure-openai:gpt-5.6-sol", "azure-anthropic:claude-opus-5-5",
                                   "vertex-anthropic:claude-sonnet-5", "openai:x", "anthropic:x",
                                   "azure-foundry:x"])
def test_other_routes_refuse_video(model, monkeypatch):
    monkeypatch.setattr(chat, "_post", lambda *a, **k: pytest.fail("must not reach the network"))
    with pytest.raises(chat.BackendError, match="does not accept video"):
        chat.complete(model, chat.Request("q", videos=[b"mp4"]))
