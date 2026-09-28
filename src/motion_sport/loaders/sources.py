"""Adapters for the concrete public sources. Each yields `Clip`s in metres, no ball.

Formats follow the loaders already used in `wheres-the-ball`
(src/wheres_the_ball/data/field_tracking.py) for Metrica and SportVU, so the two
projects read the same files the same way.
"""
from __future__ import annotations

import csv
import json
import pathlib
import zlib
from collections import defaultdict
from collections.abc import Iterator

import numpy as np

from motion_sport.loaders.common import window_stream
from motion_sport.schema import FIELD_SIZE_M, Clip

# ---------------------------------------------------------------- Metrica (soccer)
# github.com/metrica-sports/sample-data — Sample_Game_1/2 CSV format, 25 Hz,
# coordinates normalised to [0, 1]. The sample does not state the pitch size, so
# the FIFA-standard 105 x 68 m is assumed (same assumption as wheres-the-ball).


def load_metrica_game(game_dir: str | pathlib.Path, *, clip_seconds: float = 4.0,
                      stride_seconds: float = 4.0, target_fps: float | None = 5.0) -> Iterator[Clip]:
    game_dir = pathlib.Path(game_dir)
    name = game_dir.name
    length, width = FIELD_SIZE_M["soccer"]
    sides, teams = [], []
    for side, flag in (("Home", 1), ("Away", -1)):
        rows = list(csv.reader((game_dir / f"{name}_RawTrackingData_{side}_Team.csv").open()))
        arr = np.array([[float(c) if c not in ("", "NaN") else np.nan for c in r[3:]]
                        for r in rows[3:]], dtype=np.float32)
        players = arr[:, :-2]  # last x,y pair is the ball — dropped here, on purpose
        sides.append(players.reshape(len(arr), -1, 2))
        teams += [flag] * sides[-1].shape[1]
    n = min(s.shape[0] for s in sides)
    xy = np.concatenate([s[:n] for s in sides], axis=1)
    xy = xy * np.array([length, width], dtype=np.float32)
    yield from window_stream(
        xy, sport="soccer", source="metrica", match_id=name, fps=25.0,
        clip_seconds=clip_seconds, stride_seconds=stride_seconds, target_fps=target_fps,
        team=np.array(teams), field_size=(length, width),
    )


# ------------------------------------------------------------ SportVU (basketball)
# NBA 2015-16 SportVU JSON dumps (25 Hz, feet). Events overlap in time, so moments
# are de-duplicated by timestamp and re-split into continuous segments.

_FT = 0.3048


def load_sportvu_game(json_path: str | pathlib.Path, *, clip_seconds: float = 4.0,
                      stride_seconds: float = 4.0, target_fps: float | None = 5.0) -> Iterator[Clip]:
    d = json.loads(pathlib.Path(json_path).read_text())
    game_id = str(d.get("gameid", pathlib.Path(json_path).stem))
    moments: dict[int, dict[int, tuple[float, float]]] = {}
    team_of: dict[int, int] = {}
    home = d["events"][0].get("home", {}).get("teamid") if d.get("events") else None
    for ev in d["events"]:
        for m in ev["moments"]:
            ts = int(m[1])
            if ts in moments:
                continue
            ents = {}
            for e in m[5]:
                team, pid = e[0], e[1]
                if team == -1:  # the ball
                    continue
                team_of.setdefault(pid, 1 if team == home else -1)
                ents[pid] = (e[2] * _FT, e[3] * _FT)
            moments[ts] = ents
    stamps = sorted(moments)
    # split where consecutive moments are > 0.1 s apart (stoppages, quarter breaks)
    segments, cur = [], [stamps[0]] if stamps else []
    for a, b in zip(stamps, stamps[1:]):
        if b - a > 100:
            segments.append(cur)
            cur = []
        cur.append(b)
    if cur:
        segments.append(cur)
    for si, seg in enumerate(segments):
        pids = sorted({p for ts in seg for p in moments[ts]})
        idx = {p: i for i, p in enumerate(pids)}
        xy = np.full((len(seg), len(pids), 2), np.nan, dtype=np.float32)
        for fi, ts in enumerate(seg):
            for p, pos in moments[ts].items():
                xy[fi, idx[p]] = pos
        yield from window_stream(
            xy, sport="basketball", source="sportvu", match_id=game_id, fps=25.0,
            clip_seconds=clip_seconds, stride_seconds=stride_seconds, target_fps=target_fps,
            team=np.array([team_of.get(p, 0) for p in pids]),
            id_prefix=f"sportvu-{game_id}-s{si:03d}",
        )


# ------------------------------------------------- NFL Big Data Bowl (American football)
# Kaggle BDB tracking_week_*.csv (2023-2025 editions): one row per (game, play,
# player, frame), 10 Hz, yards, the ball as a row with nflId "NA". The 2026
# edition only ships a subset of players while the ball is in the air — usable,
# but the 2025 files (all 22 players, pre-snap included) are the better fit.
# Pre-snap and post-snap are emitted as separate streams: pre-snap clips are the
# purest "formation" sub-case in the whole study.

_YD = 0.9144
_NFL_HZ = 10.0
SNAP_EVENTS = ("ball_snap", "autoevent_ballsnap")  # the first of either marks the snap
NFL_PHASES = ("aligned", "random")


def _nfl_seed(game: str, play: str) -> int:
    """Deterministic per-play seed: the window of a play never depends on the others."""
    return zlib.crc32(f"{game}|{play}".encode())


def load_nfl_tracking(csv_path: str | pathlib.Path, *, clip_seconds: float = 4.0,
                      stride_seconds: float = 4.0, target_fps: float | None = 5.0,
                      max_plays: int | None = None,
                      trim_start_s: float = 0.0, phase: str = "aligned",
                      min_after_snap_s: float = 0.5,
                      stats: dict | None = None) -> Iterator[Clip]:
    """Plays -> clips.

    phase="aligned" (the original behaviour): windows are cut from the start of each
    play stream. `trim_start_s` drops the start of every stream: BDB 2023 plays start
    0.5 s before the snap, so untrimmed clips are all "still, then everyone moves at
    once" — an alignment no other sport's random windows have. Trimming 1.5 s gives
    mid-play clips, but still at a fixed offset from the snap.

    phase="random": one clip per play, placed relative to the snap (first
    `SNAP_EVENTS` event), not to the recording start. The window of `clip_seconds`
    starts at a uniformly random frame in [snap + min_after_snap_s, play end -
    clip_seconds], seeded by (gameId, playId), so no fixed phase of the play is a cue.
    Plays without a snap event or too short for the window are skipped; `stats`, if
    given, receives the counts (plays, clips, no_snap, too_short, too_few_players).
    """
    if phase not in NFL_PHASES:
        raise ValueError(f"phase must be one of {NFL_PHASES}, got {phase!r}")
    if phase == "random" and trim_start_s:
        raise ValueError("trim_start_s is for phase='aligned'; phase='random' places "
                         "clips relative to the snap")
    plays: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        f = reader.fieldnames or []
        snake = "game_id" in f  # 2026 edition uses snake_case
        has_phase = ("frame_type" if snake else "frameType") in f  # only the 2025 edition
        c = {k: (k if not snake else s) for k, s in (
            ("gameId", "game_id"), ("playId", "play_id"), ("nflId", "nfl_id"),
            ("frameId", "frame_id"), ("frameType", "frame_type"), ("club", "club"))}
        for r in reader:
            nid = r.get(c["nflId"], "")
            if nid in ("", "NA", "nan") or r.get(c["club"]) == "football":
                continue
            key = (r[c["gameId"]], r[c["playId"]])
            if max_plays and key not in plays and len(plays) >= max_plays:
                continue
            fr = plays[key].setdefault(int(r[c["frameId"]]),
                                       {"type": r.get(c["frameType"], ""), "p": {}, "snap": False})
            fr["p"][nid] = (float(r["x"]) * _YD, float(r["y"]) * _YD)
            if r.get("event") in SNAP_EVENTS:
                fr["snap"] = True
    if phase == "random":
        yield from _nfl_random_phase(plays, clip_seconds=clip_seconds, target_fps=target_fps,
                                     min_after_snap_s=min_after_snap_s, stats=stats)
        return
    for (game, play), frames in plays.items():
        ids = sorted({p for fr in frames.values() for p in fr["p"]})
        idx = {p: i for i, p in enumerate(ids)}
        order = sorted(frames)
        # Without frameType there is no reliable snap marker: one untagged-phase stream.
        phases = (("pre_snap", "BEFORE_SNAP"), ("post_snap", None)) if has_phase else (("play", None),)
        for phase_tag, want in phases:
            sel = [k for k in order if (frames[k]["type"] == want if want
                                        else frames[k]["type"] != "BEFORE_SNAP")]
            if not sel:
                continue
            xy = _nfl_xy(frames, sel, idx)
            if trim_start_s:
                xy = xy[int(round(trim_start_s * _NFL_HZ)):]
                if not len(xy):
                    continue
            yield from window_stream(
                xy, sport="american_football", source="nfl_bdb", match_id=game, fps=_NFL_HZ,
                clip_seconds=clip_seconds, stride_seconds=stride_seconds, target_fps=target_fps,
                id_prefix=f"nfl-{game}-{play}-{phase_tag}" + (f"-t{trim_start_s:g}" if trim_start_s else ""),
                tags=[phase_tag] + (["mid_play"] if trim_start_s else []),
            )


def _nfl_xy(frames: dict[int, dict], sel: list[int], idx: dict[str, int]) -> np.ndarray:
    xy = np.full((len(sel), len(idx), 2), np.nan, dtype=np.float32)
    for fi, k in enumerate(sel):
        for p, pos in frames[k]["p"].items():
            xy[fi, idx[p]] = pos
    return xy


def _nfl_random_phase(plays: dict[tuple[str, str], dict[int, dict]], *, clip_seconds: float,
                      target_fps: float | None, min_after_snap_s: float,
                      stats: dict | None) -> Iterator[Clip]:
    st = {"plays": len(plays), "clips": 0, "no_snap": 0, "too_short": 0, "too_few_players": 0}
    length = int(round(clip_seconds * _NFL_HZ))
    for (game, play), frames in plays.items():
        order = sorted(frames)
        snap = next((i for i, k in enumerate(order) if frames[k]["snap"]), None)
        if snap is None:
            st["no_snap"] += 1
            continue
        lo = snap + int(round(min_after_snap_s * _NFL_HZ))
        hi = len(order) - length  # last start whose window still ends inside the play
        if hi < lo:
            st["too_short"] += 1
            continue
        start = int(np.random.default_rng(_nfl_seed(game, play)).integers(lo, hi + 1))
        ids = sorted({p for fr in frames.values() for p in fr["p"]})
        xy = _nfl_xy(frames, order[start:start + length], {p: i for i, p in enumerate(ids)})
        clips = list(window_stream(
            xy, sport="american_football", source="nfl_bdb", match_id=game, fps=_NFL_HZ,
            clip_seconds=clip_seconds, stride_seconds=clip_seconds, target_fps=target_fps,
            id_prefix=f"nfl-{game}-{play}-rand", tags=["mid_play", "random_phase"],
        ))
        if not clips:
            st["too_few_players"] += 1
            continue
        clip = clips[0]  # the window is exactly one clip long
        clip.meta.update(start_frame=int(start), snap_frame=int(snap),
                         snap_offset_s=round((start - snap) / _NFL_HZ, 3),
                         play_seconds=round(len(order) / _NFL_HZ, 3))
        st["clips"] += 1
        yield clip
    if stats is not None:
        stats.update(st)
