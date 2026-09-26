"""Adapters for the concrete public sources. Each yields `Clip`s in metres, no ball.

Formats follow the loaders already used in `wheres-the-ball`
(src/wheres_the_ball/data/field_tracking.py) for Metrica and SportVU, so the two
projects read the same files the same way.
"""
from __future__ import annotations

import csv
import json
import pathlib
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


def load_nfl_tracking(csv_path: str | pathlib.Path, *, clip_seconds: float = 4.0,
                      stride_seconds: float = 4.0, target_fps: float | None = 5.0,
                      max_plays: int | None = None) -> Iterator[Clip]:
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
                                       {"type": r.get(c["frameType"], ""), "p": {}})
            fr["p"][nid] = (float(r["x"]) * _YD, float(r["y"]) * _YD)
    for (game, play), frames in plays.items():
        ids = sorted({p for fr in frames.values() for p in fr["p"]})
        idx = {p: i for i, p in enumerate(ids)}
        order = sorted(frames)
        # Without frameType there is no reliable snap marker: one untagged-phase stream.
        phases = (("pre_snap", "BEFORE_SNAP"), ("post_snap", None)) if has_phase else (("play", None),)
        for phase, want in phases:
            sel = [k for k in order if (frames[k]["type"] == want if want
                                        else frames[k]["type"] != "BEFORE_SNAP")]
            if not sel:
                continue
            xy = np.full((len(sel), len(ids), 2), np.nan, dtype=np.float32)
            for fi, k in enumerate(sel):
                for p, pos in frames[k]["p"].items():
                    xy[fi, idx[p]] = pos
            yield from window_stream(
                xy, sport="american_football", source="nfl_bdb", match_id=game, fps=10.0,
                clip_seconds=clip_seconds, stride_seconds=stride_seconds, target_fps=target_fps,
                id_prefix=f"nfl-{game}-{play}-{phase}", tags=[phase],
            )
