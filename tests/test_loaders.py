import numpy as np

from motion_sport.loaders import load_long_csv, load_metrica_game, load_nfl_tracking
from motion_sport.schema import Clip, load_clips


def test_long_csv_windows_and_units(tmp_path):
    rows = ["match_id,frame,track_id,x,y"]
    for f in range(40):
        for t in range(10):
            rows.append(f"g1,{f},{t},{t + f * 0.1},{t}")
    p = tmp_path / "t.csv"
    p.write_text("\n".join(rows))
    clips = list(load_long_csv(p, sport="handball", source="tt", fps=10, clip_seconds=2,
                               stride_seconds=2, unit_scale=2.0))
    assert len(clips) == 2
    assert clips[0].xy.shape == (20, 10, 2)
    assert clips[0].xy[0, 3, 1] == 6.0  # y=3 scaled by 2


def test_nfl_splits_pre_and_post_snap_and_drops_ball(tmp_path):
    rows = ["gameId,playId,nflId,frameId,frameType,club,x,y"]
    for f in range(1, 61):
        ftype = "BEFORE_SNAP" if f <= 30 else "AFTER_SNAP"
        for pid in range(22):
            rows.append(f"1,5,{pid},{f},{ftype},A,{10 + pid},{20 + f * 0.1}")
        rows.append(f"1,5,NA,{f},{ftype},football,50,26")
    p = tmp_path / "w1.csv"
    p.write_text("\n".join(rows))
    clips = list(load_nfl_tracking(p, clip_seconds=3, stride_seconds=3, target_fps=10))
    assert {c.tags[0] for c in clips} == {"pre_snap", "post_snap"}
    assert all(c.n_players == 22 for c in clips)  # ball row excluded
    assert all(c.sport == "american_football" for c in clips)


def test_metrica_drops_ball_and_scales_to_metres(tmp_path):
    game = tmp_path / "Sample_Game_1"
    game.mkdir()
    for side in ("Home", "Away"):
        players = "0.5,0.5,0.25,0.75,0.1,0.1,0.2,0.2"  # 4 players per side
        head = ["a", "b", "Period,Frame,Time,P1,,P2,,P3,,P4,,Ball,"]
        body = [f"1,{f},{f / 25},{players},0.9,0.9" for f in range(1, 101)]
        (game / f"Sample_Game_1_RawTrackingData_{side}_Team.csv").write_text("\n".join(head + body))
    clips = list(load_metrica_game(game, clip_seconds=2, stride_seconds=2, target_fps=5))
    c = clips[0]
    assert c.n_players == 8  # 4 per side, ball removed
    assert c.team.tolist() == [1] * 4 + [-1] * 4
    assert np.allclose(c.xy[0, 0], [52.5, 34.0])


def test_clip_roundtrip(tmp_path):
    c = Clip(clip_id="x", sport="rugby_union", source="s", match_id="m", fps=5,
             xy=np.random.rand(10, 5, 2), team=np.array([1, 1, -1, -1, 1]), tags=["lineout"])
    c.save(tmp_path)
    (d,) = load_clips(tmp_path)
    assert d.tags == ["lineout"] and np.allclose(d.xy, c.xy) and d.team.tolist() == [1, 1, -1, -1, 1]


def test_nfl_without_frame_type_is_not_labelled_post_snap(tmp_path):
    rows = ["gameId,playId,nflId,frameId,club,x,y"]
    for f in range(1, 31):
        rows += [f"1,5,{pid},{f},A,{10 + pid},{20 + f * 0.1}" for pid in range(22)]
    p = tmp_path / "old.csv"
    p.write_text("\n".join(rows))
    clips = list(load_nfl_tracking(p, clip_seconds=3, stride_seconds=3, target_fps=10))
    assert clips and all(c.tags == ["play"] for c in clips)


def _bdb2023(path, plays):
    """BDB-2023-like CSV (no frameType, an `event` column). plays: {(game, play): (n_frames,
    snap_frame or None)}; frames are 1-based, 10 Hz, 22 players + the ball."""
    rows = ["gameId,playId,nflId,frameId,team,x,y,event"]
    for (game, play), (n, snap) in plays.items():
        for f in range(1, n + 1):
            ev = "ball_snap" if f == snap else ("pass_forward" if f == n - 2 else "None")
            rows += [f"{game},{play},{pid},{f},A,{10 + pid + f * 0.1},{20 + pid * 0.5},{ev}"
                     for pid in range(22)]
            rows.append(f"{game},{play},NA,{f},football,50,26,{ev}")
    path.write_text("\n".join(rows))
    return path


def test_nfl_random_phase_one_clip_per_play_after_the_snap(tmp_path):
    plays = {("g1", str(p)): (80, 6) for p in range(12)}
    plays[("g1", "short")] = (40, 6)       # 0.5 s + 4 s after the snap does not fit
    plays[("g1", "nosnap")] = (80, None)
    p = _bdb2023(tmp_path / "w.csv", plays)
    stats: dict = {}
    clips = list(load_nfl_tracking(p, clip_seconds=4, phase="random", stats=stats))
    assert stats == {"plays": 14, "clips": 12, "no_snap": 1, "too_short": 1, "too_few_players": 0}
    assert len(clips) == len({c.clip_id for c in clips}) == 12
    snap = 5  # frame 6 is index 5
    for c in clips:
        assert c.xy.shape == (20, 22, 2) and c.fps == 5.0  # 4 s decimated to 5 Hz
        assert {"mid_play", "random_phase"} <= set(c.tags)
        m = c.meta
        assert m["snap_frame"] == snap and m["snap_offset_s"] == (m["start_frame"] - snap) / 10
        assert 0.5 <= m["snap_offset_s"] <= (80 - 40 - snap) / 10
        # the window is the play from start_frame on: x of player 0 = 10 + 0.1 * frame (yards)
        assert np.isclose(c.xy[0, 0, 0], (10 + (m["start_frame"] + 1) * 0.1) * 0.9144, atol=1e-4)
    offsets = {c.meta["snap_offset_s"] for c in clips}
    assert len(offsets) > 3  # not one fixed phase


def test_nfl_random_phase_is_deterministic_per_play(tmp_path):
    a = _bdb2023(tmp_path / "a.csv", {("g1", "7"): (90, 6), ("g2", "3"): (90, 6)})
    b = _bdb2023(tmp_path / "b.csv", {("g2", "3"): (90, 6), ("g9", "1"): (90, 6), ("g1", "7"): (90, 6)})
    off = lambda p: {c.clip_id: c.meta["start_frame"] for c in load_nfl_tracking(p, phase="random")}
    oa, ob = off(a), off(b)
    assert oa == off(a)  # same file, same windows
    assert all(ob[k] == v for k, v in oa.items())  # other plays in the file do not matter


def test_nfl_random_phase_takes_the_first_snap_event_and_autoevent(tmp_path):
    p = _bdb2023(tmp_path / "w.csv", {("g1", "1"): (60, None)})
    text = p.read_text().splitlines()
    # frame 3: autoevent_ballsnap; frame 10: a later ball_snap that must be ignored
    text = [r.rsplit(",", 1)[0] + (",autoevent_ballsnap" if ",3,A," in r or ",3,football," in r
                                    else ",ball_snap" if ",10,A," in r else "," + r.rsplit(",", 1)[1])
            if i else r for i, r in enumerate(text)]
    p.write_text("\n".join(text))
    (c,) = load_nfl_tracking(p, phase="random", min_after_snap_s=0.0)
    assert c.meta["snap_frame"] == 2


def test_nfl_aligned_trim_is_unchanged_and_random_rejects_trim(tmp_path):
    import pytest

    p = _bdb2023(tmp_path / "w.csv", {("g1", "1"): (80, 6)})
    clips = list(load_nfl_tracking(p, clip_seconds=3, stride_seconds=3, trim_start_s=1.5))
    assert clips and all(c.tags == ["play", "mid_play"] for c in clips)
    assert clips[0].clip_id == "nfl-g1-1-play-t1.5-00000"
    with pytest.raises(ValueError):
        list(load_nfl_tracking(p, phase="random", trim_start_s=1.5))
    with pytest.raises(ValueError):
        list(load_nfl_tracking(p, phase="snap"))


def test_cli_ingest_nfl_random_phase_prints_summary(tmp_path, capsys):
    from motion_sport import cli

    p = _bdb2023(tmp_path / "w.csv", {("g1", "1"): (80, 6), ("g1", "2"): (30, 6)})
    cli.main(["ingest", "--source", "nfl", "--input", str(p), "--out", str(tmp_path / "c"),
              "--nfl-phase", "random", "--nfl-min-after-snap", "1.0"])
    out = capsys.readouterr().out
    assert "wrote 1 clips" in out and "2 plays -> 1 clips" in out and "1 too short" in out
    (c,) = load_clips(tmp_path / "c")
    assert "random_phase" in c.tags and c.meta["snap_offset_s"] >= 1.0
