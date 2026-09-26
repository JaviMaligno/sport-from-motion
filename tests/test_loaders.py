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
