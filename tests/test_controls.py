import numpy as np
import pytest

from motion_sport.controls import (
    PRESETS, ControlConfig, apply_controls, duplicate_tracks, fill_short_gaps, fit_window,
    fix_player_count, linear_tracks, pair_max_distance,
    frozen_tracks, max_step_speed, median_speed, normalize_space, random_rigid, select_players,
    window_span,
)
from motion_sport.loaders import make_toy_clips
from motion_sport.schema import Clip


def _clip(xy, sport="soccer", **kw):
    return Clip(clip_id="c", sport=sport, source="t", match_id="m", fps=5.0, xy=xy, **kw)


def test_fill_short_gaps_interpolates_interior_only():
    xy = np.zeros((6, 1, 2), np.float32)
    xy[:, 0, 0] = [0, 1, np.nan, 3, 4, np.nan]
    out = fill_short_gaps(_clip(xy)).xy[:, 0, 0]
    assert out[2] == pytest.approx(2.0)
    assert np.isnan(out[5])  # trailing gap: no value after it to interpolate towards


def test_fix_player_count_exact_and_rejects_short():
    rng = np.random.default_rng(0)
    c = make_toy_clips(1)[0]
    assert fix_player_count(c, 12, rng).n_players == 12
    assert fix_player_count(c, 99, rng) is None


def test_normalize_space_unit_spread():
    c = normalize_space(make_toy_clips(1)[0], "spread")
    rms = np.sqrt(np.mean(np.sum(c.xy ** 2, axis=2)))
    assert rms == pytest.approx(1.0, rel=1e-4)
    assert np.abs(c.xy.mean(axis=(0, 1))).max() < 1e-4


def test_random_rigid_preserves_distances():
    c = make_toy_clips(1)[0]
    r = random_rigid(c, np.random.default_rng(3))
    d0 = np.linalg.norm(c.xy[5, 0] - c.xy[5, 1])
    d1 = np.linalg.norm(r.xy[5, 0] - r.xy[5, 1])
    assert d0 == pytest.approx(d1, rel=1e-5)


def test_fit_window_hits_tempo_target():
    c = make_toy_clips(1, seconds=12)[0]
    target = 0.5 * median_speed(normalize_space(fit_window(c, 20, space="spread"), "spread"))
    w = fit_window(c, 20, space="spread", tempo=target)
    assert w.n_frames == 20
    assert median_speed(normalize_space(w, "spread")) == pytest.approx(target, rel=0.05)


def test_fit_window_rejects_when_speed_up_needs_more_footage():
    c = make_toy_clips(1, seconds=4)[0]  # exactly 20 frames: no room to speed up
    fast = 3 * median_speed(normalize_space(c, "spread"))
    assert fit_window(c, 20, space="spread", tempo=fast) is None


def test_strict_preset_output_shape_and_no_team():
    rng = np.random.default_rng(0)
    c = make_toy_clips(1, seconds=6)[0]
    c = c.replace(team=np.ones(c.n_players, np.int8))
    out = apply_controls(c, PRESETS["strict"], rng)
    assert out.xy.shape == (20, 12, 2)
    assert out.team is None
    assert [s["name"] for s in out.meta["controls"]][:2] == ["fill_short_gaps", "fix_player_count"]


def test_auto_tempo_must_be_resolved():
    with pytest.raises(ValueError):
        apply_controls(make_toy_clips(1, seconds=10)[0], PRESETS["strict_tempo"],
                       np.random.default_rng(0))


def test_strict_smooth_reduces_jitter_and_keeps_shape():
    import numpy as np
    from motion_sport.controls import PRESETS, apply_controls
    from motion_sport.schema import Clip

    rng = np.random.default_rng(0)
    t = np.arange(40)[:, None, None] / 5.0
    base = np.concatenate([np.cos(t + np.arange(12)[None, :, None]),
                           np.sin(t + np.arange(12)[None, :, None])], axis=2) * 10
    noisy = (base + rng.normal(0, 0.8, base.shape)).astype(np.float32)
    clip = Clip(clip_id="c", sport="soccer", source="toy", match_id="m", fps=5.0, xy=noisy)

    def jerk(c):
        return float(np.abs(np.diff(c.xy, n=2, axis=0)).mean())

    a = apply_controls(clip, PRESETS["strict"], np.random.default_rng(1))
    b = apply_controls(clip, PRESETS["strict_smooth"], np.random.default_rng(1))
    assert a is not None and b is not None and a.xy.shape == b.xy.shape
    assert jerk(b) < 0.5 * jerk(a)


def _walk(n_players=12, t=20, speed=5.0, seed=0, radius=200.0):
    """Players walking at `speed` m/s, 5 Hz, in metres, on wide arcs (radius 200 m): the
    step length is `speed` / 5 to 1e-6, and the path is not an exact straight line, which
    only an interpolation is (D19: second difference v^2 dt^2 / R = 5 mm per step)."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0, 50, (n_players, 2))
    ang = rng.uniform(0, 2 * np.pi, n_players)
    u = np.stack([np.cos(ang), np.sin(ang)], 1)
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    th = (np.arange(t) / 5.0 * speed / radius)[:, None, None]
    return (start[None] + radius * (np.sin(th) * u[None] + (1 - np.cos(th)) * nrm[None])
            ).astype(np.float32)


def test_max_step_speed_is_in_metres_per_second():
    assert max_step_speed(_clip(_walk(speed=5.0))) == pytest.approx(5.0, rel=1e-4)


def test_teleport_filter_rejects_on_kept_players_only_and_counts_the_reason():
    cfg = ControlConfig(n_players=10, player_mode="central")
    ok = apply_controls(_clip(_walk()), cfg, np.random.default_rng(0))
    assert ok is not None and ok.meta["max_step_speed_ms"] == pytest.approx(5.0, rel=1e-3)
    assert ok.meta["control_config"]["max_speed_ms"] == 12.0
    xy = _walk()
    xy[10:, 3] += 30.0  # one step of 30 m in 0.2 s = 150 m/s: an ID swap
    why = {}
    all12 = ControlConfig(n_players=12)
    assert apply_controls(_clip(xy), all12, np.random.default_rng(0), reasons=why) is None
    assert why == {"teleport": 1}
    # the teleporting player is the only one far from the centre: with N=10 central of
    # 12 it is dropped first, and the clip survives
    far = _walk()
    far[:, 3] += 500.0
    far[10:, 3] += 30.0
    assert apply_controls(_clip(far), cfg, np.random.default_rng(0)) is not None
    # the check runs before any rescaling: a fast sport under the ceiling is kept
    assert apply_controls(_clip(_walk(speed=11.0)), cfg, np.random.default_rng(0)) is not None
    assert apply_controls(_clip(xy), ControlConfig(n_players=12, max_speed_ms=None),
                          np.random.default_rng(0)) is not None


def test_other_rejections_are_counted_by_reason():
    why = {}
    few = _walk(n_players=6)
    assert apply_controls(_clip(few), ControlConfig(n_players=10), np.random.default_rng(0),
                          reasons=why) is None
    short = _walk(t=10)
    assert apply_controls(_clip(short), ControlConfig(n_players=10), np.random.default_rng(0),
                          reasons=why) is None
    assert why == {"players": 1, "window": 1}


def test_every_preset_has_the_speed_ceiling():
    assert all(cfg.max_speed_ms == 12.0 for cfg in PRESETS.values())


# ---------------------------------------------------------------- frozen tracks (D18)

def test_frozen_tracks_are_exactly_constant_and_fully_observed():
    xy = _walk(n_players=5, t=20)
    xy[:, 1] = 0.0                    # TeamTrack's missing detection, whole window
    xy[:, 2] = xy[0, 2]               # held value
    xy[:, 3] = xy[0, 3]
    xy[7, 3, 0] += 1e-4               # moves once, a tenth of a millimetre: a player
    xy[:, 4] = np.nan                 # unobserved: not "frozen", just absent
    assert frozen_tracks(xy).tolist() == [False, True, True, False, False]
    assert not frozen_tracks(xy[:1]).any()  # one frame says nothing


def test_window_span_matches_the_centred_window():
    assert window_span(20, 20) == slice(0, 20)
    assert window_span(20, None) == slice(0, 20)
    assert window_span(24, 20) == slice(2, 22)
    assert window_span(23, 20) == slice(1, 22)  # half-frame centre: both neighbours count
    clip = _clip(_walk(t=24).astype(np.float32))
    win = fit_window(clip, 20, space="spread")
    assert np.allclose(win.xy, clip.xy[window_span(24, 20)])


def test_frozen_tracks_are_never_kept_and_are_counted():
    """A frozen dot among moving players is dropped as a non-player; the clip keeps N real
    players, and a clip left with fewer is rejected for players."""
    cfg = ControlConfig(n_players=10, player_mode="random")
    xy = _walk(n_players=12, t=20)
    xy[:, 0] = 0.0          # (0, 0) corner dot
    xy[:, 5] = xy[0, 5]     # held position
    for seed in range(20):
        dropped, why = {}, {}
        out = apply_controls(_clip(xy), cfg, np.random.default_rng(seed), reasons=why,
                             dropped=dropped)
        assert out is not None and why == {}
        assert dropped == {"frozen_tracks": 2, "clips_with_frozen": 1}
        assert (np.ptp(out.xy, axis=0).max(axis=1) > 1e-3).all()  # every kept dot moves
        assert any(c["name"] == "drop_frozen" and c["n"] == 2 for c in out.meta["controls"])
    xy3 = xy.copy()
    xy3[:, 7] = xy3[0, 7]   # third frozen track: 9 real players left for N = 10
    dropped, why = {}, {}
    assert apply_controls(_clip(xy3), cfg, np.random.default_rng(0), reasons=why,
                          dropped=dropped) is None
    assert why == {"players": 1}
    assert dropped == {"frozen_tracks": 3, "clips_with_frozen": 1, "clips_rejected_after_frozen": 1}


def test_frozen_only_over_the_window_counts_and_outside_it_does_not():
    cfg = ControlConfig(n_players=10, player_mode="random")
    xy = _walk(n_players=11, t=24)            # window = frames 2..21
    xy[2:22, 0] = xy[2, 0]                    # frozen inside the window, moves outside it
    d = {}
    assert select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d).n_players == 10
    assert d["frozen_tracks"] == 1
    xy = _walk(n_players=11, t=24)
    xy[3:22, 0] = xy[3, 0]                    # moves at frame 2, inside the window
    d = {}
    select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d)
    assert d == {}


def test_clips_without_frozen_tracks_are_unchanged_by_the_check():
    """Same draws and output as before D18 when nothing is frozen (other sources' clips)."""
    xy = _walk(n_players=14, t=20)
    on = apply_controls(_clip(xy), PRESETS["strict_smooth"], np.random.default_rng(3))
    off = apply_controls(_clip(xy), ControlConfig(smooth=2.0, drop_frozen=False),
                         np.random.default_rng(3))
    assert np.array_equal(on.xy, off.xy)
    assert all(cfg.drop_frozen for cfg in PRESETS.values())


def test_drop_frozen_off_keeps_the_old_behaviour_and_toy_clips_are_exempt():
    # a whole-window hold is also exactly linear (D19): the old behaviour needs both off
    cfg = ControlConfig(n_players=12, drop_frozen=False, drop_linear=False)
    xy = _walk(n_players=12, t=20)
    xy[:, 0] = 0.0
    assert apply_controls(_clip(xy), cfg, np.random.default_rng(0)) is not None
    toy = Clip(clip_id="c", sport="soccer", source="toy", match_id="m", fps=5.0, xy=xy)
    assert apply_controls(toy, ControlConfig(n_players=12), np.random.default_rng(0)) is not None
    assert apply_controls(_clip(xy), ControlConfig(n_players=12), np.random.default_rng(0)) is None


# ---------------------------------------------------------------- duplicated tracks (D19)

def _dup(xy, i, j, jitter=0.0, seed=0):
    """Track j becomes a copy of track i, with an optional per-frame jitter (metres)."""
    xy[:, j] = xy[:, i] + np.random.default_rng(seed).uniform(-jitter, jitter, xy[:, i].shape)
    return xy


def test_pair_max_distance_is_the_largest_over_frames_and_inf_off_observed():
    xy = _walk(n_players=4, t=20)
    xy[:, 1] = xy[:, 0] + [0.3, 0.0]
    xy[9, 1] = xy[9, 0] + [0.0, 2.0]          # one frame 2 m apart: the max, not the mean
    xy[5, 3] = np.nan                         # not fully observed
    d = pair_max_distance(xy)
    assert d[0, 1] == pytest.approx(2.0, abs=1e-5) and d[1, 0] == d[0, 1]
    assert np.isinf(d[0, 0]) and np.isinf(d[0, 3]) and np.isinf(d[3, 2])


def test_duplicate_tracks_keep_the_lower_index_and_use_the_max_not_the_mean():
    xy = _walk(n_players=6, t=20)
    _dup(xy, 1, 4, jitter=0.01)               # within 2 cm every frame: a copy
    _dup(xy, 2, 5)
    xy[:, 5] += [0.02, 0.0]                   # 2 cm off throughout: still a copy
    assert duplicate_tracks(xy, 0.05).tolist() == [False, False, False, False, True, True]
    part = _walk(n_players=3, t=20)
    _dup(part, 0, 2)
    part[-1, 2] += [0.5, 0.0]                 # a copy for 19 frames, 50 cm apart at the end
    assert not duplicate_tracks(part, 0.05).any()  # mean 2.5 cm, max 50 cm: not "the whole window"
    three = _walk(n_players=4, t=20)
    _dup(three, 0, 2)
    _dup(three, 0, 3, jitter=0.005, seed=1)   # two copies of one player: both go
    assert duplicate_tracks(three, 0.05).tolist() == [False, False, True, True]
    gone = _walk(n_players=3, t=20)
    _dup(gone, 0, 1)
    gone[4, 0] = np.nan                       # the original is not fully observed: nothing to copy
    assert not duplicate_tracks(gone, 0.05).any()


def test_duplicate_tracks_are_never_kept_and_are_counted():
    """SportVU 0021500368 segment 12: 10 ids, one player twice -> 9 players, rejected for
    N = 10; with 12 ids the clip keeps 10 distinct players."""
    cfg = ControlConfig(n_players=10, player_mode="random")
    xy = _dup(_walk(n_players=12, t=20), 2, 9, jitter=0.01)
    for seed in range(20):
        dropped, why = {}, {}
        out = apply_controls(_clip(xy), cfg, np.random.default_rng(seed), reasons=why,
                             dropped=dropped)
        assert out is not None and why == {}
        assert dropped == {"duplicate_tracks": 1, "clips_with_duplicate": 1}
        scale = out.meta["space_scale"]
        d = pair_max_distance(out.xy) * scale
        assert (d[np.triu_indices(10, 1)] > 0.05).all()
        assert any(c["name"] == "drop_duplicates" and c["n"] == 1 for c in out.meta["controls"])
    ten = _dup(_walk(n_players=10, t=20), 1, 2)
    dropped, why = {}, {}
    assert apply_controls(_clip(ten), cfg, np.random.default_rng(0), reasons=why,
                          dropped=dropped) is None
    assert why == {"players": 1}
    assert dropped == {"duplicate_tracks": 1, "clips_with_duplicate": 1,
                       "clips_rejected_after_duplicate": 1}


def test_duplicates_only_over_the_window_count():
    cfg = ControlConfig(n_players=10, player_mode="random")
    xy = _walk(n_players=11, t=24)            # window = frames 2..21
    xy[2:22, 10] = xy[2:22, 0]                # a copy inside the window, apart outside it
    d = {}
    assert select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d).n_players == 10
    assert d == {"duplicate_tracks": 1, "clips_with_duplicate": 1}
    xy = _walk(n_players=11, t=24)
    xy[3:22, 10] = xy[3:22, 0]                # apart at frame 2, inside the window
    d = {}
    select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d)
    assert d == {}


def test_frozen_copies_count_as_frozen_not_as_duplicates():
    cfg = ControlConfig(n_players=10, player_mode="random")
    xy = _walk(n_players=13, t=20)
    xy[:, 11] = 0.0
    xy[:, 12] = 0.0                           # two TeamTrack (0, 0) tracks: frozen, not copies
    d = {}
    assert select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d).n_players == 10
    assert d == {"frozen_tracks": 2, "clips_with_frozen": 1}


def test_clips_without_duplicates_are_unchanged_by_the_check():
    """Same draws and output as before D19 when no two tracks coincide (almost every clip)."""
    xy = _walk(n_players=14, t=20)
    on = apply_controls(_clip(xy), PRESETS["strict_smooth"], np.random.default_rng(3))
    off = apply_controls(_clip(xy), ControlConfig(smooth=2.0, drop_duplicates=False),
                         np.random.default_rng(3))
    assert np.array_equal(on.xy, off.xy)
    assert all(cfg.drop_duplicates and cfg.duplicate_tol_m == 0.05 for cfg in PRESETS.values())


def test_drop_duplicates_off_keeps_the_copy_and_toy_clips_are_exempt():
    xy = _dup(_walk(n_players=10, t=20), 1, 2)
    assert apply_controls(_clip(xy), ControlConfig(n_players=10, drop_duplicates=False),
                          np.random.default_rng(0)) is not None
    toy = Clip(clip_id="c", sport="soccer", source="toy", match_id="m", fps=5.0, xy=xy)
    assert apply_controls(toy, ControlConfig(n_players=10), np.random.default_rng(0)) is not None
    assert apply_controls(_clip(xy), ControlConfig(n_players=10), np.random.default_rng(0)) is None


# ---------------------------------------------------------------- exactly linear tracks (D19)

def test_linear_tracks_need_the_whole_min_run_of_constant_velocity():
    xy = _walk(n_players=5, t=20)                      # constant velocity: all linear
    xy += np.random.default_rng(1).normal(0, 0.01, xy.shape).astype(np.float32)  # 1 cm noise
    xy[:, 1] = np.linspace([0, 0], [8, 3], 20)          # interpolated over the whole window
    xy[:, 2] = xy[0, 2]                                 # a hold is linear at zero speed
    xy[:10, 3] = np.linspace([5, 5], [6, 7], 10)        # linear for 10 frames only
    xy[:, 4] = np.linspace([0, 0], [8, 3], 20)
    xy[6, 4] = np.nan                                   # not fully observed: never "linear"
    assert linear_tracks(xy, 20).tolist() == [False, True, True, False, False]
    assert linear_tracks(xy, 10).tolist() == [False, True, True, True, False]
    bent = np.linspace([0, 0], [8, 3], 20)
    bent[10:] += np.linspace([0, 0], [0.02, 0], 10)     # bends by 2 mm per step half-way
    assert not linear_tracks(bent[:, None].astype(np.float32), 20).any()
    far = (np.linspace([60, 40], [68, 43], 20)).astype(np.float32)  # float32 at pitch scale
    assert linear_tracks(far[:, None], 20).all()


def test_linear_tracks_are_dropped_counted_and_measured_clips_unchanged():
    cfg = ControlConfig(n_players=10, player_mode="random")
    xy = _walk(n_players=11, t=20)
    xy += np.random.default_rng(2).normal(0, 0.01, xy.shape).astype(np.float32)
    lin = xy.copy()
    lin[:, 4] = np.linspace([1, 1], [9, 4], 20)          # TeamTrack keyframe interpolation
    dropped, why = {}, {}
    out = apply_controls(_clip(lin), cfg, np.random.default_rng(0), reasons=why, dropped=dropped)
    assert out is not None and dropped == {"linear_tracks": 1, "clips_with_linear": 1}
    assert any(c["name"] == "drop_linear" and c["n"] == 1 for c in out.meta["controls"])
    lin[:, 6] = np.linspace([3, 1], [2, 4], 20)          # a second one: 9 players left
    dropped, why = {}, {}
    assert apply_controls(_clip(lin), cfg, np.random.default_rng(0), reasons=why,
                          dropped=dropped) is None
    assert why == {"players": 1}
    assert dropped == {"linear_tracks": 2, "clips_with_linear": 1,
                       "clips_rejected_after_linear": 1}
    xy14 = _walk(n_players=14, t=20)          # measured-like clip: the check changes nothing
    on = apply_controls(_clip(xy14), PRESETS["strict_smooth"], np.random.default_rng(3))
    off = apply_controls(_clip(xy14), ControlConfig(smooth=2.0, drop_linear=False),
                         np.random.default_rng(3))
    assert np.array_equal(on.xy, off.xy)
    assert all(c.drop_linear and c.linear_min_s == 4.0 for c in PRESETS.values())


def test_linear_min_is_in_seconds_and_counts_inside_the_window_only():
    """At 8 s (40 frames) a 4 s straight stretch is enough; outside the window it is not."""
    cfg = ControlConfig(n_players=10, player_mode="random", n_frames=40)
    xy = _walk(n_players=11, t=44)                      # window = frames 2..41
    xy += np.random.default_rng(3).normal(0, 0.01, xy.shape).astype(np.float32)
    xy[20:40, 0] = np.linspace([1, 1], [5, 2], 20)      # 4 s straight, inside the window
    d = {}
    assert select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d).n_players == 10
    assert d == {"linear_tracks": 1, "clips_with_linear": 1}
    xy = _walk(n_players=11, t=44)
    xy += np.random.default_rng(3).normal(0, 0.01, xy.shape).astype(np.float32)
    xy[:21, 0] = np.linspace([1, 1], [5, 2], 21)        # 21 frames straight, 2 of them outside
    d = {}
    select_players(_clip(xy), cfg, np.random.default_rng(0), dropped=d)
    assert d == {}
