import numpy as np
import pytest

from motion_sport.controls import (
    PRESETS, ControlConfig, apply_controls, fill_short_gaps, fit_window, fix_player_count,
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


def _walk(n_players=12, t=20, speed=5.0, seed=0):
    """Players walking in straight lines at `speed` m/s, 5 Hz, in metres."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0, 50, (n_players, 2))
    ang = rng.uniform(0, 2 * np.pi, n_players)
    v = np.stack([np.cos(ang), np.sin(ang)], 1) * speed
    return (start[None] + np.arange(t)[:, None, None] / 5.0 * v[None]).astype(np.float32)


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
    cfg = ControlConfig(n_players=12, drop_frozen=False)
    xy = _walk(n_players=12, t=20)
    xy[:, 0] = 0.0
    assert apply_controls(_clip(xy), cfg, np.random.default_rng(0)) is not None
    toy = Clip(clip_id="c", sport="soccer", source="toy", match_id="m", fps=5.0, xy=xy)
    assert apply_controls(toy, ControlConfig(n_players=12), np.random.default_rng(0)) is not None
    assert apply_controls(_clip(xy), ControlConfig(n_players=12), np.random.default_rng(0)) is None
