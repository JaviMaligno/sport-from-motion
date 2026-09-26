import numpy as np
import pytest

from motion_sport.controls import (
    PRESETS, apply_controls, fill_short_gaps, fit_window, fix_player_count, median_speed,
    normalize_space, random_rigid,
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
