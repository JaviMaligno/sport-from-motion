import json

import numpy as np
import pytest

from motion_sport import learners, pipeline
from motion_sport.controls import PRESETS, apply_controls
from motion_sport.loaders import make_toy_clips


def _controlled(n=1, seconds=6):
    rng = np.random.default_rng(0)
    return [apply_controls(c, PRESETS["strict"], rng) for c in make_toy_clips(n, seconds=seconds)]


def test_series_invariant_to_player_order_and_rotation():
    c = _controlled()[0]
    base = learners.invariant_series(c.xy, c.fps)
    perm = learners.invariant_series(c.xy[:, np.random.default_rng(1).permutation(c.n_players)], c.fps)
    a = 1.1
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    rotated = learners.invariant_series(c.xy @ rot.T, c.fps)
    assert base.shape == (len(learners.SERIES_CHANNELS), c.n_frames - 2)
    assert np.allclose(base, perm, atol=1e-4)
    assert np.allclose(base, rotated, atol=1e-3)


def test_grouped_oof_never_mixes_a_match_across_train_and_test():
    y = ["a", "b"] * 20
    groups = [f"m{i % 8}" for i in range(40)]
    seen = []

    def fit_predict(train, test):
        seen.append(({groups[i] for i in train}, {groups[i] for i in test}))
        return ["a", "b"], np.tile([0.5, 0.5], (len(test), 1))

    pred, probs = learners.grouped_oof(y, groups, fit_predict)
    assert seen and all(not (tr & te) for tr, te in seen)
    assert all(p for p in pred) and all(abs(sum(d.values()) - 1) < 1e-9 for d in probs)


def test_player_tokens_follow_the_view():
    c = _controlled()[0]
    m = learners.player_tokens(c, "motion", seed=3, k=8)
    s = learners.player_tokens(c, "motion_shuffled", seed=3, k=8)
    f = learners.player_tokens(c, "formation", seed=3, k=8)
    assert m.shape == (12, 32) and s.shape == (12, 32) and f.shape == (12, 4)
    assert not np.allclose(m, s)


@pytest.fixture()
def prepared(tmp_path):
    for c in make_toy_clips(24, seconds=6):
        c.save(tmp_path / "clips")
    return pipeline.prepare(str(tmp_path / "clips"), str(tmp_path / "items"), preset="strict",
                            conditions=["motion"], reprs=["text"])


def _acc(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return sum(r["label"] == r["sport"] for r in rows) / len(rows), rows


def test_minirocket_end_to_end(prepared):
    pytest.importorskip("aeon")
    acc, rows = _acc(pipeline.run_learner(str(prepared), "minirocket", "motion"))
    assert len(rows) == 48 and rows[0]["model"] == "minirocket" and rows[0]["repr"] == "series"
    assert acc > 0.8  # toy sports differ in motion by construction


def test_minirocket_rejects_single_frame():
    with pytest.raises(ValueError):
        learners.clip_series(_controlled()[0], "formation", seed=0)


def test_deepsets_end_to_end(prepared):
    pytest.importorskip("torch")
    acc, rows = _acc(pipeline.run_learner(str(prepared), "deepsets", "motion", epochs=60))
    assert len(rows) == 48 and rows[0]["repr"] == "tracks"
    assert acc > 0.7
