import numpy as np
import pytest

from motion_sport.evaluate import (_prior_corrected_pred, _prob_matrix, boot_counts, holm,
                                   paired_difference, summarize)

CLS = ["a", "b"]


def _rows(p_a_given_true, n_matches=6, per=10):
    """Balanced rows; the model's p(a) depends on the true class only."""
    rows = []
    for m in range(n_matches):
        for k in range(per):
            s = CLS[k % 2]
            pa = p_a_given_true[s]
            rows.append({"clip_id": f"{m}-{k}", "match_id": f"m{m}", "sport": s,
                         "label": "a" if pa >= 0.5 else "b", "probs": {"a": pa, "b": 1 - pa}})
    return rows


def test_constant_answer_is_not_rewarded():
    s = summarize(_rows({"a": 0.9, "b": 0.9}), CLS, n_boot=200)
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["balanced_accuracy"] == pytest.approx(0.5)
    assert s["kappa"] == pytest.approx(0.0)
    assert s["macro_f1"] == pytest.approx(1 / 3)  # F1(a) = 2/3, F1(b) = 0
    assert s["per_class_recall"] == {"a": 1.0, "b": 0.0}
    assert s["predicted_share"]["a"] == 1.0
    lo, hi = s["kappa_ci"]
    assert lo <= 0 <= hi


def test_prior_correction_recovers_signal_hidden_by_bias():
    # always answers "a", but p(a) is higher when the truth is a: the ranking has signal
    s = summarize(_rows({"a": 0.9, "b": 0.6}), CLS, n_boot=200)
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["prior_corrected_accuracy"] == pytest.approx(1.0)
    lo, hi = s["prior_corrected_accuracy_ci"]
    assert 0.9 <= lo <= hi <= 1.0


def test_prior_is_leave_one_match_out():
    rng = np.random.default_rng(3)
    rows = [{"match_id": f"m{i % 3}", "sport": "a", "probs": dict(zip("abc", rng.dirichlet([1, 1, 1])))}
            for i in range(30)]
    classes = ["a", "b", "c"]
    P, valid = _prob_matrix(rows, classes)
    idx, _ = boot_counts([r["match_id"] for r in rows], 1)
    got = _prior_corrected_pred(P, valid, idx, np.ones((1, 3)))[0]
    for i, r in enumerate(rows):
        others = [j for j, q in enumerate(rows) if q["match_id"] != r["match_id"]]
        prior = P[others].mean(0)
        assert got[i] == np.argmax(P[i] / prior)


def test_errors_count_as_wrong_everywhere():
    rows = _rows({"a": 0.9, "b": 0.1})
    for r in rows[:10]:
        r.update(label=None, probs={}, error="boom")
    s = summarize(rows, CLS, n_boot=50)
    assert s["n_unparsed"] == 10
    assert s["accuracy"] == pytest.approx(50 / 60)
    assert s["prior_corrected_accuracy"] == pytest.approx(50 / 60)


def test_paired_difference_p_value():
    good = _rows({"a": 0.9, "b": 0.1})
    same = paired_difference(good, good, n_boot=500)
    assert same["diff"] == 0 and same["p"] == 1.0
    bad = [{**r, "label": "b" if r["sport"] == "a" else "a"} for r in good]
    far = paired_difference(good, bad, n_boot=500)
    assert far["diff"] == 1.0 and far["p"] < 0.01 and far["ci"][0] > 0
    # replicate-averaged correctness is used when present
    half = [{**r, "correct": 0.5} for r in good]
    assert paired_difference(good, half, n_boot=50)["diff"] == pytest.approx(0.5)



def test_paired_difference_needs_enough_matches():
    """One or two matches leaning the same way must not yield p ~ 2/(B+1)."""
    def rows(n_matches, n_right):
        return [{"clip_id": i, "match_id": i // 10, "sport": "a",
                 "label": "a" if i % 10 < n_right else "b"} for i in range(10 * n_matches)]

    for n_m in (1, 2, 4):
        r = paired_difference(rows(n_m, 2), rows(n_m, 1), n_boot=500)
        assert r["diff"] == pytest.approx(0.1) and r["n_matches"] == n_m
        assert r["too_few_matches"] and r["p"] != r["p"] and r["ci"][0] != r["ci"][0]
    r = paired_difference(rows(5, 2), rows(5, 1), n_boot=500)
    assert not r["too_few_matches"] and r["p"] == r["p"]
    # the NaN p of a too-small contrast stays out of the Holm family
    few = paired_difference(rows(1, 2), rows(1, 1), n_boot=500)["p"]
    assert holm([few, r["p"]])[1] == pytest.approx(r["p"])


def test_holm():
    adj = holm([0.01, 0.04, 0.03, 0.005, float("nan")])
    assert adj[:4] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert adj[4] != adj[4]
