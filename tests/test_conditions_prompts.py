import numpy as np

from motion_sport.conditions import build_view, kinematics_layout
from motion_sport.loaders import make_toy_clips
from motion_sport.prompts import build_prompt, candidate_order, parse_answer
from motion_sport.serialize import view_to_text

CANDS = ["rugby_union", "soccer", "american_football"]


def test_shuffled_is_a_nontrivial_permutation_of_motion():
    c = make_toy_clips(1)[0]
    m = build_view(c, "motion", np.random.default_rng(0))
    s = build_view(c, "motion_shuffled", np.random.default_rng(0))
    assert sorted(s.order) == m.order and s.order != m.order
    assert not s.ordered


def test_formation_is_one_frame():
    v = build_view(make_toy_clips(1)[0], "formation", np.random.default_rng(0))
    assert len(v.frames) == 1


def test_kinematics_keeps_displacements_and_solo_keeps_speeds():
    xy = make_toy_clips(1)[0].xy
    kin = kinematics_layout(xy, np.random.default_rng(0))
    solo = kinematics_layout(xy, np.random.default_rng(0), solo=True)
    step = lambda a: np.linalg.norm(np.diff(a, axis=0), axis=2)  # noqa: E731
    assert np.allclose(step(kin), step(xy), atol=1e-4)
    assert np.allclose(step(solo), step(xy), atol=1e-4)
    # but solo changes directions
    assert not np.allclose(np.diff(solo, axis=0), np.diff(kin, axis=0), atol=1e-3)


def test_text_hides_timestamps_when_shuffled():
    c = make_toy_clips(1)[0]
    assert "t=" in view_to_text(build_view(c, "motion", np.random.default_rng(0)))
    assert "t=" not in view_to_text(build_view(c, "motion_shuffled", np.random.default_rng(0)))


def test_candidate_order_is_seeded_and_complete():
    a, b = candidate_order(CANDS, 1), candidate_order(CANDS, 1)
    assert a == b and sorted(a) == sorted(CANDS)
    assert len({tuple(candidate_order(CANDS, s)) for s in range(30)}) > 1


def test_prompt_lists_every_option_and_no_count_hint():
    v = build_view(make_toy_clips(1)[0], "motion", np.random.default_rng(0))
    prompt, order = build_prompt(v, "image", CANDS, seed=4)
    for c in ("rugby union", "association football (soccer)", "American football"):
        assert c in prompt
    assert "NOT the number of players" in prompt


def test_parse_answer_variants():
    ok = parse_answer('```json\n{"probabilities": {"rugby union": 0.7, "soccer": 0.2, '
                      '"American football": 0.1}, "answer": "rugby union", "rationale": "rucks"}\n```',
                      CANDS)
    assert ok["label"] == "rugby_union" and abs(sum(ok["probs"].values()) - 1) < 1e-9
    # answer missing -> argmax of probabilities; unnormalised probabilities are renormalised
    p = parse_answer('Sure: {"probabilities": {"soccer": 3, "rugby union": 1}}', CANDS)
    assert p["label"] == "soccer" and p["probs"]["soccer"] == 0.75
    assert parse_answer("I think rugby", CANDS)["label"] is None


def test_informed_prompt_is_symmetric_and_keeps_the_option_order():
    import re

    from motion_sport.prompts import MOVEMENT
    from motion_sport.schema import SPORTS

    assert set(MOVEMENT) == set(SPORTS)  # every registered sport has a description
    lens = [len(d) for d in MOVEMENT.values()]
    assert max(lens) <= 1.25 * min(lens)  # similar length: no sport gets more help
    assert not any(re.search(r"\d", d) for d in MOVEMENT.values())  # no numbers
    v = build_view(make_toy_clips(1)[0], "motion", np.random.default_rng(0))
    neutral, order = build_prompt(v, "image", CANDS, seed=4)
    informed, order_i = build_prompt(v, "image", CANDS, seed=4, style="informed")
    assert order_i == order
    assert "How players typically move" in informed and "How players" not in neutral
    guide = informed.split("How players typically move", 1)[1]
    pos = [guide.index(f"- {SPORTS[c]}:") for c in order]
    assert pos == sorted(pos)  # descriptions listed in the shown (shuffled) order
    assert informed.startswith(neutral.split("\n\n")[0])  # same preamble + view description
    assert informed.endswith(neutral.rsplit("\n\n", 1)[1])  # same question


def test_informed_decision_criteria():
    from motion_sport.prompts import build_decision

    v = build_view(make_toy_clips(1)[0], "motion", np.random.default_rng(0))
    _, neutral = build_decision(v, CANDS, seed=1)
    _, informed = build_decision(v, CANDS, seed=1, style="informed")
    assert list(neutral) == list(informed)
    assert all(informed[c].startswith(neutral[c] + ": ") for c in CANDS)
