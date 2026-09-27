"""Prompt construction and answer parsing for the closed-set sport question.

Design choices that matter for the measurement:
- The candidate list is closed and **shuffled per item** (seeded), so a model's
  preference for the first option cannot masquerade as knowledge.
- The prompt says, truthfully, what was removed and that dot count / scale /
  orientation are meaningless. Without that, a model spends its effort counting
  dots — a shortcut the controls already destroyed — and the error looks like
  "cannot read motion" when it is really "was told nothing".
- The answer is a probability per option, not just a label, so we get log-loss
  and calibration, and can tell a confident mistake from a coin flip.
- Two prompt styles. "neutral" names the sports and nothing else. "informed" adds a
  short description of how players typically move in each option, answering the
  objection "you never told it what to look for". The descriptions are symmetric
  (similar length and structure: rhythm, collective shape, a distinctive pattern),
  general knowledge only, and carry no numbers or facts about our data.
"""
from __future__ import annotations

import json
import re

import numpy as np

from motion_sport.conditions import View
from motion_sport.schema import SPORTS

_PREAMBLE = (
    "You are looking at player movement from a team sport, seen from directly above. "
    "Each dot is one player. The ball, the field markings, the playing surface, "
    "the team colours and any equipment have all been removed. Positions have been "
    "re-centred, rescaled and rotated/reflected at random, so absolute size and "
    "direction carry no information, and only a fixed subset of the players is "
    "shown, so the number of dots is NOT the number of players in the sport."
)

_WHAT = {
    ("formation", "image"): "You are shown a single instant.",
    ("motion", "image"): "You are shown {k} snapshots of the same play, {dt:.1f} s apart, in "
                         "chronological order (panel 1 is the earliest).",
    ("motion_shuffled", "image"): "You are shown {k} snapshots of the same play, taken over "
                                  "{span:.1f} s, in an unknown order.",
    ("motion", "trails"): "You are shown one image: each dot is a player's final position and "
                          "the fading line behind it is the path over the previous {span:.1f} s.",
    ("kinematics", "image"): "Each player's path has been moved to its own spot on a neutral "
                             "grid, so the team shape is destroyed and only each individual's "
                             "movement over {span:.1f} s remains ({k} snapshots in chronological "
                             "order, panel 1 earliest).",
    ("kinematics_solo", "image"): "Each player's path has been moved to its own spot on a "
                                  "neutral grid AND rotated by its own random angle, so both the "
                                  "team shape and any shared running direction are destroyed; "
                                  "only each individual's movement rhythm over {span:.1f} s "
                                  "remains ({k} snapshots in chronological order, panel 1 earliest).",
    ("kinematics_solo", "trails"): "Each player's path over {span:.1f} s has been moved to its own "
                                   "spot on a neutral grid AND rotated by its own random angle, so "
                                   "team shape and shared direction are destroyed; the fading line "
                                   "shows each individual's movement, ending at the dot.",
    ("motion", "video"): "You are shown a short silent video of the play: {k} frames over "
                         "{span:.1f} s ({fps:.0f} frames per second), in chronological order.",
    ("motion_shuffled", "video"): "You are shown a short silent video made of {k} snapshots of "
                                  "the same play, taken over {span:.1f} s but played in a "
                                  "shuffled, unknown order ({fps:.0f} frames per second).",
    ("kinematics", "video"): "You are shown a short silent video ({k} frames over {span:.1f} s, "
                             "{fps:.0f} frames per second, in chronological order). Each "
                             "player's path has been moved to its own spot on a neutral grid, "
                             "so the team shape is destroyed and only each individual's "
                             "movement remains.",
    ("kinematics_solo", "video"): "You are shown a short silent video ({k} frames over "
                                  "{span:.1f} s, {fps:.0f} frames per second, in chronological "
                                  "order). Each player's path has been moved to its own spot on "
                                  "a neutral grid AND rotated by its own random angle, so both "
                                  "the team shape and any shared running direction are "
                                  "destroyed; only each individual's movement rhythm remains.",
    ("kinematics", "trails"): "Each player's path over {span:.1f} s has been moved to its own spot "
                              "on a neutral grid, so the team shape is destroyed; the fading line "
                              "shows each individual's movement, ending at the dot.",
}

PROMPT_STYLES = ("neutral", "informed")

# How players move in each registered sport. Every entry: rhythm, collective shape,
# one distinctive pattern; no numbers, no player counts (the controls fix those), no
# field sizes (removed by the controls), nothing taken from our datasets.
MOVEMENT: dict[str, str] = {
    "soccer": "Continuous, flowing play: long spells of jogging and walking broken by "
              "occasional sprints. The team keeps a wide, loosely structured shape that "
              "shifts across the field as a block, its lines stretching and compressing "
              "as possession changes.",
    "basketball": "Continuous play in a tight space with constant stop-start movement: "
                  "short bursts, sharp cuts and quick changes of direction. Players gather "
                  "around one end at a time, circling and screening each other, then all "
                  "run to the other end together.",
    "handball": "Continuous play with fast transitions: players sprint together from one "
                "end to the other, then settle into a set arrangement in which attackers "
                "move side to side along an arc facing a compact defensive line that "
                "shuffles laterally.",
    "american_football": "Play comes in short, separate bursts. Players stand almost still "
                         "in two facing lines, then all start moving at the same instant; "
                         "some run long straight or angled routes while others collide near "
                         "the starting line, and the action stops abruptly.",
    "rugby_union": "Mostly continuous play in which players repeatedly converge into tight, "
                   "nearly static clusters, then spread out into staggered, flat lines that "
                   "advance or retreat across the field together, with frequent short "
                   "sprints into contact.",
    "rugby_sevens": "Continuous, very open play with players spread thinly across a large "
                    "space. Long sprints and wide sweeping runs dominate; brief tight "
                    "clusters form around tackles and quickly break up into fast, stretched "
                    "attacking lines.",
    "rugby_league": "Continuous play in repeated short sets: after each tackle the players "
                    "quickly reset into two roughly flat, opposing lines with a gap between "
                    "them, which then advance and retreat together as attackers run straight "
                    "at the defence.",
    "field_hockey": "Continuous play across a large space with frequent short accelerations "
                    "and quick changes of direction. The team keeps a spread-out, structured "
                    "shape that slides as a block, with small groups forming moving "
                    "triangles around the play.",
    "ice_hockey": "Continuous, very fast play in an enclosed space. Movement is smooth and "
                  "gliding, with wide curved paths, long coasting phases and loops behind "
                  "the play; groups rush together towards one end and then sweep back the "
                  "other way.",
    "futsal": "Continuous play in a small space with constant rotation: players swap "
              "positions in fluid patterns, with short sprints and quick changes of "
              "direction. The team keeps a compact diamond or square shape that moves "
              "together as possession changes.",
}

_GUIDE = ("How players typically move in each option (general descriptions of each sport, "
          "not of this clip):\n{lines}")

_ASK = (
    "Which sport is this? Choose among exactly these options: {options}.\n"
    "Reply with JSON only, no prose around it:\n"
    '{{"probabilities": {{<option>: <probability>, ...}}, "answer": <option>, '
    '"rationale": <one or two sentences on the cues you used>}}\n'
    "Probabilities must cover every option and sum to 1."
)


def candidate_order(candidates: list[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [candidates[i] for i in rng.permutation(len(candidates))]


def describe_view(view: View, repr_kind: str, where: str = "given below") -> str:
    """The shared description: what was removed, and what this view shows."""
    k = len(view.frames)
    span = max(view.frame_times) - min(view.frame_times) if k > 1 else 0.0
    dt = span / (k - 1) if k > 1 else 0.0
    key_repr = "image" if repr_kind in ("image", "text") else repr_kind
    what = _WHAT.get((view.condition, key_repr))
    if what is None:
        raise ValueError(f"no prompt for condition={view.condition} repr={repr_kind}")
    what = what.format(k=k, dt=dt, span=span, fps=(k - 1) / span if span else 0.0)
    if repr_kind == "text":
        what = what.replace("snapshots", f"snapshots ({where} as coordinates)")
        what = what.replace("panel 1", "snapshot 1")
        if view.condition == "formation":
            what += f" Player positions are {where} as coordinates."
    return f"{_PREAMBLE}\n\n{what}"


def movement_guide(order: list[str]) -> str:
    """The informed prompt's extra paragraph, one line per option in the shown order."""
    missing = [c for c in order if c not in MOVEMENT]
    if missing:
        raise ValueError(f"no movement description for {missing}: add it to prompts.MOVEMENT")
    return _GUIDE.format(lines="\n".join(f"- {SPORTS[c]}: {MOVEMENT[c]}" for c in order))


def _check_style(style: str) -> None:
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}; known: {PROMPT_STYLES}")


def build_prompt(view: View, repr_kind: str, candidates: list[str], *, seed: int,
                 text_payload: str | None = None, style: str = "neutral") -> tuple[str, list[str]]:
    """Chat prompt. Return (prompt, shown_order). `repr_kind` in {"image", "trails",
    "text", "video"}. Both styles show the options in the same seeded order."""
    _check_style(style)
    order = candidate_order(candidates, seed)
    parts = [describe_view(view, repr_kind)]
    if style == "informed":
        parts.append(movement_guide(order))
    if text_payload:
        parts.append(text_payload)
    parts.append(_ASK.format(options=", ".join(f'"{SPORTS[c]}"' for c in order)))
    return "\n\n".join(parts), order


def build_decision(view: View, candidates: list[str], *, seed: int,
                   style: str = "neutral") -> tuple[str, dict[str, str]]:
    """Instructions + criteria for typed-decision models (Jev, Laya); the coordinates
    go in the separate `state` field. Neutral criteria are just the sport names; the
    informed ones add the same movement descriptions as the chat prompt."""
    _check_style(style)
    order = candidate_order(candidates, seed)
    instructions = describe_view(view, "text", where="given in the input") + \
        "\n\nWhich team sport is this?"
    if style == "informed":
        return instructions, {c: f"{SPORTS[c]}: {MOVEMENT[c]}" for c in order}
    return instructions, {c: SPORTS[c] for c in order}


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _label_of(name: str, candidates: list[str]) -> str | None:
    n = name.strip().lower()
    for c in candidates:
        if n in (c.lower(), SPORTS[c].lower()):
            return c
    for c in candidates:  # lenient: "soccer" -> "association football (soccer)"
        disp = SPORTS[c].lower()
        if n and (n in disp or disp in n or n.replace(" ", "_") == c):
            return c
    return None


def parse_answer(raw: str, candidates: list[str]) -> dict:
    """-> {"label": id|None, "probs": {id: p}, "rationale": str, "error": str|None}."""
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return {"label": None, "probs": {}, "rationale": "", "error": "no JSON"}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"label": None, "probs": {}, "rationale": "", "error": f"bad JSON: {e}"}
    probs: dict[str, float] = {}
    for k, v in (data.get("probabilities") or {}).items():
        lab = _label_of(str(k), candidates)
        try:
            p = float(v)
        except (TypeError, ValueError):
            continue
        if lab and p >= 0:
            probs[lab] = probs.get(lab, 0.0) + p
    total = sum(probs.values())
    probs = {c: probs.get(c, 0.0) / total for c in candidates} if total > 0 else {}
    label = _label_of(str(data.get("answer", "")), candidates)
    if label is None and probs:
        label = max(probs, key=probs.get)
    return {"label": label, "probs": probs, "rationale": str(data.get("rationale", ""))[:500],
            "error": None if label else "unparseable answer"}
