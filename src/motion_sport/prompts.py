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
    ("kinematics", "trails"): "Each player's path over {span:.1f} s has been moved to its own spot "
                              "on a neutral grid, so the team shape is destroyed; the fading line "
                              "shows each individual's movement, ending at the dot.",
}

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


def build_prompt(view: View, repr_kind: str, candidates: list[str], *, seed: int,
                 text_payload: str | None = None) -> tuple[str, list[str]]:
    """Return (prompt, shown_order). `repr_kind` in {"image", "trails", "text"}."""
    order = candidate_order(candidates, seed)
    k = len(view.frames)
    span = max(view.frame_times) - min(view.frame_times) if k > 1 else 0.0
    dt = span / (k - 1) if k > 1 else 0.0
    key_repr = "image" if repr_kind in ("image", "text") else repr_kind
    what = _WHAT.get((view.condition, key_repr))
    if what is None:
        raise ValueError(f"no prompt for condition={view.condition} repr={repr_kind}")
    what = what.format(k=k, dt=dt, span=span)
    if repr_kind == "text":
        what = what.replace("snapshots", "snapshots (given below as coordinates)")
        what = what.replace("panel 1", "snapshot 1")
    parts = [_PREAMBLE, what]
    if text_payload:
        parts.append(text_payload)
    parts.append(_ASK.format(options=", ".join(f'"{SPORTS[c]}"' for c in order)))
    return "\n\n".join(parts), order


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
