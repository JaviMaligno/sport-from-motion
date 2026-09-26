"""Typed-decision models: Jev (TypeSafe) and Laya (Convai). Text in, calibrated choice out.

These are not chat models. They take a *state* (text or JSON) plus typed questions
and return, in one forward pass, a choice with a probability per option — exactly
the output this experiment needs, with no text to parse. They read text only, so
they are run on the `text` representation (coordinates serialised as text).

Both speak the same wire protocol, TypeSafe's `POST /v1/systemone`:

    {"state": ..., "model": ..., "questions": {"sport": {"type": "choice",
        "instructions": "...", "criteria": {"soccer": "...", ...}}}}
    -> {"answers": {"sport": {"choice": "soccer", "probabilities": {...}, ...}}, "usage": {...}}

Routes:
  jev:<model>          TypeSafe API. env TYPESAFE_API_KEY (+ TYPESAFE_BASE_URL,
                       default https://api.typesafe.ai/v1). Model e.g. "jev-latest".
  laya-http:<model>    a `laya-serve` server (same protocol): local, or deployed as a
                       container on Azure. env LAYA_BASE_URL (e.g. http://localhost:8000/v1),
                       optional LAYA_API_KEY. Model: english | multilingual | typed-decisions.
  laya:<checkpoint>    Laya in-process via `pip install laya` (needs torch). Same
                       checkpoint names. env LAYA_MAX_LEN (default 8192).

Context limits matter here: Laya's English checkpoint reads 512 tokens and the
multilingual one up to 8,192 with max_len=8192; Jev reads 32k. The serialisation
(serialize.py) is compact on purpose, and every prediction records the input
token count the server reports, so truncation shows up in the data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from motion_sport.backends.chat import BackendError, _env, _post

QUESTION = "sport"


@dataclass
class Decision:
    state: str | dict
    instructions: str
    criteria: dict[str, str]  # option id -> description, in the order shown


def _questions(d: Decision) -> dict:
    return {QUESTION: {"type": "choice", "instructions": d.instructions, "criteria": d.criteria}}


def _systemone(base: str, key: str | None, model: str, d: Decision) -> dict:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return _post(f"{base.rstrip('/')}/systemone", headers,
                 {"state": d.state, "model": model, "questions": _questions(d)})


def _jev(model: str, d: Decision) -> dict:
    base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
    return _systemone(base, _env("TYPESAFE_API_KEY"), model, d)


def _laya_http(model: str, d: Decision) -> dict:
    return _systemone(_env("LAYA_BASE_URL"), os.environ.get("LAYA_API_KEY"), model, d)


_ROUTER = None


def _laya_local(model: str, d: Decision) -> dict:
    global _ROUTER
    if _ROUTER is None:
        try:
            from laya import Router
        except ImportError as e:
            raise BackendError("laya is not installed: pip install -e '.[laya]'") from e
        _ROUTER = Router()
    max_len = int(os.environ.get("LAYA_MAX_LEN", "8192"))
    return _ROUTER.predict(d.state, _questions(d), model=model, max_len=max_len)


ROUTES = {"jev": _jev, "laya-http": _laya_http, "laya": _laya_local}


def is_decision_model(model_id: str) -> bool:
    return model_id.split(":", 1)[0] in ROUTES


def decide(model_id: str, d: Decision) -> dict:
    """-> {"label", "probs", "confidence", "input_tokens", "raw"} (label None on failure)."""
    route, model = model_id.split(":", 1)
    resp = ROUTES[route](model, d)
    ans = (resp.get("answers") or {}).get(QUESTION) or {}
    probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items() if k in d.criteria}
    total = sum(probs.values())
    probs = {k: probs.get(k, 0.0) / total for k in d.criteria} if total > 0 else {}
    label = ans.get("choice")
    if label not in d.criteria:
        label = max(probs, key=probs.get) if probs else None
    usage = resp.get("usage") or {}
    return {"label": label, "probs": probs, "confidence": ans.get("confidence"),
            "input_tokens": usage.get("input_tokens"),
            "error": None if label else "no choice in response", "raw": str(resp)[:2000]}
