"""Multimodal chat backends, stdlib only. Azure first.

Model ids are "<route>:<deployment>", mirroring experiments/judge-bias/providers.py
(same env var names, so one shell setup serves both harnesses):

  azure-openai:<deployment>     GPT & any model on the OpenAI v1 route of a Foundry
                                resource (also xAI Grok).
                                env AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY
                                -> {endpoint}/openai/v1/chat/completions
  azure-anthropic:<deployment>  Claude on Microsoft Foundry (native Messages API).
                                env AZURE_ANTHROPIC_ENDPOINT, AZURE_ANTHROPIC_KEY
                                (or Anthropic's own ANTHROPIC_FOUNDRY_RESOURCE /
                                ANTHROPIC_FOUNDRY_BASE_URL + ANTHROPIC_FOUNDRY_API_KEY)
                                -> {endpoint}/anthropic/v1/messages
  azure-foundry:<deployment>    Other catalog models (Llama, Mistral, Qwen, Phi...) on
                                the Azure AI Model Inference route, as used by
                                wheres-the-ball/scripts/fase1_run_foundry.py.
                                env AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_KEY
                                -> {endpoint}/models/chat/completions
  openai:<model>, anthropic:<model>   direct vendor APIs (OPENAI_API_KEY, ANTHROPIC_API_KEY)
  dummy:<uniform|first|echo>    offline; for smoke tests only

The deployment name is whatever you called it in Foundry (by default the model id,
e.g. "claude-opus-5-5" or "gpt-5.4").
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

TIMEOUT = 180
RETRIES = 5


class BackendError(RuntimeError):
    pass


@dataclass
class Request:
    prompt: str
    images: list[bytes] = field(default_factory=list)  # PNG bytes, in display order
    max_tokens: int = 1024
    temperature: float = 0.0


def _post(url: str, headers: dict, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(RETRIES):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            last = BackendError(f"HTTP {e.code} from {url}: {detail}")
            if e.code < 500 and e.code != 429:
                raise last from None
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(min(float(wait), 60) if wait and wait.replace(".", "").isdigit() else 2 ** attempt)
            continue
        except (urllib.error.URLError, TimeoutError) as e:
            last = BackendError(f"network error calling {url}: {e}")
        time.sleep(2 ** attempt)
    raise last  # type: ignore[misc]


def _env(*names: str) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise BackendError(f"none of {names} is set")


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


# ------------------------------------------------------------------ OpenAI shape

_FIXED_TEMPERATURE: set[str] = set()


def _chat_completions(url: str, headers: dict, model: str, req: Request,
                      token_field: str = "max_completion_tokens") -> str:
    content: list[dict] = [{"type": "text", "text": req.prompt}]
    for png in req.images:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(png)}"}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               token_field: req.max_tokens}
    if req.temperature != 1.0 and model not in _FIXED_TEMPERATURE:
        payload["temperature"] = req.temperature
    try:
        data = _post(url, headers, payload)
    except BackendError as e:
        # reasoning models (gpt-5.x) accept only their default temperature
        if "temperature" not in str(e) or "temperature" not in payload:
            raise
        print(f"  ! {model} rejects temperature; using its default from now on", file=sys.stderr)
        _FIXED_TEMPERATURE.add(model)
        payload.pop("temperature")
        data = _post(url, headers, payload)
    return data["choices"][0]["message"].get("content") or ""


def _azure_openai(model: str, req: Request) -> str:
    base = _env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    return _chat_completions(f"{base}/openai/v1/chat/completions",
                             {"api-key": _env("AZURE_OPENAI_KEY")}, model, req)


def _azure_foundry(model: str, req: Request) -> str:
    base = _env("AZURE_FOUNDRY_ENDPOINT").rstrip("/")
    version = os.environ.get("AZURE_FOUNDRY_API_VERSION", "2024-05-01-preview")
    return _chat_completions(f"{base}/models/chat/completions?api-version={version}",
                             {"api-key": _env("AZURE_FOUNDRY_KEY")}, model, req,
                             token_field="max_tokens")


def _openai(model: str, req: Request) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return _chat_completions(f"{base}/chat/completions",
                             {"Authorization": f"Bearer {_env('OPENAI_API_KEY')}"}, model, req)


# --------------------------------------------------------------- Anthropic shape


def _messages(url: str, headers: dict, model: str, req: Request) -> str:
    content: list[dict] = []
    for png in req.images:
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": _b64(png)}})
    content.append({"type": "text", "text": req.prompt})
    payload = {"model": model, "max_tokens": req.max_tokens, "temperature": req.temperature,
               "messages": [{"role": "user", "content": content}]}
    data = _post(url, {**headers, "anthropic-version": "2023-06-01"}, payload)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _foundry_anthropic_base() -> str:
    if os.environ.get("AZURE_ANTHROPIC_ENDPOINT"):
        return os.environ["AZURE_ANTHROPIC_ENDPOINT"].rstrip("/") + "/anthropic"
    if os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL"):
        return os.environ["ANTHROPIC_FOUNDRY_BASE_URL"].rstrip("/")
    if os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"):
        return f"https://{os.environ['ANTHROPIC_FOUNDRY_RESOURCE']}.services.ai.azure.com/anthropic"
    raise BackendError("set AZURE_ANTHROPIC_ENDPOINT, ANTHROPIC_FOUNDRY_BASE_URL or "
                       "ANTHROPIC_FOUNDRY_RESOURCE")


def _azure_anthropic(model: str, req: Request) -> str:
    key = _env("AZURE_ANTHROPIC_KEY", "ANTHROPIC_FOUNDRY_API_KEY")
    return _messages(f"{_foundry_anthropic_base()}/v1/messages", {"x-api-key": key}, model, req)


def _anthropic(model: str, req: Request) -> str:
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    return _messages(f"{base}/v1/messages", {"x-api-key": _env("ANTHROPIC_API_KEY")}, model, req)


# ---------------------------------------------------------------------- offline


def _dummy(mode: str, req: Request) -> str:
    """Answers from the option list in the prompt; never looks at the images."""
    import re

    opts = re.findall(r'"([^"]+)"', req.prompt.rsplit("Choose among exactly these options:", 1)[-1]
                      .split("\n", 1)[0])
    if not opts:
        return "{}"
    if mode == "first":
        probs = {o: (1.0 if i == 0 else 0.0) for i, o in enumerate(opts)}
    else:
        probs = {o: 1 / len(opts) for o in opts}
    return json.dumps({"probabilities": probs, "answer": opts[0], "rationale": f"dummy:{mode}"})


ROUTES = {
    "azure-openai": _azure_openai,
    "azure-anthropic": _azure_anthropic,
    "azure-foundry": _azure_foundry,
    "openai": _openai,
    "anthropic": _anthropic,
    "dummy": _dummy,
}


def complete(model_id: str, req: Request) -> str:
    if ":" not in model_id:
        raise BackendError(f"model id must be '<route>:<deployment>', got {model_id!r}")
    route, model = model_id.split(":", 1)
    if route not in ROUTES:
        raise BackendError(f"unknown route {route!r}; known: {sorted(ROUTES)}")
    return ROUTES[route](model, req)
