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
                                Without a key, an Entra ID token from `az login` is used.
  vertex:<model>                Gemini on Vertex AI (generateContent), token from `gcloud`.
                                The only route that takes video (Request.videos, mp4).
                                env VERTEX_PROJECT (+ VERTEX_LOCATION, default "global")
  vertex-anthropic:<model>      Claude on Vertex AI (rawPredict, Messages body), same env.
  openai:<model>, anthropic:<model>   direct vendor APIs (OPENAI_API_KEY, ANTHROPIC_API_KEY)
  dummy:<uniform|first|echo>    offline; for smoke tests only

The deployment name is whatever you called it in Foundry (by default the model id,
e.g. "claude-opus-5-5" or "gpt-5.6-sol").
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

TIMEOUT = 180
RETRIES = 5
# Reasoning models (gpt-5.x, Gemini 2.5+/3.x) count hidden thinking against the output
# limit; without headroom the visible JSON is cut off mid-answer.
REASONING_HEADROOM = int(os.environ.get("MOTION_SPORT_REASONING_HEADROOM", "16384"))


class BackendError(RuntimeError):
    pass


_LOCAL = threading.local()


def last_usage() -> dict:
    """Token usage of this thread's last successful call (provider's own field names)."""
    return getattr(_LOCAL, "usage", {}) or {}


@dataclass
class Request:
    prompt: str
    images: list[bytes] = field(default_factory=list)  # PNG bytes, in display order
    max_tokens: int = 1024
    temperature: float = 0.0
    videos: list[bytes] = field(default_factory=list)  # mp4 bytes; only VIDEO_ROUTES take them
    video_fps: float | None = None  # frame rate the model should sample the videos at


def _post(url: str, headers: dict, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(RETRIES):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
                _LOCAL.usage = data.get("usage") or data.get("usageMetadata") or {}
                return data
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


_TOKENS: dict[str, tuple[str, float]] = {}


def _cli_token(name: str, cmd: list[str]) -> str:
    """Bearer token from a logged-in CLI (az / gcloud), cached for 30 min."""
    tok, at = _TOKENS.get(name, ("", 0.0))
    if tok and time.time() - at < 1800:
        return tok
    exe = shutil.which(cmd[0]) or os.environ.get(f"{cmd[0].upper()}_BIN")
    if not exe:
        raise BackendError(f"{cmd[0]} not found (set {cmd[0].upper()}_BIN) and no API key set")
    try:
        tok = subprocess.run([exe, *cmd[1:]], check=True, capture_output=True,
                             text=True, timeout=60).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise BackendError(f"{cmd[0]} token failed (log in again?): {e.stderr[:300]}") from None
    _TOKENS[name] = (tok, time.time())
    return tok


_GCLOUD = ["gcloud", "auth", "print-access-token"]


def _post_gcloud(url: str, payload: dict, post=None) -> dict:
    """POST with a gcloud bearer token; on 401 (token expired mid-run) refresh once."""
    post = post or (lambda u, h, p: _post(u, h, p))
    try:
        return post(url, {"Authorization": f"Bearer {_cli_token('gcloud', _GCLOUD)}"}, payload)
    except BackendError as e:
        if "HTTP 401" not in str(e):
            raise
        _TOKENS.pop("gcloud", None)
        return post(url, {"Authorization": f"Bearer {_cli_token('gcloud', _GCLOUD)}"}, payload)


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
    limit = req.max_tokens + (REASONING_HEADROOM if token_field == "max_completion_tokens" else 0)
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               token_field: limit}
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


def _messages_payload(model: str | None, req: Request) -> dict:
    content: list[dict] = []
    for png in req.images:
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": _b64(png)}})
    content.append({"type": "text", "text": req.prompt})
    # Claude 5.x thinks adaptively and counts thinking against max_tokens; without
    # headroom a long thought leaves an empty answer (D21: Sonnet 5 motion/text, 36/400)
    payload = {"max_tokens": req.max_tokens + REASONING_HEADROOM, "temperature": req.temperature,
               "messages": [{"role": "user", "content": content}]}
    if model:
        payload["model"] = model
    return payload


def _post_messages(url: str, headers: dict, payload: dict, model: str) -> dict:
    """POST a Messages body; drop `temperature` for models that reject it (Claude 5.x)."""
    if model in _FIXED_TEMPERATURE:
        payload.pop("temperature", None)
    try:
        return _post(url, headers, payload)
    except BackendError as e:
        if "temperature" not in str(e) or "temperature" not in payload:
            raise
        print(f"  ! {model} rejects temperature; using its default from now on", file=sys.stderr)
        _FIXED_TEMPERATURE.add(model)
        payload.pop("temperature")
        return _post(url, headers, payload)


def _messages_text(data: dict) -> str:
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _messages(url: str, headers: dict, model: str, req: Request) -> str:
    data = _post_messages(url, {**headers, "anthropic-version": "2023-06-01"},
                          _messages_payload(model, req), model)
    return _messages_text(data)


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
    key = os.environ.get("AZURE_ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
    if key:
        auth = {"x-api-key": key}
    else:  # Entra ID, as anthropic.AnthropicFoundry does with DefaultAzureCredential
        auth = {"Authorization": "Bearer " + _cli_token("az", [
            "az", "account", "get-access-token", "--resource",
            "https://cognitiveservices.azure.com", "--query", "accessToken", "-o", "tsv"])}
    return _messages(f"{_foundry_anthropic_base()}/v1/messages", auth, model, req)


def _anthropic(model: str, req: Request) -> str:
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    return _messages(f"{base}/v1/messages", {"x-api-key": _env("ANTHROPIC_API_KEY")}, model, req)


# ------------------------------------------------------------------ Vertex (Gemini)


def _vertex_base(publisher: str, model: str) -> str:
    project = _env("VERTEX_PROJECT")
    loc = os.environ.get("VERTEX_LOCATION", "global")
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{loc}/publishers/{publisher}/models/{model}"


def _vertex_anthropic(model: str, req: Request) -> str:
    payload = {"anthropic_version": "vertex-2023-10-16", **_messages_payload(None, req)}
    data = _post_gcloud(f"{_vertex_base('anthropic', model)}:rawPredict", payload,
                        lambda u, h, p: _post_messages(u, h, p, model))
    return _messages_text(data)


def _vertex(model: str, req: Request) -> str:
    url = f"{_vertex_base('google', model)}:generateContent"
    parts: list[dict] = []
    for mp4 in req.videos:
        part: dict = {"inlineData": {"mimeType": "video/mp4", "data": _b64(mp4)}}
        if req.video_fps:  # Gemini samples 1 fps by default: fewer frames than the clip has
            part["videoMetadata"] = {"fps": req.video_fps}
        parts.append(part)
    parts += [{"inlineData": {"mimeType": "image/png", "data": _b64(png)}} for png in req.images]
    parts.append({"text": req.prompt})
    payload = {"contents": [{"role": "user", "parts": parts}],
               "generationConfig": {"temperature": req.temperature,
                                    "maxOutputTokens": req.max_tokens + REASONING_HEADROOM}}
    data = _post_gcloud(url, payload)
    cands = data.get("candidates") or [{}]
    return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", [])
                   if not p.get("thought"))


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


# routes that accept video input; every other route refuses a request with videos
VIDEO_ROUTES = {"vertex", "dummy"}

ROUTES = {
    "azure-openai": _azure_openai,
    "azure-anthropic": _azure_anthropic,
    "azure-foundry": _azure_foundry,
    "openai": _openai,
    "anthropic": _anthropic,
    "vertex": _vertex,
    "vertex-anthropic": _vertex_anthropic,
    "dummy": _dummy,
}


def complete(model_id: str, req: Request) -> str:
    if ":" not in model_id:
        raise BackendError(f"model id must be '<route>:<deployment>', got {model_id!r}")
    route, model = model_id.split(":", 1)
    if route not in ROUTES:
        raise BackendError(f"unknown route {route!r}; known: {sorted(ROUTES)}")
    if req.videos and route not in VIDEO_ROUTES:
        raise BackendError(f"route {route!r} does not accept video; video routes: "
                           f"{sorted(VIDEO_ROUTES)}")
    return ROUTES[route](model, req)
