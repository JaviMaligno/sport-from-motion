import pytest

from motion_sport.backends import chat


@pytest.fixture()
def captured(monkeypatch):
    seen = {}

    def fake_post(url, headers, payload):
        seen.update(url=url, headers=headers, payload=payload)
        if "/anthropic/" in url:
            return {"content": [{"type": "text", "text": "{}"}]}
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(chat, "_post", fake_post)
    for k in ("AZURE_ANTHROPIC_ENDPOINT", "ANTHROPIC_FOUNDRY_BASE_URL", "ANTHROPIC_FOUNDRY_RESOURCE"):
        monkeypatch.delenv(k, raising=False)
    return seen


REQ = chat.Request("which sport?", [b"\x89PNG fake"])


def test_azure_anthropic_uses_native_messages_route(captured, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "myres")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "k")
    chat.complete("azure-anthropic:claude-opus-5-5", REQ)
    assert captured["url"] == "https://myres.services.ai.azure.com/anthropic/v1/messages"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[-1]["type"] == "text"
    assert captured["payload"]["model"] == "claude-opus-5-5"


def test_azure_openai_uses_v1_route_with_data_url(captured, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "k")
    chat.complete("azure-openai:gpt-5.4", REQ)
    assert captured["url"] == "https://r.openai.azure.com/openai/v1/chat/completions"
    img = captured["payload"]["messages"][0]["content"][1]
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["headers"]["api-key"] == "k"


def test_azure_foundry_inference_route(captured, monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_KEY", "k")
    chat.complete("azure-foundry:llama4-maverick", REQ)
    assert captured["url"].startswith("https://x.services.ai.azure.com/models/chat/completions?api-version=")
    assert "max_tokens" in captured["payload"]


def test_bad_ids_fail_loudly():
    with pytest.raises(chat.BackendError):
        chat.complete("no-colon", REQ)
    with pytest.raises(chat.BackendError):
        chat.complete("nope:model", REQ)
