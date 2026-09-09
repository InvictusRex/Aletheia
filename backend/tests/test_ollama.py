"""Deterministic tests for the Ollama provider + selection factory.

Scope: configuration, provider selection, HTTP request construction,
and error mapping. The HTTP layer (``httpx.Client``) is stubbed with
in-memory fakes — no network, no Ollama server, and no model. Real
inference validation happens separately against a live server, never
here. No Groq behavior is faked anywhere in this module.
"""

import uuid

import httpx
import pytest

from app.core.config import Settings, settings
from app.llm.ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_MAX_OUTPUT_TOKENS,
    OLLAMA_TEMPERATURE,
    OllamaFactsProvider,
    OllamaJudgmentTransport,
)
from app.llm.provider import (
    LLMMalformedError,
    LLMProvider,
    LLMTimeoutError,
    LLMTransportError,
)


def _chat_payload(content, prompt_n=10, eval_n=5):
    return {
        "model": "qwen3.5:9b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_n,
        "eval_count": eval_n,
    }


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else str(payload)[:200]

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request(
                "POST", "http://test.invalid/api/chat"
            )
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class _FakeClient:
    """Stand-in for ``httpx.Client``; behavior set per test."""

    def __init__(self, state, **kwargs):
        self._state = state
        self._state["init_kwargs"] = kwargs
        self._state.setdefault("posts", [])

    def post(self, path, json=None):
        self._state["posts"].append({"path": path, "json": json})
        action = self._state.get("action")
        if callable(action):
            return action()
        if isinstance(action, BaseException):
            raise action
        return action

    def close(self):
        self._state["closed"] = True


def _install_fake_http(monkeypatch, action):
    state = {"action": action}
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: _FakeClient(state, **kwargs)
    )
    return state


def _valid_draft_payload():
    eid = str(uuid.uuid4())
    return _chat_payload(
        '{"drafts": [{"subject": "Revenue", "predicate": "total", '
        f'"value_text": "900", "evidence_ids": ["{eid}"], '
        '"confidence": 0.9}]}'
    )


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


def test_ollama_config_defaults(monkeypatch, tmp_path):
    for var in (
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_THINK",
        "OLLAMA_TIMEOUT_S",
        "OLLAMA_MAX_RETRIES",
        "OLLAMA_BACKOFF_BASE_S",
        "OLLAMA_BACKOFF_MAX_S",
        "OLLAMA_KEEP_ALIVE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    assert DEFAULT_OLLAMA_BASE_URL == "http://host.docker.internal:11434"
    fresh = Settings()
    assert fresh.llm_provider == "groq"  # Groq stays the default
    assert fresh.ollama_base_url == DEFAULT_OLLAMA_BASE_URL
    assert fresh.ollama_model == DEFAULT_OLLAMA_MODEL
    assert fresh.ollama_think is None
    assert OLLAMA_TEMPERATURE == 0
    assert OLLAMA_MAX_OUTPUT_TOKENS == 1200


# ---------------------------------------------------------------------------
# Provider selection factory
# ---------------------------------------------------------------------------


def test_factory_selects_groq(monkeypatch):
    from app.facts.service import get_llm_provider
    from app.llm.groq import GroqFactsProvider

    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "construct-only-key")
    provider = get_llm_provider()
    assert isinstance(provider, GroqFactsProvider)
    assert isinstance(provider, LLMProvider)


def test_factory_selects_ollama_without_api_key(monkeypatch):
    from app.facts.service import get_llm_provider

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama-test:11434")
    monkeypatch.setattr(settings, "ollama_model", "qwen3.5:9b")
    provider = get_llm_provider()
    assert isinstance(provider, OllamaFactsProvider)
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "qwen3.5:9b"


def test_factory_rejects_unknown_provider(monkeypatch):
    from app.facts.service import get_llm_provider

    monkeypatch.setattr(settings, "llm_provider", "watson")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_llm_provider()


def test_judgment_factory_selects_ollama(monkeypatch):
    from app.matching.service import get_judgment_transport

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama-test:11434")
    transport = get_judgment_transport()
    assert isinstance(transport, OllamaJudgmentTransport)


def test_judgment_factory_rejects_unknown_provider(monkeypatch):
    from app.matching.service import get_judgment_transport

    monkeypatch.setattr(settings, "llm_provider", "watson")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_judgment_transport()


# ---------------------------------------------------------------------------
# HTTP request construction
# ---------------------------------------------------------------------------


def test_chat_body_model_format_temperature_cap(monkeypatch):
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload('{"drafts": []}'))
    )
    provider = OllamaFactsProvider(
        base_url="http://ollama-test:11434/",
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )
    batch = provider.extract_facts("extract facts about X")
    assert batch.model == "qwen3.5:9b"
    assert batch.drafts == []
    assert state["init_kwargs"]["base_url"] == "http://ollama-test:11434"
    assert len(state["posts"]) == 1
    post = state["posts"][0]
    assert post["path"] == "/api/chat"
    body = post["json"]
    assert body["model"] == "qwen3.5:9b"
    assert body["stream"] is False
    assert body["format"] == "json"
    assert body["messages"] == [{"role": "user", "content": "extract facts about X"}]
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_predict"] == OLLAMA_MAX_OUTPUT_TOKENS
    assert "think" not in body  # omitted unless explicitly configured
    assert state["closed"] is True


def test_think_passthrough_when_configured(monkeypatch):
    for think in (True, False):
        state = _install_fake_http(
            monkeypatch, _FakeResponse(200, _chat_payload('{"drafts": []}'))
        )
        provider = OllamaFactsProvider(
            think=think, backoff_base_s=0.0, backoff_max_s=0.0
        )
        provider.extract_facts("prompt")
        assert state["posts"][0]["json"]["think"] is think


def test_validated_drafts_returned(monkeypatch):
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _valid_draft_payload())
    )
    provider = OllamaFactsProvider(backoff_base_s=0.0, backoff_max_s=0.0)
    batch = provider.extract_facts("prompt")
    assert len(batch.drafts) == 1
    assert batch.drafts[0].subject == "Revenue"
    assert state["posts"][0]["json"]["model"] == provider.model_name


def test_empty_prompt_rejected_without_http(monkeypatch):
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload('{"drafts": []}'))
    )
    provider = OllamaFactsProvider()
    with pytest.raises(ValueError, match="non-empty string"):
        provider.extract_facts("   ")
    assert state.get("posts", []) == []


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_connection_refused_fails_fast(monkeypatch):
    state = _install_fake_http(
        monkeypatch, httpx.ConnectError("connection refused")
    )
    provider = OllamaFactsProvider(
        base_url="http://ollama-test:11434",
        max_retries=5,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )
    with pytest.raises(LLMTransportError) as excinfo:
        provider.extract_facts("prompt")
    assert len(state["posts"]) == 1  # never retried
    assert "http://ollama-test:11434" in str(excinfo.value)


def test_timeout_retries_then_surfaces(monkeypatch):
    state = _install_fake_http(
        monkeypatch, httpx.TimeoutException("timed out")
    )
    provider = OllamaFactsProvider(max_retries=2)
    with pytest.raises(LLMTimeoutError):
        provider.extract_facts("prompt")
    assert len(state["posts"]) == 3  # 1 initial + 2 retries


def test_404_suggests_model_pull_without_retry(monkeypatch):
    state = _install_fake_http(monkeypatch, _FakeResponse(404, None))
    provider = OllamaFactsProvider(
        max_retries=5, backoff_base_s=0.0, backoff_max_s=0.0
    )
    with pytest.raises(LLMTransportError) as excinfo:
        provider.extract_facts("prompt")
    assert len(state["posts"]) == 1
    assert "ollama pull" in str(excinfo.value)


def test_500_retries_then_surfaces(monkeypatch):
    state = _install_fake_http(monkeypatch, _FakeResponse(500, None))
    provider = OllamaFactsProvider(
        max_retries=2, backoff_base_s=0.0, backoff_max_s=0.0
    )
    with pytest.raises(LLMTransportError):
        provider.extract_facts("prompt")
    assert len(state["posts"]) == 3


def test_malformed_content_surfaces_without_fabrication(monkeypatch):
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload("not json {{{"))
    )
    provider = OllamaFactsProvider(
        max_retries=1, backoff_base_s=0.0, backoff_max_s=0.0
    )
    with pytest.raises(LLMMalformedError):
        provider.extract_facts("prompt")
    assert len(state["posts"]) == 2


# ---------------------------------------------------------------------------
# Judgment transport
# ---------------------------------------------------------------------------


def test_judgment_complete_text_and_cap(monkeypatch):
    raw = '{"relationship_type": "RELATED", "confidence": 0.5}'
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload(raw))
    )
    transport = OllamaJudgmentTransport(backoff_base_s=0.0, backoff_max_s=0.0)
    assert transport.complete_text("judge this pair") == raw
    assert state["posts"][0]["json"]["options"]["num_predict"] == 800
    assert state["posts"][0]["json"]["format"] == "json"


def test_judgment_empty_text_surfaces(monkeypatch):
    from app.llm.provider import LLMTransportError

    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload("   "))
    )
    transport = OllamaJudgmentTransport(
        max_retries=1, backoff_base_s=0.0, backoff_max_s=0.0
    )
    with pytest.raises(LLMTransportError):
        transport.complete_text("judge this pair")
    assert len(state["posts"]) == 2


def test_keep_alive_defaults_to_resident(monkeypatch):
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload('{"drafts": []}'))
    )
    provider = OllamaFactsProvider(backoff_base_s=0.0, backoff_max_s=0.0)
    provider.extract_facts("prompt")
    assert state["posts"][0]["json"]["keep_alive"] == -1
    transport = OllamaJudgmentTransport(backoff_base_s=0.0, backoff_max_s=0.0)
    transport.complete_text("judge this pair")
    assert state["posts"][1]["json"]["keep_alive"] == -1


def test_keep_alive_passthrough_when_configured(monkeypatch):
    state = _install_fake_http(
        monkeypatch, _FakeResponse(200, _chat_payload('{"drafts": []}'))
    )
    provider = OllamaFactsProvider(
        keep_alive=0, backoff_base_s=0.0, backoff_max_s=0.0
    )
    provider.extract_facts("prompt")
    assert state["posts"][0]["json"]["keep_alive"] == 0
