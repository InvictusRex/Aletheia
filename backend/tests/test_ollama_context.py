import httpx
import pytest

from app.llm.ollama import OllamaFactsProvider, is_context_overflow
from app.llm.provider import LLMTransportError


class _CtxResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = body

    def raise_for_status(self):
        request = httpx.Request("POST", "http://test.invalid/api/chat")
        response = httpx.Response(
            self.status_code, content=self._body.encode(), request=request
        )
        raise httpx.HTTPStatusError(
            f"HTTP {self.status_code}", request=request, response=response
        )

    def json(self):
        raise AssertionError("must not parse a failed response")


class _CtxClient:
    def __init__(self, state, **kwargs):
        self._state = state
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
        pass


def _install(monkeypatch, action):
    state = {"action": action}
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: _CtxClient(state, **kwargs)
    )
    return state


_CONTEXT_BODY = (
    '{"error": {"code": 400, "message": "request (7045 tokens) exceeds '
    'the available context size (4096 tokens), try increasing it", '
    '"type": "exceed_context_size_error"}}'
)


def test_is_context_overflow_matches_deterministic_errors():
    assert is_context_overflow(Exception(_CONTEXT_BODY)) is True
    assert is_context_overflow(Exception("context length exceeded")) is True
    assert is_context_overflow(Exception("request too large for context")) is True
    assert is_context_overflow(Exception("429 RESOURCE_EXHAUSTED")) is False
    assert is_context_overflow(Exception("tokens per day (TPD): Limit 1")) is False
    assert is_context_overflow(Exception("timeout")) is False
    assert is_context_overflow(Exception("")) is False


def test_context_overflow_fails_fast_without_retry(monkeypatch):
    state = _install(monkeypatch, _CtxResponse(400, _CONTEXT_BODY))
    provider = OllamaFactsProvider(
        max_retries=5, backoff_base_s=0.0, backoff_max_s=0.0
    )
    with pytest.raises(LLMTransportError) as excinfo:
        provider.extract_facts("extract facts about X")
    assert len(state["posts"]) == 1
    message = str(excinfo.value).lower()
    assert "context" in message
    assert "not retried" in message


def test_plain_500_still_retries(monkeypatch):
    calls = []

    def fail():
        calls.append(1)
        return _CtxResponse(500, '{"error": "boom"}')

    state = _install(monkeypatch, fail)
    provider = OllamaFactsProvider(
        max_retries=2, backoff_base_s=0.0, backoff_max_s=0.0
    )
    with pytest.raises(LLMTransportError) as excinfo:
        provider.extract_facts("extract facts about X")
    assert len(state["posts"]) == 3
    assert "context" not in str(excinfo.value).lower()
