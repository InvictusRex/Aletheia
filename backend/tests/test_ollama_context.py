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


class _OkResponse:
    status_code = 200
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": '{"drafts": []}'}}


def test_num_ctx_is_sent_so_ollama_does_not_truncate(monkeypatch):
    """Ollama defaults to its own small context unless num_ctx is sent.

    Without this the server silently drops the tail of an oversized
    prompt and the model returns unparseable JSON.
    """
    state = _install(monkeypatch, _OkResponse())
    provider = OllamaFactsProvider(model="m", context_tokens=16384)
    provider.extract_facts("prompt")

    options = state["posts"][0]["json"]["options"]
    assert options["num_ctx"] == 16384


def test_num_ctx_omitted_when_not_configured(monkeypatch):
    state = _install(monkeypatch, _OkResponse())
    provider = OllamaFactsProvider(model="m")
    provider.extract_facts("prompt")

    assert "num_ctx" not in state["posts"][0]["json"]["options"]


def test_judgment_transport_also_sends_num_ctx(monkeypatch):
    from app.llm.ollama import OllamaJudgmentTransport

    class _JudgeResponse(_OkResponse):
        def json(self):
            return {"message": {"content": "verdict"}}

    state = _install(monkeypatch, _JudgeResponse())
    OllamaJudgmentTransport(model="m", context_tokens=8192).complete_text("p")

    assert state["posts"][0]["json"]["options"]["num_ctx"] == 8192
