"""Tests: serial pacing, backoff, retry mapping, JSON parsing.

No network, no real sleeps, no randomness outside seeded generators:
time and randomness are injected fakes throughout.

Operational contract under test:

- One chunk per Groq request, serial execution, >=2.5s start spacing.
- No token scheduling of any kind.
- Bounded exponential backoff honoring Retry-After.
- Robust JSON cleaning/repair + Pydantic validation (no fabrication).
"""

import random
import sys
import types

import pytest

from app.llm.rate_limit import (
    RateLimiter,
    compute_backoff_delay,
    retry_after_seconds,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeSleeper:
    def __init__(self, clock):
        self.clock = clock
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)
        self.clock.now += seconds


def make_limiter(**overrides):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    params = {"min_interval_s": 2.5}
    params.update(overrides)
    limiter = RateLimiter(clock=clock, sleeper=sleeper, **params)
    return limiter, clock, sleeper


# ---------------------------------------------------------------------------
# A. Request spacing (serial pacing, no token machinery)
# ---------------------------------------------------------------------------


def test_second_request_waits_full_interval():
    limiter, _, sleeper = make_limiter()
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 2.5
    assert sleeper.slept == [2.5]


def test_first_request_never_blocks():
    limiter, _, sleeper = make_limiter()
    assert limiter.acquire() == 0.0
    assert sleeper.slept == []


def test_interval_elapsed_does_not_block():
    limiter, clock, sleeper = make_limiter()
    assert limiter.acquire() == 0.0
    clock.now += 10.0
    assert limiter.acquire() == 0.0
    assert sleeper.slept == []


# ---------------------------------------------------------------------------
# B. Backoff math
# ---------------------------------------------------------------------------


def test_backoff_doubles_within_cap():
    rng = random.Random(7)
    delays = [
        compute_backoff_delay(attempt=i, base_s=1.0, max_s=60.0, rng=rng)
        for i in (1, 2, 3)
    ]
    for i, delay in enumerate(delays, start=1):
        nominal = min(60.0, 2.0 ** (i - 1))
        assert nominal / 2.0 <= delay <= nominal


def test_backoff_respects_max_cap():
    rng = random.Random(1)
    assert compute_backoff_delay(
        attempt=10, base_s=1.0, max_s=60.0, rng=rng) <= 60.0


def test_retry_after_overrides_upward_within_cap():
    rng = random.Random(3)
    delay = compute_backoff_delay(
        attempt=1, base_s=1.0, max_s=60.0, retry_after=25.0, rng=rng)
    assert 12.5 <= delay <= 25.0
    capped = compute_backoff_delay(
        attempt=1, base_s=1.0, max_s=60.0, retry_after=600.0, rng=rng)
    assert capped <= 60.0


def test_backoff_deterministic_with_seeded_rng():
    kwargs = {"attempt": 2, "base_s": 1.0, "max_s": 60.0}
    first = compute_backoff_delay(rng=random.Random(42), **kwargs)
    second = compute_backoff_delay(rng=random.Random(42), **kwargs)
    assert first == second


# ---------------------------------------------------------------------------
# C. Retry-After extraction
# ---------------------------------------------------------------------------


class _Headers(dict):
    def get(self, key, default=None):  # case-insensitive like httpx
        for stored, value in self.items():
            if stored.lower() == key.lower():
                return value
        return default


class _Response:
    def __init__(self, headers):
        self.headers = headers


class _Exc(Exception):
    def __init__(self, headers=None):
        self.response = _Response(headers) if headers is not None else None


def test_retry_after_seconds_header():
    assert retry_after_seconds(_Exc(_Headers({"Retry-After": "7"}))) == 7.0


def test_retry_after_milliseconds_header():
    assert retry_after_seconds(_Exc(_Headers({"retry-after-ms": "1500"}))) == 1.5


def test_retry_after_missing_or_garbage_is_none():
    assert retry_after_seconds(_Exc(None)) is None
    assert retry_after_seconds(_Exc(_Headers({}))) is None
    assert retry_after_seconds(_Exc(_Headers({"Retry-After": "soon"}))) is None
    assert retry_after_seconds(Exception("plain")) is None


# ---------------------------------------------------------------------------
# D. Serial contract: one worker, sequential acquires accumulate spacing
# ---------------------------------------------------------------------------
# NOTE: there is deliberately NO multithreaded limiter test. Production
# fact extraction runs exactly ONE worker (one chunk → one Groq request
# → >=2.5s spacing → next chunk), so concurrent acquisition is out of
# contract. A threaded test against an injected fake clock can livelock
# on sub-ULP float dust (clock.now += tiny_delay is a no-op, so the
# remaining delay never reaches zero) — a test-only artifact with no
# production counterpart, since real sleeps always advance real time.


def test_serial_acquires_accumulate_spacing():
    limiter, _, sleeper = make_limiter()
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 2.5
    assert limiter.acquire() == 2.5
    assert sleeper.slept == [2.5, 2.5]


# ---------------------------------------------------------------------------
# E. Retry behavior at the provider level (faked SDK via sys.modules)
# ---------------------------------------------------------------------------


def _install_fake_groq(monkeypatch, behavior):
    """Install a fake ``groq`` module; behavior["calls"] maps 1-based call
    index to "boom"/"auth"/"timeout"/"rate_limit", else canned JSON wins."""
    state = {"calls": 0, "sleeps": [], "kwargs": []}
    calls = behavior.get("calls", {})

    class _Err(Exception):
        """Generic transient failure (maps to the retryable branch)."""
        pass

    class _AuthErr(Exception):
        """Deterministic auth failure. Sibling of _Err, mirroring the real
        SDK taxonomy where RateLimitError/BadRequestError are siblings
        under APIStatusError — never parent/child."""

    class _RateLimitErr(Exception):
        def __init__(self, msg, headers=None):
            super().__init__(msg)
            self.response = types.SimpleNamespace(headers=headers or {})

    class _TimeoutErr(Exception):
        pass

    class _BadRequestErr(Exception):
        pass

    class _NotFoundErr(Exception):
        pass

    class _PermissionDeniedErr(Exception):
        pass

    errors = {
        "auth": _AuthErr,
        "rate_limit": _RateLimitErr,
        "timeout": _TimeoutErr,
        "boom": _Err,
    }

    def _raise(kind, **kwargs):
        cls = errors[kind]
        if kind == "rate_limit":
            raise cls("429 RESOURCE_EXHAUSTED", headers=kwargs.get("headers"))
        raise cls("synthetic failure")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    canned = behavior.get("canned", '{"drafts": []}')

    class FakeCompletions:
        def create(self, **kwargs):
            state["calls"] += 1
            state["kwargs"].append(kwargs)
            action = calls.get(state["calls"])
            if action is None:
                return FakeResponse(canned)
            if isinstance(action, tuple):
                kind, extra = action
                _raise(kind, **(extra or {}))
            else:
                _raise(action)

    fake = types.ModuleType("groq")
    fake.Groq = lambda **kwargs: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()))
    fake.APITimeoutError = _TimeoutErr
    fake.AuthenticationError = _AuthErr
    fake.BadRequestError = _BadRequestErr
    fake.NotFoundError = _NotFoundErr
    fake.PermissionDeniedError = _PermissionDeniedErr
    monkeypatch.setitem(sys.modules, "groq", fake)
    return state


def test_transient_failures_retry_then_succeed(monkeypatch):
    import time as _time
    from app.llm.groq import GroqFactsProvider

    state = _install_fake_groq(monkeypatch, {"calls": {1: "boom", 2: "boom"}})
    monkeypatch.setattr(_time, "sleep", lambda s: state["sleeps"].append(s))
    provider = GroqFactsProvider(
        api_key="k", max_retries=5, backoff_base_s=0.0, backoff_max_s=0.0)
    batch = provider.extract_facts("extract facts about X")
    assert batch.model == provider.model_name
    assert state["calls"] == 3


def test_retry_limit_surfaces_after_five(monkeypatch):
    import time as _time
    from app.llm.groq import GroqFactsProvider
    from app.llm.provider import LLMTransportError

    state = _install_fake_groq(
        monkeypatch, {"calls": {i: "boom" for i in range(1, 20)}})
    monkeypatch.setattr(_time, "sleep", lambda s: state["sleeps"].append(s))
    provider = GroqFactsProvider(
        api_key="k", max_retries=5, backoff_base_s=0.0, backoff_max_s=0.0)
    with pytest.raises(LLMTransportError):
        provider.extract_facts("extract facts about X")
    assert state["calls"] == 6  # 1 initial + 5 retries, then surfaces


def test_deterministic_401_never_retried(monkeypatch):
    from app.llm.groq import GroqFactsProvider
    from app.llm.provider import LLMTransportError

    state = _install_fake_groq(
        monkeypatch, {"calls": {i: "auth" for i in range(1, 20)}})
    provider = GroqFactsProvider(api_key="k", max_retries=5)
    with pytest.raises(LLMTransportError, match="not retried"):
        provider.extract_facts("extract facts about X")
    assert state["calls"] == 1


def test_retry_after_header_drives_backoff(monkeypatch):
    import time as _time
    from app.llm.groq import GroqFactsProvider

    state = _install_fake_groq(
        monkeypatch,
        {"calls": {1: ("rate_limit", {"headers": {"Retry-After": "30"}})}})
    recorded = []
    monkeypatch.setattr(_time, "sleep", recorded.append)
    provider = GroqFactsProvider(
        api_key="k", max_retries=5, backoff_base_s=1.0, backoff_max_s=60.0)
    batch = provider.extract_facts("extract facts about X")
    assert batch.model == provider.model_name
    assert state["calls"] == 2
    # Retry-After 30 overrides the ~0.5-1.0s exponential value (capped at 60).
    assert len(recorded) == 1
    assert 15.0 <= recorded[0] <= 30.0


def test_timeout_retries_then_surfaces(monkeypatch):
    import time as _time
    from app.llm.groq import GroqFactsProvider
    from app.llm.provider import LLMTimeoutError

    state = _install_fake_groq(
        monkeypatch, {"calls": {i: "timeout" for i in range(1, 20)}})
    monkeypatch.setattr(_time, "sleep", lambda s: state["sleeps"].append(s))
    provider = GroqFactsProvider(
        api_key="k", max_retries=2, backoff_base_s=0.0, backoff_max_s=0.0)
    with pytest.raises(LLMTimeoutError):
        provider.extract_facts("extract facts about X")
    assert state["calls"] == 3


def test_provider_uses_deterministic_conservative_params(monkeypatch):
    from app.llm.groq import (
        EXTRACTION_MAX_OUTPUT_TOKENS,
        EXTRACTION_TEMPERATURE,
        GroqFactsProvider,
    )

    state = _install_fake_groq(monkeypatch, {})
    provider = GroqFactsProvider(api_key="k")
    provider.extract_facts("extract facts about X")
    assert state["calls"] == 1
    kwargs = state["kwargs"][0]
    assert kwargs["temperature"] == EXTRACTION_TEMPERATURE == 0
    assert kwargs["max_tokens"] == EXTRACTION_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# F. JSON robustness (pure functions, no network)
# ---------------------------------------------------------------------------


def test_clean_json_strips_fences_and_prose():
    from app.llm.groq import clean_json_text

    assert clean_json_text('```json\n{"drafts": []}\n```') == '{"drafts": []}'
    assert clean_json_text('```\n{"drafts": []}\n```') == '{"drafts": []}'
    wrapped = 'Here is the result:\n{"drafts": []}\nHope this helps.'
    assert clean_json_text(wrapped) == '{"drafts": []}'


def test_repair_truncated_json_closes_structures():
    import json

    from app.llm.groq import repair_truncated_json

    truncated = '{"drafts": [{"subject": "X"'
    repaired = repair_truncated_json(truncated)
    payload = json.loads(repaired)
    assert isinstance(payload, dict)
    assert "drafts" in payload


def test_malformed_output_surfaces_without_fabrication(monkeypatch):
    import time as _time
    from app.llm.groq import GroqFactsProvider
    from app.llm.provider import LLMMalformedError

    state = _install_fake_groq(monkeypatch, {"canned": "not json at all {{{"})
    monkeypatch.setattr(_time, "sleep", lambda s: state["sleeps"].append(s))
    provider = GroqFactsProvider(
        api_key="k", max_retries=1, backoff_base_s=0.0, backoff_max_s=0.0)
    with pytest.raises(LLMMalformedError):
        provider.extract_facts("extract facts about X")
    assert state["calls"] == 2  # retried once, then surfaced — never guessed
