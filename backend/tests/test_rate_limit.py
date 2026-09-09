"""Tests: token estimator, global pacing, backoff, retry mapping.

No network, no real sleeps, no randomness outside seeded generators:
time and randomness are injected fakes throughout.
"""

import random
import sys
import threading
import types

import pytest

from app.llm.rate_limit import (
    RateLimiter,
    compute_backoff_delay,
    estimate_tokens,
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
    params = {"min_interval_s": 2.5, "tpm_target": 6500.0}
    params.update(overrides)
    limiter = RateLimiter(clock=clock, sleeper=sleeper, **params)
    return limiter, clock, sleeper


# ---------------------------------------------------------------------------
# A. Token estimator
# ---------------------------------------------------------------------------


def test_estimate_tokens_monotonic_and_bounded():
    assert estimate_tokens("") >= 1
    short = estimate_tokens("hello")
    long = estimate_tokens("hello world, this is much longer text")
    assert 0 < short <= long
    assert estimate_tokens("x" * 4000) >= estimate_tokens("x" * 40)


# ---------------------------------------------------------------------------
# B. Request spacing
# ---------------------------------------------------------------------------


def test_second_request_waits_full_interval():
    limiter, _, sleeper = make_limiter()
    assert limiter.acquire(10) == 0.0
    assert limiter.acquire(10) == 2.5
    assert sleeper.slept == [2.5]


def test_first_request_never_blocks():
    limiter, _, sleeper = make_limiter()
    assert limiter.acquire(6000) == 0.0
    assert sleeper.slept == []


# ---------------------------------------------------------------------------
# C. Token pacing
# ---------------------------------------------------------------------------


def test_token_budget_blocks_until_window_rolls():
    limiter, _, sleeper = make_limiter(tpm_target=100.0, min_interval_s=0.0)
    assert limiter.acquire(90) == 0.0
    # 90 used + 20 requested > 100: must wait until the first entry ages out.
    # Sleeps happen in bounded steps; the total must equal the window wait.
    assert limiter.acquire(20) == pytest.approx(60.0)
    assert abs(sum(sleeper.slept) - 60.0) < 1e-9
    assert all(step <= 5.0 + 1e-9 for step in sleeper.slept)


def test_token_usage_under_budget_does_not_block():
    limiter, _, sleeper = make_limiter(tpm_target=100.0, min_interval_s=0.0)
    assert limiter.acquire(40) == 0.0
    assert limiter.acquire(40) == 0.0
    assert sleeper.slept == []


# ---------------------------------------------------------------------------
# D. Backoff math
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
# E. Retry-After extraction
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
# F. Concurrency: shared limiter cannot be bypassed
# ---------------------------------------------------------------------------


def test_concurrent_acquires_stay_spaced():
    limiter, _, _ = make_limiter(min_interval_s=0.05, tpm_target=10**9)
    starts = []
    lock = threading.Lock()

    def worker():
        limiter.acquire(1)
        with lock:
            starts.append(limiter._clock())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(starts) == 4
    ordered = sorted(starts)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    assert all(gap >= 0.04 for gap in gaps)


# ---------------------------------------------------------------------------
# G. Retry behavior at the provider level (faked SDK via sys.modules)
# ---------------------------------------------------------------------------


def _install_fake_groq(monkeypatch, behavior):
    """Install a fake ``groq`` module; behavior["calls"] maps 1-based call
    index to "boom"/"auth"/"timeout"/"rate_limit", else canned JSON wins."""
    state = {"calls": 0, "sleeps": []}
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
