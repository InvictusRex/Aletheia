"""Groq request management: token estimation, global pacing, backoff.

Free-tier reality (RPM 30 / TPM 8K) demands conservative local control:

- :func:`estimate_tokens` — deterministic, dependency-free character
  heuristic (documented approximation, monotonic in input length).
- :class:`RateLimiter` — process-global pacing: minimum interval between
  request starts plus a rolling 60-second token budget. Thread-safe.
  Tests inject fake clock/sleeper; production uses real time.
- :func:`compute_backoff_delay` — bounded exponential backoff with
  equal jitter, honoring provider ``Retry-After`` subject to a cap.

Nothing here performs I/O. Providers call :meth:`RateLimiter.acquire`
before each attempt and sleep the computed backoff between retries.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque

__all__ = [
    "RateLimiter",
    "compute_backoff_delay",
    "estimate_tokens",
    "get_shared_limiter",
    "retry_after_seconds",
]

# Conservative character heuristic: ~4 chars/token is typical for mixed
# prose/IDs; using 4 keeps batches useful while the TPM target below
# carries the real safety margin. Deterministic and monotonic.
_CHARS_PER_TOKEN = 4

_WINDOW_SECONDS = 60.0
# Upper bound so a single acquire never sleeps longer than this per
# iteration; the loop re-evaluates (also keeps fake-clock tests exact).
_MAX_SLEEP_STEP = 5.0


def estimate_tokens(text: str) -> int:
    """Approximate token count for pacing/batching decisions.

    Deliberately approximate (never exact billing): ``ceil(bytes/4)``,
    minimum 1. Monotonic in input length. Do NOT use for correctness
    decisions, only for conservative budgeting.
    """
    if not isinstance(text, str):
        return 1
    size = len(text.encode("utf-8", errors="ignore"))
    return max(1, -(-size // _CHARS_PER_TOKEN))


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a ``Retry-After`` delay from a provider exception, if any.

    Reads ``exc.response.headers`` (httpx-style, case-insensitive) for
    ``retry-after`` (seconds) or ``retry-after-ms``. Returns None when
    absent, unparsable, or negative. Never raises.
    """
    try:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        # httpx Headers are case-insensitive; plain dicts may not be,
        # so compare lowered names explicitly.
        try:
            items = list(headers.items())
        except Exception:
            return None
        lowered = {
            str(name).lower(): value
            for name, value in items
            if isinstance(name, str)
        }
        for name in ("retry-after", "retry-after-ms"):
            raw = lowered.get(name)
            if raw is None:
                continue
            try:
                value = float(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if "ms" in name:
                value /= 1000.0
            if value >= 0:
                return value
    except Exception:
        return None
    return None


def compute_backoff_delay(
    *,
    attempt: int,
    base_s: float,
    max_s: float,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Bounded exponential backoff with equal jitter.

    ``attempt`` is 1-based (first retry == 1): ``base * 2**(attempt-1)``,
    capped at ``max_s``. A provider ``Retry-After`` overrides the local
    value upward (still capped). Final delay is halved plus uniform
    jitter over the other half, so tests inject a seeded ``rng`` for
    determinism and assert ranges, never exact sleeps.
    """
    grown = min(max_s, base_s * (2 ** max(0, attempt - 1)))
    if retry_after is not None and retry_after >= 0:
        grown = min(max_s, max(grown, retry_after))
    source = rng if rng is not None else random
    return grown / 2.0 + source.uniform(0, grown / 2.0)


class RateLimiter:
    """Global request pacing: start-interval plus rolling token budget.

    ``acquire(estimated_tokens)`` blocks until both (a) at least
    ``min_interval_s`` elapsed since the previous recorded start and
    (b) tokens started in the trailing 60 s plus the estimate fit within
    ``tpm_target`` — then records this start and returns seconds waited.
    Thread-safe: sleeps happen outside the lock with re-evaluation, so
    concurrent callers cannot bypass the limits.
    """

    def __init__(
        self,
        min_interval_s: float = 2.5,
        tpm_target: float = 6500.0,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._tpm_target = max(0.0, float(tpm_target))
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()

    def sleep(self, seconds: float) -> None:
        """Sleep through the injected sleeper (tests substitute fakes)."""
        if seconds > 0:
            self._sleeper(seconds)

    def _evict(self, now: float) -> None:
        while self._starts and now - self._starts[0] >= _WINDOW_SECONDS:
            self._starts.popleft()
        while self._tokens and now - self._tokens[0][0] >= _WINDOW_SECONDS:
            self._tokens.popleft()

    def _wait_for(self, estimated_tokens: int, now: float) -> float:
        wait = 0.0
        if self._starts:
            wait = max(wait, self._min_interval_s - (now - self._starts[-1]))
        windowed = sum(tokens for _, tokens in self._tokens)
        if windowed + estimated_tokens > self._tpm_target:
            oldest = self._tokens[0][0]
            wait = max(wait, (oldest + _WINDOW_SECONDS) - now)
        return max(0.0, wait)

    def acquire(self, estimated_tokens: int) -> float:
        """Block until a request may start; return seconds waited."""
        estimated = max(0, int(estimated_tokens))
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._evict(now)
                delay = self._wait_for(estimated, now)
                if delay <= 0:
                    started = self._clock()
                    self._starts.append(started)
                    self._tokens.append((started, estimated))
                    return waited
            step = min(delay, _MAX_SLEEP_STEP)
            self._sleeper(step)
            waited += step


_shared_limiter: RateLimiter | None = None
_shared_lock = threading.Lock()


def get_shared_limiter() -> RateLimiter:
    """Process-global limiter configured from application settings.

    Created once on first use so every provider instance and transport
    paces through the same limiter. Import of settings is deferred to
    call time to keep module import side-effect free.
    """
    global _shared_limiter
    if _shared_limiter is None:
        with _shared_lock:
            if _shared_limiter is None:
                from app.core.config import settings

                _shared_limiter = RateLimiter(
                    min_interval_s=settings.groq_min_request_interval_s,
                    tpm_target=settings.groq_tpm_target,
                )
    return _shared_limiter
