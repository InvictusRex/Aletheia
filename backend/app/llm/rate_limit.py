"""Groq request pacing and retry helpers.

Operational model (deliberately boring):

- Exactly one active Groq extraction worker (the service layer calls
  providers serially, one chunk per request).
- Minimum 2.5 seconds between request START times, enforced by
  :class:`RateLimiter`.
- Bounded exponential backoff with jitter for real transient failures,
  honoring provider ``Retry-After`` subject to a cap.

There is intentionally NO token-aware scheduling here: no TPM targets,
no token budgets, no token windows, no request packing, no throughput
optimization. Per-request token totals are usage-logged by callers for
observability only.
"""

from __future__ import annotations

import random
import threading
import time

__all__ = [
    "RateLimiter",
    "compute_backoff_delay",
    "extract_total_tokens",
    "get_shared_limiter",
    "retry_after_seconds",
]


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
    """Process-global serial pacing: minimum interval between starts.

    ``acquire()`` blocks until at least ``min_interval_s`` elapsed since
    the previous recorded start, then records this start and returns
    seconds waited. Thread-safe: sleeps happen outside the lock with
    re-evaluation, so concurrent callers cannot bypass the limit.

    The service layer runs exactly one extraction worker, so in practice
    there is no contention — the limiter simply spaces serial requests.
    """

    def __init__(
        self,
        min_interval_s: float = 2.5,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._last_start: float | None = None

    def sleep(self, seconds: float) -> None:
        """Sleep through the injected sleeper (tests substitute fakes)."""
        if seconds > 0:
            self._sleeper(seconds)

    def acquire(self) -> float:
        """Block until a request may start; return seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                if self._last_start is None:
                    self._last_start = now
                    return waited
                delay = self._min_interval_s - (now - self._last_start)
                if delay <= 0:
                    self._last_start = now
                    return waited
            self._sleeper(delay)
            waited += delay


def extract_total_tokens(response: object) -> int | None:
    """Read total (input + output) tokens from a provider response.

    Understands OpenAI-compatible ``usage`` payloads as either
    attributes or mappings (``total_tokens``, else prompt+completion).
    Returns None when unavailable or invalid. Never raises.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if total is None:
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total = prompt + completion
        else:
            total = getattr(usage, "total_tokens", None)
            if total is None:
                prompt = getattr(usage, "prompt_tokens", 0) or 0
                completion = getattr(usage, "completion_tokens", 0) or 0
                total = prompt + completion
        total = int(total)
        return total if total >= 0 else None
    except Exception:
        return None


_shared_limiter: RateLimiter | None = None
_shared_lock = threading.Lock()


def get_shared_limiter() -> RateLimiter:
    """Process-global limiter configured from application settings.

    Created once on first use so every provider instance paces through
    the same limiter. Import of settings is deferred to call time to
    keep module import side-effect free.
    """
    global _shared_limiter
    if _shared_limiter is None:
        with _shared_lock:
            if _shared_limiter is None:
                from app.core.config import settings

                _shared_limiter = RateLimiter(
                    min_interval_s=settings.groq_min_request_interval_s,
                )
    return _shared_limiter
