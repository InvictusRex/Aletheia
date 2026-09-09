"""Groq fact-extraction provider (serial, one chunk per request).

Document-agnostic adapter between the frozen ``LLMProvider`` protocol
(``app.llm.provider``) and Groq's OpenAI-compatible API
(``https://api.groq.com/openai/v1``) via the official ``groq`` SDK.

Operational contract:

- One bounded semantic chunk per request, called serially by the
  service layer (exactly one active extraction worker).
- Minimum 2.5s between request starts via the shared
  :class:`RateLimiter` (pure interval pacing, no token scheduling).
- Deterministic generation: ``temperature=0``.
- Conservative output: ``max_tokens=2000`` hard cap; the prompt asks
  for concise structured facts only, so real output is far smaller.
- Plain JSON output + local Pydantic validation (no provider-side
  strict-schema machinery). Model text is cleaned (markdown fences,
  surrounding prose) and truncated-output repair is attempted; if the
  response still cannot be parsed/validated the chunk is marked failed
  and processing continues — facts are never fabricated.

Contract (matches ``app.llm.provider`` exactly):

- returns ``FactExtractionBatch(drafts, model)``;
- exposes ``model_name: str`` + ``extract_facts(prompt)``;
- raises only ``LLMError`` subclasses for provider failures;
- default model ``openai/gpt-oss-120b``.

Lazy SDK rule: ``from groq import ...`` happens ONLY inside methods, so
importing this module never requires the SDK. A missing SDK at call time
raises ``LLMUnavailableError`` (never ``ImportError``).

Retry policy: up to ``max_retries`` retries (total attempts = 1 +
``max_retries``) on transient failures ONLY (rate limits / HTTP 429,
timeouts, connection errors, server errors, parse failures) with
bounded exponential backoff plus jitter — honoring provider
``Retry-After`` subject to ``backoff_max_s``. Deterministic 4xx
failures (authentication, bad request, permission, not-found) are
raised immediately without retry. Pacing of request starts goes through
the shared :class:`RateLimiter`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import ValidationError

from app.models import FactDraft
from app.llm.rate_limit import (
    RateLimiter,
    compute_backoff_delay,
    extract_total_tokens,
    retry_after_seconds,
)
from app.llm.provider import (
    FactExtractionBatch,
    LLMMalformedError,
    LLMTimeoutError,
    LLMTransportError,
    LLMUnavailableError,
)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

#: Deterministic generation for fact extraction.
EXTRACTION_TEMPERATURE = 0.0

#: Hard output cap. Prompts normally produce far less; input + output
#: both count toward the provider TPM limit, so this stays conservative.
EXTRACTION_MAX_OUTPUT_TOKENS = 2000

logger = logging.getLogger(__name__)

_SDK_INSTALL_GUIDANCE = (
    "The 'groq' SDK is not installed. Install it with "
    "'pip install groq' and provide a Groq API key to use "
    "GroqFactsProvider."
)

_RAW_SNIPPET_LIMIT = 500

__all__ = [
    "DEFAULT_GROQ_MODEL",
    "EXTRACTION_MAX_OUTPUT_TOKENS",
    "EXTRACTION_TEMPERATURE",
    "GroqFactsProvider",
    "clean_json_text",
    "repair_truncated_json",
]


def clean_json_text(content: str) -> str:
    """Strip fences and isolate the JSON payload (pure function).

    Handles markdown code fences, surrounding whitespace, and harmless
    prose around the object/array. Returns the stripped candidate;
    raises no error — ``json.loads`` validates afterward.
    """
    text = content.strip()
    # Remove markdown code fences (```json ... ``` or ``` ... ```).
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Isolate the JSON object/array when wrapped in prose.
    obj_start = text.find("{")
    arr_start = text.find("[")
    starts = [p for p in (obj_start, arr_start) if p != -1]
    if starts:
        start = min(starts)
        if start > 0:
            text = text[start:]
    if text.startswith("{"):
        end = text.rfind("}")
        if end != -1:
            text = text[: end + 1]
    elif text.startswith("["):
        end = text.rfind("]")
        if end != -1:
            text = text[: end + 1]
    return text.strip()


def repair_truncated_json(content: str) -> str:
    """Attempt safe recovery of truncated JSON (pure function).

    Conservative by construction: a repaired candidate is returned ONLY
    when it parses as JSON. Candidates are built by truncating to the
    end of a complete value (longest first) — or by closing one trailing
    unterminated string with exactly what the model already wrote — then
    closing any open brackets/braces. Model-written content is never
    altered and no fact fields are invented; the repaired payload is
    still fully validated by Pydantic afterward, so unrecoverable output
    (dangling keys, missing values) simply fails validation and the
    chunk is recorded as failed/retried instead of fabricated.
    """
    text = content.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Single scan: container stack + string state. Snapshot the open
    # stack at the end of every complete value so candidates can be
    # tried longest-first without rescanning.
    snapshots: list[tuple[int, list[str]]] = []  # (end_index, open stack)
    stack: list[str] = []
    in_string = False
    escape = False
    for i, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
                snapshots.append((i + 1, list(stack)))
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
            snapshots.append((i + 1, list(stack)))

    def _closers(opens: list[str]) -> str:
        return "".join("}" if o == "{" else "]" for o in reversed(opens))

    # Longest candidate first: close one trailing unterminated string
    # using only the model's own characters, then close containers.
    if in_string:
        candidate = text + '"' + _closers(stack)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    for end, opens in reversed(snapshots):
        candidate = text[:end] + _closers(opens)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    # Unrecoverable (e.g. a dangling key with no value): return the
    # original so the caller raises a decode failure and the chunk is
    # marked failed — never invent the missing value.
    return text


class _UnparseableOutput(Exception):
    """Internal: model output that is not valid ``{drafts: [...]}`` JSON."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.message = message
        self.snippet = raw_text[:_RAW_SNIPPET_LIMIT] if isinstance(raw_text, str) else ""


def _load_payload(raw_text: str) -> Any:
    """Parse raw model text as JSON with fence/prose tolerance + repair."""
    cleaned = clean_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = repair_truncated_json(cleaned)
        return json.loads(repaired)


def _parse_drafts(raw_text: Any) -> list[FactDraft]:
    """Validate raw model text into ``FactDraft`` items."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise _UnparseableOutput("empty response text from model", "")
    try:
        payload = _load_payload(raw_text)
    except json.JSONDecodeError as exc:
        raise _UnparseableOutput(f"response is not valid JSON: {exc}", raw_text) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("drafts"), list):
        raise _UnparseableOutput(
            "expected a JSON object with a 'drafts' list", raw_text
        )
    try:
        return [FactDraft.model_validate(item) for item in payload["drafts"]]
    except ValidationError as exc:
        raise _UnparseableOutput(f"FactDraft validation failed: {exc}", raw_text) from exc


class GroqFactsProvider:
    """Document-agnostic Groq implementation of the ``LLMProvider`` protocol.

    The constructor stores parameters only (no client construction, no
    network). All SDK surface lives in :meth:`extract_facts`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        timeout_s: int = 60,
        max_retries: int = 5,
        limiter: RateLimiter | None = None,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        max_output_tokens: int = EXTRACTION_MAX_OUTPUT_TOKENS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        # Pacing is opt-in at construction: the service layer passes the
        # process-global shared limiter; direct constructions (tests)
        # pace nothing and stay fast/deterministic.
        self._limiter = limiter
        self._backoff_base_s = max(0.0, float(backoff_base_s))
        self._backoff_max_s = max(0.0, float(backoff_max_s))
        self._max_output_tokens = max(1, int(max_output_tokens))

    @property
    def model_name(self) -> str:
        """The model name reported to the service layer."""
        return self._model

    def extract_facts(self, prompt: str) -> FactExtractionBatch:
        """Send a ready-made prompt and return the validated batch.

        One chunk per call; the caller runs calls serially with ≥2.5s
        start spacing. Raises:

            ValueError: on an empty prompt (caller bug, not retried).
            LLMUnavailableError: when the ``groq`` SDK is missing.
            LLMTimeoutError: when SDK calls time out on every attempt.
            LLMTransportError: when SDK calls fail on every attempt, or
                immediately on deterministic 4xx failures (auth, bad
                request, permission, not-found) which are never retried.
            LLMMalformedError: when output is not valid FactDraft JSON
                after all attempts (message carries a raw snippet,
                truncated to 500 chars, for debuggability).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
            from groq import (
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                Groq,
                NotFoundError,
                PermissionDeniedError,
            )
        except ImportError as exc:
            raise LLMUnavailableError(_SDK_INSTALL_GUIDANCE) from exc

        client = Groq(api_key=self._api_key)

        def _pace_and_maybe_wait() -> None:
            if self._limiter is None:
                return
            self._limiter.acquire()

        def _backoff(retry_number: int, exc: BaseException) -> None:
            delay = compute_backoff_delay(
                attempt=retry_number,
                base_s=self._backoff_base_s,
                max_s=self._backoff_max_s,
                retry_after=retry_after_seconds(exc),
            )
            logger.info(
                "Groq retry %d/%d after %.1fs (%s): %s",
                retry_number,
                self._max_retries,
                delay,
                type(exc).__name__,
                str(exc)[:200],
            )
            if self._limiter is not None:
                self._limiter.sleep(delay)
            else:
                time.sleep(delay)

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            last = attempt >= attempts
            _pace_and_maybe_wait()
            try:
                # Single SDK call site in this codebase: one chunk per
                # request, deterministic output, conservative cap.
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=EXTRACTION_TEMPERATURE,
                    max_tokens=self._max_output_tokens,
                    timeout=self._timeout_s,
                )
            except (APITimeoutError, TimeoutError) as exc:
                if last:
                    raise LLMTimeoutError(
                        f"Groq request timed out after {attempts} "
                        f"attempt(s) (model={self._model})."
                    ) from exc
                _backoff(attempt, exc)
                continue
            except (
                AuthenticationError,
                BadRequestError,
                PermissionDeniedError,
                NotFoundError,
            ) as exc:
                # Deterministic 4xx: retrying cannot help.
                raise LLMTransportError(
                    f"Groq deterministic API error, not retried "
                    f"(model={self._model}): {exc}"
                ) from exc
            except Exception as exc:
                if last:
                    raise LLMTransportError(
                        f"Groq transport/API error after {attempts} "
                        f"attempt(s) (model={self._model}): {exc}"
                    ) from exc
                _backoff(attempt, exc)
                continue
            try:
                content = response.choices[0].message.content
                drafts = _parse_drafts(content)
            except _UnparseableOutput as exc:
                if last:
                    raise LLMMalformedError(
                        f"Groq returned malformed FactDraft JSON after "
                        f"{attempts} attempt(s) (model={self._model}): "
                        f"{exc.message} | raw snippet: {exc.snippet!r}"
                    ) from exc
                _backoff(attempt, exc)
                continue
            self._log_usage(response)
            return FactExtractionBatch(drafts=drafts, model=self._model)
        raise AssertionError("unreachable: retry loop always returns or raises")  # pragma: no cover

    def _log_usage(self, response: object) -> None:
        """Log provider-reported token usage (numbers only, best-effort).

        Observability for the acceptance run: input + output totals per
        request. Never affects extraction.
        """
        try:
            total = extract_total_tokens(response)
            if total is None:
                return
            logger.info(
                "Groq usage model=%s total_tokens=%d", self._model, total
            )
        except Exception:
            return
