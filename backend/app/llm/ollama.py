"""Ollama fact-extraction provider (local inference, Groq alternative).

Document-agnostic adapter between the frozen ``LLMProvider`` protocol
(``app.llm.provider``) and a locally hosted Ollama server
(``{base_url}/api/chat``) over plain HTTP via ``httpx``.

Operational contract (same shape as the Groq provider):

- One bounded semantic chunk per request, called serially by the
  service layer (exactly one active extraction worker).
- Deterministic generation: ``temperature=0``.
- Conservative output: ``num_predict`` cap (1200 extraction / 800
  judgment); real output is normally far smaller.
- ``format: "json"`` requested where supported; model text goes through
  the same cleaning/repair + local Pydantic validation as Groq output.
  Unparseable output marks the chunk failed — facts are never fabricated.
- Model reasoning (``thinking`` content) is never surfaced: only
  ``message.content`` is read, validated, and returned.

Reasoning control: the Ollama chat API accepts an optional ``think``
field for thinking models. ``think`` is sent ONLY when explicitly
configured (``OLLAMA_THINK``); the default omits it (model default).
Live-model verification of this field is pending.

Contract (matches ``app.llm.provider`` exactly):

- returns ``FactExtractionBatch(drafts, model)``;
- exposes ``model_name: str`` + ``extract_facts(prompt)``;
- raises only ``LLMError`` subclasses for provider failures;
- default model ``qwen3.5:9b``.

Lazy HTTP rule: ``import httpx`` happens ONLY inside methods, so
importing this module never requires the library. A missing library at
call time raises ``LLMUnavailableError`` (never ``ImportError``).

Retry policy: up to ``max_retries`` retries on transient failures ONLY
(timeouts, HTTP 5xx) with bounded exponential backoff plus jitter.
Connection failures (server down) and HTTP 404 (model not pulled) fail
immediately without retry. No request pacing: local inference has no
TPM tier to protect.

Docker note: the backend runs in Docker while Ollama runs natively on
the Windows host; ``http://host.docker.internal:11434`` reaches the
host from Docker Desktop. The model is owned by the host installation
— never downloaded, bundled, or baked into images here.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.llm.groq import clean_json_text, repair_truncated_json
from app.llm.provider import (
    FactExtractionBatch,
    LLMMalformedError,
    LLMTimeoutError,
    LLMTransportError,
    LLMUnavailableError,
)
from app.llm.rate_limit import compute_backoff_delay
from app.models import FactDraft

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"

#: Deterministic generation for fact extraction.
OLLAMA_TEMPERATURE = 0.0

#: Conservative output caps (parity with the Groq extraction path).
OLLAMA_MAX_OUTPUT_TOKENS = 1200
OLLAMA_JUDGMENT_MAX_TOKENS = 800

logger = logging.getLogger(__name__)

_HTTPX_INSTALL_GUIDANCE = (
    "The 'httpx' library is not installed. Install it with "
    "'pip install httpx' to use the Ollama provider."
)

_RAW_SNIPPET_LIMIT = 500

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "OLLAMA_JUDGMENT_MAX_TOKENS",
    "OLLAMA_MAX_OUTPUT_TOKENS",
    "OLLAMA_TEMPERATURE",
    "OllamaFactsProvider",
    "OllamaJudgmentTransport",
    "is_context_overflow",
]


def _context_overflow_in(text: str) -> bool:
    lowered = text.lower()
    if "exceed_context_size_error" in lowered:
        return True
    return "context" in lowered and (
        "exceed" in lowered or "too large" in lowered or "length" in lowered
    )


def is_context_overflow(exc: BaseException) -> bool:
    try:
        return _context_overflow_in(str(exc))
    except Exception:
        return False


def _sleep_backoff(
    retry_number: int,
    exc: BaseException,
    max_retries: int,
    base_s: float,
    max_s: float,
) -> None:
    """Log one retry and sleep its backoff delay.

    Shared by the facts provider and the judgment transport so both
    retry loops pace identically.
    """
    delay = compute_backoff_delay(
        attempt=retry_number, base_s=base_s, max_s=max_s
    )
    logger.info(
        "Ollama retry %d/%d after %.1fs (%s): %s",
        retry_number,
        max_retries,
        delay,
        type(exc).__name__,
        str(exc)[:200],
    )
    time.sleep(delay)


class _FatalTransport(LLMTransportError):
    """Internal: transport failure that must NOT be retried.

    Still an :class:`LLMTransportError` to every caller — the service
    layer records it as a chunk failure exactly like any other provider
    error. Only the in-provider retry loop distinguishes it.
    """


class _UnparseableOutput(Exception):
    """Internal: model output that is not valid ``{drafts: [...]}`` JSON."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.message = message
        self.snippet = raw_text[:_RAW_SNIPPET_LIMIT] if isinstance(raw_text, str) else ""


def _parse_drafts(raw_text: Any) -> list[FactDraft]:
    """Validate raw model text into ``FactDraft`` items.

    Same cleaning/repair + Pydantic validation as the Groq path: model
    text is never trusted, and missing fields fail loudly instead of
    being fabricated.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise _UnparseableOutput("empty response text from model", "")
    try:
        cleaned = clean_json_text(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = json.loads(repair_truncated_json(cleaned))
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


def _chat_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    think: bool | None,
    keep_alive: int | None,
    timeout_s: int,
) -> dict[str, Any]:
    """One non-streaming ``POST /api/chat`` call; returns the payload.

    Raises:
        LLMUnavailableError: when ``httpx`` is missing.
        LLMTimeoutError: on timeouts (retryable by the caller).
        _FatalTransport: on connection failure (server down) or HTTP
            404 (model not pulled) — never retried.
        LLMTransportError: on other HTTP/API failures (retryable).
    """
    try:
        import httpx
    except ImportError as exc:
        raise LLMUnavailableError(_HTTPX_INSTALL_GUIDANCE) from exc

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if think is not None:
        body["think"] = think
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)
    try:
        try:
            response = client.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise _FatalTransport(
                f"Ollama unavailable at {base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out (model={model}): {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 404:
                raise _FatalTransport(
                    f"Ollama model {model!r} not found at {base_url}: "
                    f"pull it with `ollama pull {model}`"
                ) from exc
            snippet = ""
            try:
                snippet = (exc.response.text or "")[:200]
            except Exception:
                pass
            if is_context_overflow(exc) or _context_overflow_in(snippet):
                raise _FatalTransport(
                    f"Ollama request exceeds model context (deterministic, "
                    f"not retried) (model={model}): {snippet}"
                ) from exc
            raise LLMTransportError(
                f"Ollama HTTP {status} (model={model}): {snippet}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMTransportError(
                f"Ollama transport error (model={model}): {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMTransportError(
                f"Ollama returned non-JSON output (model={model}): {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LLMTransportError(
                f"Ollama returned an unexpected payload (model={model})."
            )
        return payload
    finally:
        try:
            client.close()
        except Exception:
            pass


def _log_eval_counts(model: str, payload: dict[str, Any]) -> None:
    """Log Ollama token statistics (numbers only, best-effort).

    Observability parity with the Groq usage log. Never affects
    extraction.
    """
    try:
        prompt_n = payload.get("prompt_eval_count")
        eval_n = payload.get("eval_count")
        if prompt_n is None and eval_n is None:
            return
        logger.info(
            "Ollama usage model=%s prompt_eval_count=%s eval_count=%s",
            model, prompt_n, eval_n,
        )
    except Exception:
        return


class OllamaFactsProvider:
    """Document-agnostic Ollama implementation of the ``LLMProvider`` protocol.

    The constructor stores parameters only (no client construction, no
    network). All HTTP surface lives in :meth:`extract_facts`.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_s: int = 120,
        max_retries: int = 2,
        think: bool | None = None,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        max_output_tokens: int = OLLAMA_MAX_OUTPUT_TOKENS,
        keep_alive: int | None = -1,
    ) -> None:
        self._base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._think = think
        self._keep_alive = keep_alive
        self._backoff_base_s = max(0.0, float(backoff_base_s))
        self._backoff_max_s = max(0.0, float(backoff_max_s))
        self._max_output_tokens = max(1, int(max_output_tokens))

    @property
    def model_name(self) -> str:
        """The model name reported to the service layer."""
        return self._model

    def extract_facts(self, prompt: str) -> FactExtractionBatch:
        """Send a ready-made prompt and return the validated batch.

        One chunk per call; the caller runs calls serially. Raises:

            ValueError: on an empty prompt (caller bug, not retried).
            LLMUnavailableError: when ``httpx`` is missing.
            LLMTimeoutError: when calls time out on every attempt.
            LLMTransportError: when calls fail on every attempt, or
                immediately when Ollama is unreachable / the model is
                not pulled (never retried).
            LLMMalformedError: when output is not valid FactDraft JSON
                after all attempts (message carries a raw snippet for
                debuggability).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        def _backoff(retry_number: int, exc: BaseException) -> None:
            _sleep_backoff(
                retry_number,
                exc,
                self._max_retries,
                self._backoff_base_s,
                self._backoff_max_s,
            )

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            last = attempt >= attempts
            try:
                # Single HTTP call site in this class.
                payload = _chat_once(
                    base_url=self._base_url,
                    model=self._model,
                    prompt=prompt,
                    temperature=OLLAMA_TEMPERATURE,
                    num_predict=self._max_output_tokens,
                    think=self._think,
                    keep_alive=self._keep_alive,
                    timeout_s=self._timeout_s,
                )
            except _FatalTransport as exc:
                raise exc
            except (LLMTimeoutError, LLMTransportError) as exc:
                if last:
                    raise
                _backoff(attempt, exc)
                continue
            try:
                message = payload.get("message") or {}
                drafts = _parse_drafts(message.get("content"))
            except _UnparseableOutput as exc:
                if last:
                    raise LLMMalformedError(
                        f"Ollama returned malformed FactDraft JSON after "
                        f"{attempts} attempt(s) (model={self._model}): "
                        f"{exc.message} | raw snippet: {exc.snippet!r}"
                    ) from exc
                _backoff(attempt, exc)
                continue
            _log_eval_counts(self._model, payload)
            return FactExtractionBatch(drafts=drafts, model=self._model)
        raise AssertionError("unreachable: retry loop always returns or raises")  # pragma: no cover


class OllamaJudgmentTransport:
    """Ollama text transport for judgment prompts.

    Implements the :class:`JudgmentTransport` protocol: sends a
    ready-made prompt, returns raw model text. Parsing into
    :class:`RelationshipJudgment` stays in ``app.llm.judgment`` —
    structured enforcement happens in validation, not here.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_s: int = 120,
        max_retries: int = 2,
        think: bool | None = None,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        keep_alive: int | None = -1,
    ) -> None:
        self._base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._think = think
        self._keep_alive = keep_alive
        self._backoff_base_s = max(0.0, float(backoff_base_s))
        self._backoff_max_s = max(0.0, float(backoff_max_s))

    @property
    def model_name(self) -> str:
        """The model name reported to the caller."""
        return self._model

    def complete_text(self, prompt: str) -> str:
        """Send a ready-made prompt and return the raw model text.

        Raises:
            ValueError: on an empty prompt (caller bug, not retried).
            LLMUnavailableError: when ``httpx`` is missing.
            LLMTimeoutError: when calls time out on every attempt.
            LLMTransportError: when calls fail or return empty text on
                every attempt, or immediately when Ollama is unreachable
                / the model is not pulled.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        def _backoff(retry_number: int, exc: BaseException) -> None:
            _sleep_backoff(
                retry_number,
                exc,
                self._max_retries,
                self._backoff_base_s,
                self._backoff_max_s,
            )

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            last = attempt >= attempts
            try:
                # Single HTTP call site in this class.
                payload = _chat_once(
                    base_url=self._base_url,
                    model=self._model,
                    prompt=prompt,
                    temperature=OLLAMA_TEMPERATURE,
                    num_predict=OLLAMA_JUDGMENT_MAX_TOKENS,
                    think=self._think,
                    keep_alive=self._keep_alive,
                    timeout_s=self._timeout_s,
                )
            except _FatalTransport as exc:
                raise exc
            except (LLMTimeoutError, LLMTransportError) as exc:
                if last:
                    raise
                _backoff(attempt, exc)
                continue
            message = payload.get("message") or {}
            text = message.get("content", "")
            if not isinstance(text, str) or not text.strip():
                exc: Exception = LLMTransportError("Ollama returned empty text.")
                if last:
                    raise LLMTransportError(
                        f"Ollama returned empty text after {attempts} "
                        f"attempt(s) (model={self._model})."
                    ) from exc
                _backoff(attempt, exc)
                continue
            _log_eval_counts(self._model, payload)
            return text
        raise AssertionError("unreachable: retry loop always returns or raises")  # pragma: no cover
