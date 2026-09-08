"""Gemini fact-extraction provider (task F-P2).

Document-agnostic adapter between the frozen ``LLMProvider`` protocol
(``app.llm.provider``) and the Gemini API via the ``google-genai`` SDK.

Contract (matches ``app.llm.provider`` exactly):
- returns ``FactExtractionBatch(drafts, model)``;
- exposes ``model_name: str`` + ``extract_facts(prompt)``;
- raises only ``LLMError`` subclasses for provider failures;
- default model ``gemini-3.5-flash``.

Lazy SDK rule: ``from google import genai`` happens ONLY inside
:meth:`GeminiFactsProvider.extract_facts`, so importing this module never
requires the SDK. A missing SDK at call time raises
``LLMUnavailableError`` (never ``ImportError``) with install guidance.

PLAN compliance (sections 11/12/14): this layer does no arithmetic, unit
conversion, date parsing, or provenance generation (11); it constrains
output with ``response_mime_type="application/json"`` plus a
``response_schema`` derived from ``FactDraft.model_json_schema()`` and
validates every item into ``FactDraft`` (12); the caller supplies
evidence-scoped prompts (never whole documents) and the schema bounds
output tokens (14). Temperature/top_p/top_k are deliberately NOT set,
per current-model guidance. No dataset/metric/year/filename logic lives
here, and the constructor takes no Settings.

Retry policy: up to ``max_retries`` immediate retries (total attempts =
1 + ``max_retries``) on transport, timeout, and parse failures. No sleep
or backoff: prototype scale, bounded free-tier calls, and the service
layer owns scheduling.

SDK-API uncertainty (this environment has no SDK/network, so the call
shape is not verified against the live SDK): the single SDK call site in
``extract_facts`` is ``genai.Client(api_key=...)`` plus
``client.models.generate_content(model=..., contents=..., config={
"response_mime_type": "application/json", "response_schema": ...})``
with ``response.text`` readout. Timeout classification is duck-typed
(``TimeoutError`` -> ``LLMTimeoutError``; any other call failure ->
``LLMTransportError``) so no ``google.genai.errors`` imports are needed.
``timeout_s`` is stored/plumbed for the service layer; per-call timeout
enforcement via client ``HttpOptions`` is deliberately left unwired until
verified against the installed SDK. The ``$defs`` hoisting in
:func:`build_response_schema` keeps ``$ref`` pointers resolvable at the
schema root; Gemini's structured-output dialect may need further
flattening once verified live.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.models import FactDraft
from app.llm.provider import (
    FactExtractionBatch,
    LLMMalformedError,
    LLMTimeoutError,
    LLMTransportError,
    LLMUnavailableError,
)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

_SDK_INSTALL_GUIDANCE = (
    "The 'google-genai' SDK is not installed. Install it with "
    "'pip install google-genai' and provide a Gemini API key to use "
    "GeminiFactsProvider."
)

_RAW_SNIPPET_LIMIT = 500

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "GeminiFactsProvider",
    "build_response_schema",
]


def build_response_schema() -> dict[str, Any]:
    """Build the GenerateContent ``response_schema`` for fact extraction.

    Returns a JSON-schema dict for ``{"drafts": [FactDraft-shape]}`` where
    the item schema comes straight from ``FactDraft.model_json_schema()``.
    ``$defs`` (enums such as ValueKind/EstimateStatus) are hoisted to the
    schema root so ``#/$defs/...`` refs resolve against the document root.
    """
    draft_schema = FactDraft.model_json_schema()
    defs = draft_schema.pop("$defs", None)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"drafts": {"type": "array", "items": draft_schema}},
        "required": ["drafts"],
    }
    if defs:
        schema["$defs"] = defs
    return schema


class _UnparseableOutput(Exception):
    """Internal: model output that is not valid ``{drafts: [...]}`` JSON."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.message = message
        self.snippet = raw_text[:_RAW_SNIPPET_LIMIT] if isinstance(raw_text, str) else ""


def _load_payload(raw_text: str) -> Any:
    """Parse raw model text as JSON, tolerating markdown code fences."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        stripped = raw_text.strip()
        if not stripped.startswith("```"):
            raise
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return json.loads("\n".join(lines))


def _parse_drafts(raw_text: Any) -> list[FactDraft]:
    """Validate raw model text into ``FactDraft`` items.

    Raises:
        _UnparseableOutput: on empty text, JSON-decode failure, wrong
            top-level shape, or ``FactDraft`` validation failure.
    """
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


class GeminiFactsProvider:
    """Document-agnostic Gemini implementation of the ``LLMProvider`` protocol.

    The constructor stores parameters only (no client construction, no
    network). All SDK surface lives in :meth:`extract_facts`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_s: int = 60,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))

    @property
    def model_name(self) -> str:
        """The model name reported to the service layer."""
        return self._model

    def extract_facts(self, prompt: str) -> FactExtractionBatch:
        """Send a ready-made prompt and return the validated batch.

        Raises:
            ValueError: on an empty prompt (caller bug, not retried).
            LLMUnavailableError: when the ``google-genai`` SDK is missing.
            LLMTimeoutError: when SDK calls time out on every attempt.
            LLMTransportError: when SDK calls fail on every attempt.
            LLMMalformedError: when output is not valid FactDraft JSON
                after all attempts (message carries a raw snippet,
                truncated to 500 chars, for debuggability).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
            from google import genai  # lazy: module import must never need the SDK
        except ImportError as exc:
            raise LLMUnavailableError(_SDK_INSTALL_GUIDANCE) from exc

        config = {
            "response_mime_type": "application/json",
            "response_schema": build_response_schema(),
        }
        client = genai.Client(api_key=self._api_key)

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            last = attempt >= attempts
            try:
                # Single SDK call site in this codebase.
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
            except TimeoutError as exc:
                if last:
                    raise LLMTimeoutError(
                        f"Gemini request timed out after {attempts} "
                        f"attempt(s) (model={self._model})."
                    ) from exc
                continue
            except Exception as exc:
                if last:
                    raise LLMTransportError(
                        f"Gemini transport/API error after {attempts} "
                        f"attempt(s) (model={self._model}): {exc}"
                    ) from exc
                continue
            try:
                drafts = _parse_drafts(getattr(response, "text", ""))
            except _UnparseableOutput as exc:
                if last:
                    raise LLMMalformedError(
                        f"Gemini returned malformed FactDraft JSON after "
                        f"{attempts} attempt(s) (model={self._model}): "
                        f"{exc.message} | raw snippet: {exc.snippet!r}"
                    ) from exc
                continue
            return FactExtractionBatch(drafts=drafts, model=self._model)
        raise AssertionError("unreachable: retry loop always returns or raises")  # pragma: no cover
