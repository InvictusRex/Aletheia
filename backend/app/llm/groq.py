"""Groq fact-extraction provider.

Document-agnostic adapter between the frozen ``LLMProvider`` protocol
(``app.llm.provider``) and Groq's OpenAI-compatible API
(``https://api.groq.com/openai/v1``) via the official ``groq`` SDK.

Contract (matches ``app.llm.provider`` exactly):
- returns ``FactExtractionBatch(drafts, model)``;
- exposes ``model_name: str`` + ``extract_facts(prompt)``;
- raises only ``LLMError`` subclasses for provider failures;
- default model ``openai/gpt-oss-120b``.

Lazy SDK rule: ``from groq import ...`` happens ONLY inside methods, so
importing this module never requires the SDK. A missing SDK at call time
raises ``LLMUnavailableError`` (never ``ImportError``).

Strict JSON Schema (Groq requirement, verified against the actual
Pydantic output for ``FactDraft``):
- every object gets ``"additionalProperties": false``;
- every object lists ALL of its properties in ``required``;
- local ``#/$defs/...`` references are inlined (the only ``$ref`` form
  Pydantic emits for these models; anything else raises loudly rather
  than shipping a schema Groq cannot enforce);
- ``default``/``title``/``description``/``format``/length and numeric
  bounds are dropped — Pydantic re-validates every item afterward, so no
  semantic constraint is lost, only wire-level hints strict mode rejects.
- ``anyOf`` (Optional fields) is preserved as-is.

Retry policy mirrors the previous provider: up to ``max_retries``
immediate retries (total attempts = 1 + ``max_retries``) on transient
failures ONLY (rate limits, timeouts, connection errors, server errors,
parse failures). Deterministic 4xx failures (authentication, bad
request, permission, not-found) are raised immediately without retry.
No sleep or backoff: prototype scale and bounded calls.
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

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

_SDK_INSTALL_GUIDANCE = (
    "The 'groq' SDK is not installed. Install it with "
    "'pip install groq' and provide a Groq API key to use "
    "GroqFactsProvider."
)

_RAW_SNIPPET_LIMIT = 500

_STRICT_KEEP_KEYS = frozenset(
    {"type", "enum", "properties", "required", "additionalProperties",
     "items", "anyOf"}
)

__all__ = [
    "DEFAULT_GROQ_MODEL",
    "GroqFactsProvider",
    "build_groq_schema",
    "to_strict_schema",
]


def _inline_refs(node: Any, defs: dict[str, Any], seen: frozenset = frozenset()) -> Any:
    """Inline local ``#/$defs/...`` references from ``defs``."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"unsupported non-local $ref: {ref!r}")
            name = ref[len("#/$defs/"):]
            if name in seen:
                raise ValueError(f"cyclic $ref: {ref!r}")
            if name not in defs:
                raise ValueError(f"unresolvable $ref: {ref!r}")
            target = {k: v for k, v in defs[name].items() if k != "$ref"}
            return _inline_refs(target, defs, seen | {name})
        return {k: _inline_refs(v, defs, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs, seen) for v in node]
    return node


def _strict_node(node: Any) -> Any:
    """Enforce strict-mode object rules; drop non-essential keywords.

    ``properties`` (and ``$defs``) map field names to subschemas, so
    their keys are preserved verbatim — only schema-keyword positions
    are filtered.
    """
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("properties", "$defs") and isinstance(value, dict):
                result[key] = {
                    name: _strict_node(sub) for name, sub in value.items()
                }
            elif key in _STRICT_KEEP_KEYS:
                result[key] = _strict_node(value)
        if result.get("type") == "object":
            props = result.get("properties")
            if props is None:
                # Free-form object (e.g. an arbitrary ``dict`` field):
                # strict mode demands ``additionalProperties: false`` on
                # every object yet rejects empty ``properties`` mappings
                # and ``required``-without-``properties`` alike, so open
                # content is unrepresentable. Emit the minimal closed
                # shape: the model cannot populate this field, and
                # Pydantic fills its default afterward. Documented
                # limitation, not silent loss (callers see the empty
                # value explicitly).
                return {"type": "object", "additionalProperties": False}
            if not isinstance(props, dict):
                raise ValueError("object schema without properties mapping")
            result["required"] = sorted(props)
            result["additionalProperties"] = False
        return result
    if isinstance(node, list):
        return [_strict_node(v) for v in node]
    return node


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema into Groq strict-mode form.

    Inlines ``$defs``, requires all properties, pins
    ``additionalProperties: false``, and drops wire-level hints strict
    mode rejects. Returns a new structure; the input is not mutated.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("strict root schema must be a JSON object schema")
    defs = dict(schema.get("$defs", {}))
    body = {k: v for k, v in schema.items() if k != "$defs"}
    return _strict_node(_inline_refs(body, defs))


def build_groq_schema() -> dict[str, Any]:
    """Build the strict response schema for ``{"drafts": [FactDraft]}``."""
    draft_schema = FactDraft.model_json_schema()
    defs = draft_schema.pop("$defs", None)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"drafts": {"type": "array", "items": draft_schema}},
        "required": ["drafts"],
    }
    if defs:
        schema["$defs"] = defs
    return to_strict_schema(schema)


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

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "fact_extraction",
                "schema": build_groq_schema(),
                "strict": True,
            },
        }
        client = Groq(api_key=self._api_key)

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            last = attempt >= attempts
            try:
                # Single SDK call site in this codebase.
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_format,
                    timeout=self._timeout_s,
                )
            except (APITimeoutError, TimeoutError) as exc:
                if last:
                    raise LLMTimeoutError(
                        f"Groq request timed out after {attempts} "
                        f"attempt(s) (model={self._model})."
                    ) from exc
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
                continue
            return FactExtractionBatch(drafts=drafts, model=self._model)
        raise AssertionError("unreachable: retry loop always returns or raises")  # pragma: no cover
