"""LLM judgment for deterministic-ambiguous pairs (task R-LLM).

Frozen rules:
- The LLM is used only for deterministic-ambiguous pairs selected by an
  earlier stage. Unrelated pairs never reach the LLM.
- Never send PDFs/documents — only two structured facts, precomputed
  verdicts, and bounded excerpts.
- The numeric comparison is precomputed deterministically; the model must
  accept it as given and must not recompute it or perform arithmetic.
- Structured output validated with Pydantic
  (:class:`RelationshipJudgment`); never default, never guess.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Protocol

from app.llm.rate_limit import (
    RateLimiter,
    compute_backoff_delay,
    estimate_tokens,
    retry_after_seconds,
)
from app.llm.provider import (
    LLMMalformedError,
    LLMTimeoutError,
    LLMTransportError,
    LLMUnavailableError,
)
from app.models.fact import Fact
from app.models.relationship import (
    ContextComparison,
    NumericVerdict,
    RelationshipJudgment,
)

DEFAULT_JUDGMENT_MODEL = "openai/gpt-oss-120b"

_SDK_INSTALL_GUIDANCE = (
    "The 'groq' SDK is not installed. Install it with "
    "'pip install groq' and provide a Groq API key to use "
    "GroqJudgmentTransport."
)

_RAW_SNIPPET_LIMIT = 500

logger = logging.getLogger(__name__)

_ALLOWED_TYPES = frozenset(
    {
        "CORROBORATES",
        "CONTRADICTS",
        "CONTEXTUAL_DIFFERENCE",
        "RELATED",
    }
)

__all__ = [
    "DEFAULT_JUDGMENT_MODEL",
    "GroqJudgmentTransport",
    "JudgmentTransport",
    "build_judgment_prompt",
    "judge_pair",
    "parse_judgment",
]


class JudgmentTransport(Protocol):
    """Tiny transport contract so tests can inject fakes without the SDK."""

    def complete_text(self, prompt: str) -> str:
        """Send a ready-made prompt and return the raw model text."""
        ...


def _render_fact(label: str, fact: Fact) -> str:
    """Render one fact's supplied fields as plain lines."""
    lines = [f"{label}:"]
    lines.append(f"  subject: {fact.subject}")
    lines.append(f"  predicate: {fact.predicate}")
    lines.append(f"  value: {fact.value_text} (kind={fact.value_kind.value})")
    if fact.value_number is not None:
        lines.append(f"  value_number: {fact.value_number}")
    lines.append(f"  unit: {fact.unit}")
    if fact.normalized_number is not None or fact.normalized_unit is not None:
        lines.append(
            "  normalized: "
            f"{fact.normalized_number} {fact.normalized_unit}".strip()
        )
    lines.append(f"  time: {fact.time_text} (kind={fact.time_kind.value})")
    if fact.time_start is not None or fact.time_end is not None:
        lines.append(f"  time_range: {fact.time_start} to {fact.time_end}")
    lines.append(f"  scope: {fact.scope_text}")
    lines.append(f"  estimate: {fact.estimate_status.value}")
    lines.append(f"  geography: {fact.geography}")
    lines.append(f"  confidence: {fact.extraction_confidence}")
    return "\n".join(lines)


def _truncate(snippet: str, limit: int) -> str:
    """Truncate one excerpt to at most ``limit`` characters."""
    if not isinstance(snippet, str):
        snippet = str(snippet)
    if limit is not None and limit >= 0 and len(snippet) > limit:
        return snippet[:limit]
    return snippet


def build_judgment_prompt(
    fact_a: Fact,
    fact_b: Fact,
    numeric_verdict: NumericVerdict,
    context: ContextComparison,
    evidence_a: list[str],
    evidence_b: list[str],
    max_snippet_chars: int = 500,
) -> str:
    """Render the judgment prompt for one ambiguous pair.

    Only structured fields, precomputed verdicts, and bounded excerpts
    are included — never source documents.
    """
    dimension_lines = []
    for name in sorted(context.dimensions):
        dimension_lines.append(f"  - {name}: {context.dimensions[name].value}")
    dimensions_block = (
        "\n".join(dimension_lines) if dimension_lines else "  (none)"
    )

    snippets_a = [_truncate(s, max_snippet_chars) for s in (evidence_a or [])]
    snippets_b = [_truncate(s, max_snippet_chars) for s in (evidence_b or [])]
    excerpts_a = (
        "\n".join(f"  [{i + 1}] {s}" for i, s in enumerate(snippets_a))
        if snippets_a
        else "  (none)"
    )
    excerpts_b = (
        "\n".join(f"  [{i + 1}] {s}" for i, s in enumerate(snippets_b))
        if snippets_b
        else "  (none)"
    )

    return "\n".join(
        [
            "You are judging the relationship between two structured facts.",
            "This step applies only to pairs already selected as potentially",
            "related by an earlier stage.",
            "",
            _render_fact("Fact A", fact_a),
            "",
            _render_fact("Fact B", fact_b),
            "",
            f"Numeric comparison (precomputed): {numeric_verdict.value}",
            "This numeric comparison was already computed deterministically",
            "and is given as final — do NOT recompute or dispute it.",
            "Do not perform any arithmetic. Accept it as given and combine",
            "it with the fields below to reach your judgment.",
            "",
            "Context comparison (precomputed per-dimension verdicts):",
            dimensions_block,
            f"Incompatible dimensions: {context.incompatible}",
            f"Unknown dimensions: {context.unknown}",
            "Unknown means unknown — never treat it as incompatibility.",
            "",
            "Supporting excerpts (bounded, truncated to "
            f"{max_snippet_chars} characters each):",
            "Excerpts for Fact A:",
            excerpts_a,
            "Excerpts for Fact B:",
            excerpts_b,
            "",
            "Instructions:",
            "- Choose exactly one relationship type from: CORROBORATES,",
            "  CONTRADICTS, CONTEXTUAL_DIFFERENCE, or RELATED.",
            "- UNRELATED is not an option — unrelated pairs never reach",
            "  this step, so every pair seen here is already known to be",
            "  related at some level.",
            "- Base your explanation ONLY on the supplied fields and the",
            "  excerpts above. Do not introduce outside information.",
            "- Do not recompute the numeric comparison and do not perform",
            "  arithmetic of any kind.",
            "- Respond with exactly one JSON object and nothing else:",
            '  {"relationship_type": "...", "confidence": <number 0..1>,',
            '   "explanation": "..."}.',
            "- relationship_type must be one of the four allowed values.",
            "- confidence is your certainty in the chosen type, from 0 to 1.",
            "- explanation must be a non-empty string citing only the",
            "  supplied fields and excerpts.",
        ]
    )


def _load_payload(raw_text: str) -> object:
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


def _snippet_of(raw_text: object) -> str:
    if isinstance(raw_text, str):
        return raw_text[:_RAW_SNIPPET_LIMIT]
    return ""


def parse_judgment(raw_text: str) -> RelationshipJudgment:
    """Validate raw model text into a :class:`RelationshipJudgment`.

    Raises:
        LLMMalformedError: on empty text, JSON-decode failure, wrong
            shape, disallowed relationship type (including UNRELATED),
            non-numeric or out-of-range confidence, or empty explanation.
            The message carries a raw snippet truncated to 500 chars.
            Never defaults, never guesses.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LLMMalformedError(
            "judgment output is empty "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )
    try:
        payload = _load_payload(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMMalformedError(
            f"judgment output is not valid JSON: {exc} "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise LLMMalformedError(
            "judgment output must be a JSON object with "
            "'relationship_type', 'confidence', and 'explanation' "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )
    missing = [
        key
        for key in ("relationship_type", "confidence", "explanation")
        if key not in payload
    ]
    if missing:
        raise LLMMalformedError(
            f"judgment output is missing keys: {missing} "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )

    rel_type = payload["relationship_type"]
    if not isinstance(rel_type, str) or rel_type.strip() not in _ALLOWED_TYPES:
        raise LLMMalformedError(
            "judgment 'relationship_type' must be one of "
            f"{sorted(_ALLOWED_TYPES)} (UNRELATED is not allowed here); "
            f"got {rel_type!r} "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )

    raw_conf = payload["confidence"]
    if isinstance(raw_conf, bool):
        raise LLMMalformedError(
            "judgment 'confidence' must be a number in 0..1; "
            f"got {raw_conf!r} "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )
    try:
        confidence = (
            float(raw_conf)
            if isinstance(raw_conf, (int, float))
            else float(str(raw_conf).strip())
            if isinstance(raw_conf, str) and str(raw_conf).strip() != ""
            else (_ for _ in ()).throw(
                ValueError(f"non-numeric confidence: {raw_conf!r}")
            )
        )
    except (TypeError, ValueError) as exc:
        raise LLMMalformedError(
            "judgment 'confidence' must be a number in 0..1; "
            f"got {raw_conf!r} "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        ) from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise LLMMalformedError(
            "judgment 'confidence' must be a finite number in 0..1; "
            f"got {raw_conf!r} "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )

    explanation = payload["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise LLMMalformedError(
            "judgment 'explanation' must be a non-empty string "
            f"| raw snippet: {_snippet_of(raw_text)!r}"
        )

    return RelationshipJudgment(
        relationship_type=rel_type.strip(),
        confidence=confidence,
        explanation=explanation,
    )


class GroqJudgmentTransport:
    """Groq text transport for judgment prompts.

    The constructor stores parameters only (no client construction, no
    network). All SDK surface lives in :meth:`complete_text`. Output is
    parsed as free-form JSON by :func:`parse_judgment` (same contract as
    before — structured enforcement happens in validation, not here).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_JUDGMENT_MODEL,
        timeout_s: int = 60,
        max_retries: int = 2,
        limiter: RateLimiter | None = None,
        expected_output_tokens: int = 1000,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._limiter = limiter
        self._expected_output_tokens = max(0, int(expected_output_tokens))
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
            LLMUnavailableError: when the ``groq`` SDK is missing.
            LLMTimeoutError: when SDK calls time out on every attempt.
            LLMTransportError: when SDK calls fail or return empty text
                on every attempt, or immediately on deterministic 4xx
                failures (auth, bad request, permission, not-found).
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
            self._limiter.acquire(
                estimate_tokens(prompt) + self._expected_output_tokens
            )

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
                # Single SDK call site in this module.
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self._timeout_s,
                )
            except (APITimeoutError, TimeoutError) as exc:
                if last:
                    raise LLMTimeoutError(
                        f"Groq judgment request timed out after {attempts} "
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
                raise LLMTransportError(
                    f"Groq judgment deterministic API error, not retried "
                    f"(model={self._model}): {exc}"
                ) from exc
            except Exception as exc:
                if last:
                    raise LLMTransportError(
                        f"Groq judgment transport/API error after {attempts} "
                        f"attempt(s) (model={self._model}): {exc}"
                    ) from exc
                _backoff(attempt, exc)
                continue
            text = getattr(getattr(response, "choices", [None])[0], "message", None)
            text = getattr(text, "content", "") if text is not None else ""
            if not isinstance(text, str) or not text.strip():
                exc: Exception = LLMTransportError("Groq returned empty text.")
                if last:
                    raise LLMTransportError(
                        f"Groq judgment returned empty text after {attempts} "
                        f"attempt(s) (model={self._model})."
                    ) from exc
                _backoff(attempt, exc)
                continue
            return text
        raise AssertionError("unreachable: retry loop always returns or raises")  # pragma: no cover


def judge_pair(
    transport: JudgmentTransport,
    fact_a: Fact,
    fact_b: Fact,
    numeric_verdict: NumericVerdict,
    context: ContextComparison,
    evidence_a: list[str],
    evidence_b: list[str],
) -> RelationshipJudgment:
    """Judge one ambiguous pair: build prompt, call transport, parse output."""
    prompt = build_judgment_prompt(
        fact_a, fact_b, numeric_verdict, context, evidence_a, evidence_b
    )
    raw_text = transport.complete_text(prompt)
    return parse_judgment(raw_text)
