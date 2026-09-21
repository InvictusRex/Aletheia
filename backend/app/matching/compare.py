"""Deterministic context comparison + relationship classification (R-CMP).

This module types pairs of :class:`~app.models.fact.Fact` objects without any
learned similarity score: every decision is a pure function of normalized
values, normalized units, and explicitly recorded context fields. Similarity
scores are not an input anywhere in this file.

Tri-state rule (applies to every context dimension):

* ``COMPATIBLE`` -- both sides present and agreeing.
* ``INCOMPATIBLE`` -- both sides present (known) and disagreeing.
* ``UNKNOWN`` -- either side missing/unknown. Missing data is never treated
  as disagreement, and ``UNKNOWN`` alone never yields ``CONTEXTUAL_DIFFERENCE``.

Frozen precedence (``classify`` implements exactly this order):

* (a) numeric-equivalent + compatible context + strong match -> ``CORROBORATES``
* (a2) numeric-equivalent + no NON-NAMING incompatible dim + weak match
  -> ``CORROBORATES`` (reviewable: confidence <=0.6, ``needs_llm``)
* (b) materially-different + zero ``INCOMPATIBLE`` dims + strong match
  -> ``CONTRADICTS`` (unknowns recorded, confidence capped at 0.7)
* (b2) materially-different + no NON-NAMING incompatible dim + weak match
  -> ``CONTRADICTS`` (reviewable: confidence <=0.6, ``needs_llm``)
* (c) >=1 ``KNOWN`` differing dimension explaining the gap + claims_comparable
  gate + grounded comparison -> ``CONTEXTUAL_DIFFERENCE``
* (d) partial semantic overlap via the claims_comparable gate + grounded
  comparison -> ``RELATED``, else ``UNRELATED``
* (e) otherwise -> ``UNRELATED``

Match tiers. ``strong_match`` requires exact canonical subject AND predicate
equality; it is the only tier that yields a confident verdict. Independently
extracted documents almost never phrase a claim identically, so that tier
alone leaves corroboration and contradiction unreachable in practice. The
``weak_match`` tier compares the COMBINED subject+predicate phrase, because
the extractor splits a claim between those two fields arbitrarily, and the
naming dimensions (entity, predicate) are then excluded from the context
test -- a different wording for the same metric is naming variance, not a
contextual reason for a value gap. A weak-tier verdict is always reviewable
(``needs_llm=True``, confidence <=0.6) and never asserted.

Grounded means comparable normalized numbers (``EXACT``/``APPROXIMATE``/
``DIFFERENT``) or the same claim on both sides (shared canonical subject
and predicate, or identical effective subject and predicate): text-only
pairs about different claims are never typed, no matter the token
overlap.

``claims_comparable`` compares content tokens only: generic scaffolding
tokens (copulas, prepositions, articles, and extraction-filler words such
as "value" or "reported") never count toward overlap, so pairs sharing
only boilerplate are not comparable. ``UNRELATED`` pairs are counted but
never persisted.

Two genuinely ambiguous configurations return a provisional ``RELATED`` with
``needs_llm=True`` so the judgment layer can arbitrate; every other outcome
is final (``needs_llm=False``).

Standard library only.
"""

from __future__ import annotations

import re

from app.models.fact import Comparability, EstimateStatus, Fact, TimeKind, ValueKind
from app.models.relationship import (
    ContextComparison,
    DimensionVerdict,
    NumericVerdict,
    RelationshipType,
)

RELATIVE_TOLERANCE = 0.01
"""Maximum relative difference for two numbers to count as approximate.

Justification: a 1% band absorbs display/rounding precision (one unit in the
last shown digit of a two-to-three significant-figure rendering) while still
flagging genuine restatements as different. A relative difference of exactly
0.0 is always EXACT, never merely approximate.
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_TIME_KIND_LABELS = {
    TimeKind.FISCAL_YEAR: "fiscal",
    TimeKind.QUARTER: "quarter",
    TimeKind.DATE: "date",
    TimeKind.RANGE: "range",
    TimeKind.UNKNOWN: "unknown",
}

_SEMANTIC_OVERLAP_THRESHOLD = 0.2

_SCAFFOLDING_STOPWORDS = frozenset({
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "of", "from", "in", "on", "at", "to", "for", "with", "by", "as",
    "and", "or", "it", "its", "this", "that", "these", "those",
    "value", "values", "amount", "amounts", "number", "numbers",
    "total", "totals", "status", "reported", "report", "stated",
})

_DIMENSION_ORDER = (
    "entity",
    "predicate",
    "unit",
    "time",
    "estimate_status",
    "geography",
    "scope",
)

_COMPATIBLE = DimensionVerdict.COMPATIBLE
_INCOMPATIBLE = DimensionVerdict.INCOMPATIBLE
_UNKNOWN = DimensionVerdict.UNKNOWN


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _relative_difference(x: float, y: float) -> float:
    """Symmetric relative difference ``|x - y| / max(|x|, |y|)``.

    Both-zero yields 0.0 (no division by zero). The denominator uses the
    larger magnitude so the measure is symmetric in its arguments.
    """
    if x == 0.0 and y == 0.0:
        return 0.0
    denom = max(abs(x), abs(y))
    if denom == 0.0:  # pragma: no cover -- defensive; both-zero handled above.
        return 0.0
    return abs(x - y) / denom


def _effective_entity(fact: Fact) -> str | None:
    return fact.canonical_subject or fact.subject or None


def _effective_predicate(fact: Fact) -> str | None:
    return fact.canonical_predicate or fact.predicate or None


def _effective_unit(fact: Fact) -> str | None:
    if fact.normalized_unit is not None:
        return fact.normalized_unit
    return fact.unit


def _present(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _equality_verdict(a_value: str | None, b_value: str | None) -> DimensionVerdict:
    """Casefolded equality with unknown-passthrough.

    Either side missing (``None``/blank) -> ``UNKNOWN``; both present and
    casefold-equal -> ``COMPATIBLE``; both present but different ->
    ``INCOMPATIBLE``.
    """
    if not _present(a_value) or not _present(b_value):
        return _UNKNOWN
    if a_value.strip().casefold() == b_value.strip().casefold():
        return _COMPATIBLE
    return _INCOMPATIBLE


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------


def compare_numeric(
    a: Fact, b: Fact, tolerance: float = RELATIVE_TOLERANCE
) -> NumericVerdict:
    """Compare the normalized numbers of two facts.

    Requires both ``normalized_number`` values to be set and both
    ``normalized_unit`` values to be equal non-``None`` strings; anything
    else (``TEXT`` kind on either side, a missing number, or a unit mismatch
    -- including ``None`` vs ``None``) is ``INCOMPARABLE``. Units are
    compared with exact string equality because they are already normalized
    upstream.

    With comparable numbers: relative difference 0 (including both-zero,
    which short-circuits before any division) -> ``EXACT``; relative
    difference within ``RELATIVE_TOLERANCE`` -> ``APPROXIMATE``; otherwise
    -> ``DIFFERENT``.
    """
    if a.value_kind == ValueKind.TEXT or b.value_kind == ValueKind.TEXT:
        return NumericVerdict.INCOMPARABLE
    if a.normalized_number is None or b.normalized_number is None:
        return NumericVerdict.INCOMPARABLE
    if a.normalized_unit is None or b.normalized_unit is None:
        return NumericVerdict.INCOMPARABLE
    if a.normalized_unit != b.normalized_unit:
        return NumericVerdict.INCOMPARABLE
    if a.normalized_number == 0.0 and b.normalized_number == 0.0:
        return NumericVerdict.EXACT
    rel = _relative_difference(a.normalized_number, b.normalized_number)
    if rel == 0.0:
        return NumericVerdict.EXACT
    if rel <= tolerance:
        return NumericVerdict.APPROXIMATE
    return NumericVerdict.DIFFERENT


# ---------------------------------------------------------------------------
# Context comparison
# ---------------------------------------------------------------------------


def _compare_time(a: Fact, b: Fact) -> DimensionVerdict:
    """Compare time references: kind first, then text, then date ranges.

    * Either side ``UNKNOWN``/missing -> ``UNKNOWN``.
    * Different known ``time_kind`` values -> ``INCOMPATIBLE``. Fiscal labels
      therefore never equal calendar dates: ``FISCAL_YEAR`` vs ``DATE`` is
      incompatible even when the label text looks similar, because kind is
      compared alongside text.
    * Same known kind: identical non-blank ``time_text`` (casefolded) or
      overlapping ``[time_start, time_end]`` ranges (all four bounds known,
      inclusive overlap) -> ``COMPATIBLE``.
    * Same known kind with fully known but disjoint ranges, or with both
      labels present but different -> ``INCOMPATIBLE``. A dateless fiscal
      reference carries no dates, so it can never overlap: it is compatible
      only via identical label text, incompatible when both labels are known
      and differ, and unknown otherwise.
    * Anything else (a label missing on either side and no decisive ranges)
      -> ``UNKNOWN``.
    """
    if a.time_kind == TimeKind.UNKNOWN or b.time_kind == TimeKind.UNKNOWN:
        return _UNKNOWN
    if a.time_kind != b.time_kind:
        return _INCOMPATIBLE
    a_text = a.time_text.strip().casefold() if _present(a.time_text) else None
    b_text = b.time_text.strip().casefold() if _present(b.time_text) else None
    if a_text is not None and b_text is not None and a_text == b_text:
        return _COMPATIBLE
    ranges_known = (
        a.time_start is not None
        and a.time_end is not None
        and b.time_start is not None
        and b.time_end is not None
    )
    if ranges_known:
        assert a.time_start is not None and a.time_end is not None
        assert b.time_start is not None and b.time_end is not None
        if max(a.time_start, b.time_start) <= min(a.time_end, b.time_end):
            return _COMPATIBLE
        return _INCOMPATIBLE
    if a_text is not None and b_text is not None:
        return _INCOMPATIBLE
    return _UNKNOWN


def _compare_estimate_status(a: Fact, b: Fact) -> DimensionVerdict:
    """Compare estimate status; ``UNKNOWN`` on either side -> ``UNKNOWN``."""
    a_status = a.estimate_status if a.estimate_status is not None else EstimateStatus.UNKNOWN
    b_status = b.estimate_status if b.estimate_status is not None else EstimateStatus.UNKNOWN
    if a_status == EstimateStatus.UNKNOWN or b_status == EstimateStatus.UNKNOWN:
        return _UNKNOWN
    if a_status == b_status:
        return _COMPATIBLE
    return _INCOMPATIBLE


def compare_context(a: Fact, b: Fact) -> ContextComparison:
    """Compare per-dimension context with tri-state verdicts.

    Dimensions (in fixed order): entity (``canonical_subject`` or
    ``subject``), predicate (``canonical_predicate`` or ``predicate``), unit
    (``normalized_unit`` or ``unit``), time, estimate status, geography, and
    scope (``scope_text``). Entity, predicate, unit, geography, and scope use
    casefolded equality with unknown-passthrough: both present and equal ->
    ``COMPATIBLE``; both present but different -> ``INCOMPATIBLE``; either
    side missing -> ``UNKNOWN``, never ``INCOMPATIBLE``.
    """
    dimensions = {
        "entity": _equality_verdict(_effective_entity(a), _effective_entity(b)),
        "predicate": _equality_verdict(_effective_predicate(a), _effective_predicate(b)),
        "unit": _equality_verdict(_effective_unit(a), _effective_unit(b)),
        "time": _compare_time(a, b),
        "estimate_status": _compare_estimate_status(a, b),
        "geography": _equality_verdict(a.geography, b.geography),
        "scope": _equality_verdict(a.scope_text, b.scope_text),
    }
    return ContextComparison(dimensions=dimensions)


# ---------------------------------------------------------------------------
# Alignment signals
# ---------------------------------------------------------------------------


CLAIM_MATCH_THRESHOLD = 0.6
"""Minimum combined-claim overlap for a same-claim candidate.

Measured on the starter corpus: every threshold from 0.99 down to 0.4
surfaces the same pairs when subject and predicate are compared
separately, because the extractor splits a claim between the two fields
arbitrarily ("India" / "registered a growth rate of" vs "India" /
"growth"). Comparing the combined phrase is what separates them, and 0.6
is where real restatements appear while unrelated claims sharing one
noun do not. A match at this tier is never final on its own: it yields a
reviewable verdict, not a confident one.
"""


def claim_similarity(a: Fact, b: Fact) -> float:
    """Token-Jaccard over the combined subject+predicate content phrase.

    Scaffolding stopwords are excluded so two facts sharing only filler
    never score. Canonical forms are preferred, raw text is the fallback.
    """
    a_tokens = _content_tokens(f"{_effective_entity(a) or ''} {_effective_predicate(a) or ''}")
    b_tokens = _content_tokens(f"{_effective_entity(b) or ''} {_effective_predicate(b) or ''}")
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


SUBJECT_MATCH_THRESHOLD = 0.5
"""Minimum subject overlap before a combined-phrase match is believed.

The combined phrase alone is not enough: two facts about different
entities that share a generic metric name ("Barasia" / "percentage of
total share capital" vs "Company" / "share capital percentage") reach
0.6 on the combined tokens while naming nothing in common. Requiring the
subjects themselves to overlap separates a reworded claim about one
entity from unrelated claims about two.
"""


def weak_match(a: Fact, b: Fact) -> bool:
    """Same claim by combined phrasing, but not an exact canonical match.

    Requires the same value kind, the same effective unit, subjects that
    overlap on their own, and a combined-phrase match, so only the
    wording differs. Used to surface restatements that ``strong_match``
    misses; the caller must keep the resulting verdict reviewable rather
    than confident.
    """
    if strong_match(a, b):
        return False
    # Both sides must be quantities, but NUMERIC and PERCENTAGE may mix:
    # one source writes "10.1 per cent" and another writes "25.4" with
    # unit "per cent", so the kind records how the extractor saw it
    # rather than what the value is. Once both normalize to the same
    # unit they are the same kind of quantity, and requiring the raw
    # kinds to agree would reject exactly the cross-source restatements
    # this tier exists to find.
    if not {a.value_kind, b.value_kind} <= {ValueKind.NUMERIC, ValueKind.PERCENTAGE}:
        return False
    if _effective_unit(a) != _effective_unit(b):
        return False
    a_subject = _content_tokens(_effective_entity(a) or "")
    b_subject = _content_tokens(_effective_entity(b) or "")
    if not a_subject or not b_subject:
        return False
    subject_overlap = len(a_subject & b_subject) / len(a_subject | b_subject)
    if subject_overlap < SUBJECT_MATCH_THRESHOLD:
        return False
    return claim_similarity(a, b) >= CLAIM_MATCH_THRESHOLD


def strong_match(a: Fact, b: Fact) -> bool:
    """Whether two facts align on entity, predicate, kind, and unit.

    Requires strictly equal (exact string equality) non-null
    ``canonical_subject`` values, strictly equal non-null
    ``canonical_predicate`` values, equal ``value_kind``, and equal
    (``normalized_unit`` or ``unit``) effective units where both-``None``
    counts as equal. Canonical fields are compared exactly because they are
    expected to be canonicalized upstream.
    """
    if a.canonical_subject is None or b.canonical_subject is None:
        return False
    if a.canonical_subject != b.canonical_subject:
        return False
    if a.canonical_predicate is None or b.canonical_predicate is None:
        return False
    if a.canonical_predicate != b.canonical_predicate:
        return False
    if a.value_kind != b.value_kind:
        return False
    return _effective_unit(a) == _effective_unit(b)


def semantic_overlap(a: Fact, b: Fact) -> bool:
    """Whether two facts share partial subject/predicate semantics.

    True when the token-Jaccard similarity over (subject + predicate),
    lowercased alphanumeric tokens with no stopword removal, is >= 0.2, or
    when both facts share a non-blank casefold-equal ``canonical_subject``
    or ``canonical_predicate``.
    """
    a_tokens = _tokens(f"{a.subject} {a.predicate}")
    b_tokens = _tokens(f"{b.subject} {b.predicate}")
    union = a_tokens | b_tokens
    if union and len(a_tokens & b_tokens) / len(union) >= _SEMANTIC_OVERLAP_THRESHOLD:
        return True
    for a_canon, b_canon in (
        (a.canonical_subject, b.canonical_subject),
        (a.canonical_predicate, b.canonical_predicate),
    ):
        if _present(a_canon) and _present(b_canon):
            assert a_canon is not None and b_canon is not None
            if a_canon.strip().casefold() == b_canon.strip().casefold():
                return True
    return False


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _SCAFFOLDING_STOPWORDS}


def claims_comparable(a: Fact, b: Fact) -> bool:
    a_tokens = _content_tokens(f"{a.subject} {a.predicate}")
    b_tokens = _content_tokens(f"{b.subject} {b.predicate}")
    union = a_tokens | b_tokens
    if not union:
        return False
    return len(a_tokens & b_tokens) / len(union) >= _SEMANTIC_OVERLAP_THRESHOLD


def _shared_canonical_claim(a: Fact, b: Fact) -> bool:
    if not _present(a.canonical_subject) or not _present(b.canonical_subject):
        return False
    if not _present(a.canonical_predicate) or not _present(b.canonical_predicate):
        return False
    assert a.canonical_subject is not None and b.canonical_subject is not None
    assert a.canonical_predicate is not None and b.canonical_predicate is not None
    return (
        a.canonical_subject.strip().casefold() == b.canonical_subject.strip().casefold()
        and a.canonical_predicate.strip().casefold()
        == b.canonical_predicate.strip().casefold()
    )


def _same_raw_claim(a: Fact, b: Fact) -> bool:
    a_entity, b_entity = _effective_entity(a), _effective_entity(b)
    a_pred, b_pred = _effective_predicate(a), _effective_predicate(b)
    if not all(map(_present, (a_entity, b_entity, a_pred, b_pred))):
        return False
    assert a_entity is not None and b_entity is not None
    assert a_pred is not None and b_pred is not None
    return (
        a_entity.strip().casefold() == b_entity.strip().casefold()
        and a_pred.strip().casefold() == b_pred.strip().casefold()
    )


def _same_claim(a: Fact, b: Fact) -> bool:
    return _shared_canonical_claim(a, b) or _same_raw_claim(a, b)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _contradiction_blocked_reason(
    a: Fact, b: Fact, ctx: ContextComparison
) -> str | None:
    """Why this pair may not be called a contradiction, or None.

    Three preconditions, each learned from a false positive this system
    actually produced:

    * both sides must be COMPARABLE -- a fact without a canonical unit or
      a typed period cannot support a numeric disagreement;
    * the periods must be known AND equal -- an unknown period is not the
      same period, it is no information;
    * scope must be symmetric -- when one side says "standalone" and the
      other says nothing, the pair may be about different things.
    """
    if a.comparability != Comparability.COMPARABLE:
        return f"fact A is not comparable (missing: {', '.join(a.missing_for_comparison)})"
    if b.comparability != Comparability.COMPARABLE:
        return f"fact B is not comparable (missing: {', '.join(b.missing_for_comparison)})"
    if ctx.dimensions.get("time") != _COMPATIBLE:
        return "period not stated and equal on both sides"
    if _present(a.scope_text) != _present(b.scope_text):
        return "scope stated on only one side"
    return None


def _dim_display(dim: str, a: Fact, b: Fact) -> tuple[str, str]:
    """Raw display values for one dimension on each side."""

    def show(value: object) -> str:
        if value is None:
            return "missing"
        if isinstance(value, str) and not value.strip():
            return "missing"
        return repr(value)

    if dim == "entity":
        return (show(_effective_entity(a)), show(_effective_entity(b)))
    if dim == "predicate":
        return (show(_effective_predicate(a)), show(_effective_predicate(b)))
    if dim == "unit":
        return (show(_effective_unit(a)), show(_effective_unit(b)))
    if dim == "time":
        a_show = (
            f"{_TIME_KIND_LABELS[a.time_kind]}:{show(a.time_text)}:"
            f"{show(a.time_start)}..{show(a.time_end)}"
        )
        b_show = (
            f"{_TIME_KIND_LABELS[b.time_kind]}:{show(b.time_text)}:"
            f"{show(b.time_start)}..{show(b.time_end)}"
        )
        return (a_show, b_show)
    if dim == "estimate_status":
        a_status = a.estimate_status if a.estimate_status is not None else EstimateStatus.UNKNOWN
        b_status = b.estimate_status if b.estimate_status is not None else EstimateStatus.UNKNOWN
        return (a_status.value, b_status.value)
    if dim == "geography":
        return (show(a.geography), show(b.geography))
    return (show(a.scope_text), show(b.scope_text))  # scope


def _numbers_clause(a: Fact, b: Fact, verdict: NumericVerdict) -> str:
    """Deterministic numeric evidence clause for explanations."""
    if verdict == NumericVerdict.INCOMPARABLE:
        return (
            f"numeric verdict {verdict.value} "
            "(no shared normalized number and unit to compare)"
        )
    rel = _relative_difference(
        float(a.normalized_number or 0.0), float(b.normalized_number or 0.0)
    )
    if a.normalized_number == 0.0 and b.normalized_number == 0.0:
        rel = 0.0
    return (
        f"normalized {a.normalized_number} vs {b.normalized_number} "
        f"{a.normalized_unit}; numeric verdict {verdict.value}, "
        f"relative difference {rel:.4f}"
    )


def _dims_clause(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _explain(
    outcome: str,
    a: Fact,
    b: Fact,
    verdict: NumericVerdict,
    ctx: ContextComparison,
    strong: bool,
    overlap: bool,
    needs_llm: bool,
) -> str:
    """Build the deterministic explanation for a classification outcome."""
    compatible = sorted(
        name for name, v in ctx.dimensions.items() if v == _COMPATIBLE
    )
    numbers = _numbers_clause(a, b, verdict)
    values = f"value {a.value_text!r} and value {b.value_text!r}"
    dims = (
        f"Compatible dimensions: {_dims_clause(compatible)}. "
        f"Incompatible dimensions: {_dims_clause(ctx.incompatible)}. "
        f"Unknown dimensions: {_dims_clause(ctx.unknown)}."
    )
    if outcome == "corroborates":
        return (
            f"Corroboration: {values} agree ({numbers}). {dims} "
            "No incompatible dimension; entity, predicate, and unit alignment holds."
        )
    if outcome == "contradicts":
        capped = (
            " Confidence is capped because unknown dimensions are recorded."
            if ctx.unknown
            else ""
        )
        return (
            f"Contradiction: {values} differ materially ({numbers}) with no "
            f"incompatible context dimension. {dims} Entity, predicate, and unit "
            f"alignment holds.{capped}"
        )
    if outcome == "contextual_difference":
        details = ", ".join(
            f"{name} ({_dim_display(name, a, b)[0]} vs {_dim_display(name, a, b)[1]})"
            for name in ctx.incompatible
        )
        return (
            f"Contextual difference: {values} cannot be directly equated "
            f"({numbers}). Known context differs on: {details}. Shared semantics "
            f"(strong match: {strong}; token overlap: {overlap}) attribute the gap "
            "to differing context rather than direct disagreement."
        )
    if outcome == "related":
        provisional = (
            " Provisional typing pending further review." if needs_llm else ""
        )
        return (
            f"Related: {values} share partial semantic overlap but meet neither "
            f"corroboration nor contradiction criteria ({numbers}). {dims}{provisional}"
        )
    return (
        f"Unrelated: {values} show no qualifying alignment ({numbers}). {dims} "
        "No shared entity/predicate semantics."
    )


def classify(
    a: Fact, b: Fact, tolerance: float = RELATIVE_TOLERANCE
) -> tuple[RelationshipType, float, str, dict, bool]:
    """Classify the relationship between two facts.

    Implements the frozen precedence ``(a)``-``(e)`` using only
    :func:`compare_numeric`, :func:`compare_context`, :func:`strong_match`,
    and :func:`semantic_overlap` -- no similarity scores enter here.

    Returns ``(type, confidence, explanation, metadata, needs_llm)`` where
    metadata always carries ``numeric_verdict``, ``incompatible_dimensions``,
    and ``unknown_dimensions``. Confidences: ``CORROBORATES`` 0.95
    (exact) / 0.9 (approximate); ``CONTRADICTS`` 0.85, capped at 0.7 when any
    unknown dimension is recorded; ``CONTEXTUAL_DIFFERENCE`` 0.75;
    ``RELATED`` 0.5; ``UNRELATED`` 0.9. ``needs_llm`` is True only for (i)
    numeric EXACT/APPROXIMATE with compatible context but overlap-only
    alignment (not a strong match), and (ii) numeric INCOMPARABLE with a
    strong match and zero incompatible dimensions; both yield a provisional
    ``RELATED``.
    """
    verdict = compare_numeric(a, b, tolerance)
    ctx = compare_context(a, b)
    incompatible = ctx.incompatible
    unknown = ctx.unknown
    strong = strong_match(a, b)
    weak = weak_match(a, b)
    overlap = semantic_overlap(a, b)
    comparable = claims_comparable(a, b)
    grounded = (
        verdict != NumericVerdict.INCOMPARABLE
        or _same_claim(a, b)
    )

    numeric_equivalent = verdict in (NumericVerdict.EXACT, NumericVerdict.APPROXIMATE)
    materially_different = verdict == NumericVerdict.DIFFERENT

    # Entity/predicate disagreement is naming variance, not a contextual
    # dimension like time or scope. When the combined claim already
    # matches, those two must not be read as context that explains a
    # value gap -- otherwise every restatement of the same metric is
    # filed as a contextual difference and neither corroboration nor
    # contradiction is ever reachable across differently-worded sources.
    context_incompatible = [
        dim for dim in incompatible if dim not in ("entity", "predicate")
    ]

    # PLAN 2.4 requires the same entity, metric, TIME and compatible
    # scope before two values may be called a disagreement. Everything
    # this gate rejects is recorded as a withheld verdict rather than
    # dropped, so the candidate stays inspectable.
    blocked_reason = _contradiction_blocked_reason(a, b, ctx)

    if numeric_equivalent and not incompatible:
        if strong:
            confidence = 0.95 if verdict == NumericVerdict.EXACT else 0.9
            explanation = _explain(
                "corroborates", a, b, verdict, ctx, strong, overlap, False
            )
            return (
                RelationshipType.CORROBORATES,
                confidence,
                explanation,
                _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
                False,
            )
        if overlap:
            explanation = _explain(
                "related", a, b, verdict, ctx, strong, overlap, True
            )
            return (
                RelationshipType.RELATED,
                0.5,
                explanation,
                _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
                True,
            )
    if numeric_equivalent and not context_incompatible and weak:
        # Same claim by combined phrasing, equivalent values, no real
        # context conflict: a restatement of one fact in different words.
        # Inexact match, so reviewable rather than asserted.
        explanation = _explain(
            "corroborates", a, b, verdict, ctx, strong, overlap, True
        )
        metadata = _metadata(a, b, verdict, incompatible, unknown, strong, overlap)
        metadata["claim_similarity"] = round(claim_similarity(a, b), 4)
        metadata["match_tier"] = "weak"
        return (
            RelationshipType.CORROBORATES,
            0.5 if unknown else 0.6,
            explanation,
            metadata,
            True,
        )
    if materially_different and not incompatible and strong:
        if blocked_reason is not None:
            # An exact canonical match on entity and metric is not enough:
            # "active customers 33,250 (FY24)" and "7,900 (nine months
            # ended December 2021)" are the same claim about different
            # years, not a disagreement.
            explanation = _explain(
                "related", a, b, verdict, ctx, strong, overlap, True
            )
            metadata = _metadata(a, b, verdict, incompatible, unknown, strong, overlap)
            metadata["withheld_verdict"] = "CONTRADICTS"
            metadata["withheld_reason"] = blocked_reason
            return (
                RelationshipType.RELATED,
                0.4,
                explanation,
                metadata,
                True,
            )
        confidence = 0.7 if unknown else 0.85
        explanation = _explain(
            "contradicts", a, b, verdict, ctx, strong, overlap, False
        )
        return (
            RelationshipType.CONTRADICTS,
            confidence,
            explanation,
            _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
            False,
        )
    if materially_different and not context_incompatible and weak:
        if blocked_reason is not None:
            explanation = _explain(
                "related", a, b, verdict, ctx, strong, overlap, True
            )
            metadata = _metadata(a, b, verdict, incompatible, unknown, strong, overlap)
            metadata["claim_similarity"] = round(claim_similarity(a, b), 4)
            metadata["match_tier"] = "weak"
            metadata["withheld_verdict"] = "CONTRADICTS"
            metadata["withheld_reason"] = blocked_reason
            return (
                RelationshipType.RELATED,
                0.4,
                explanation,
                metadata,
                True,
            )
        # Same claim by combined phrasing but not by exact canonical
        # strings. Materially different values with no incompatible
        # context is the shape of a contradiction, but the match itself
        # is inexact, so this is surfaced for review (and for the
        # judgment layer) rather than asserted.
        explanation = _explain(
            "contradicts", a, b, verdict, ctx, strong, overlap, True
        )
        metadata = _metadata(a, b, verdict, incompatible, unknown, strong, overlap)
        metadata["claim_similarity"] = round(claim_similarity(a, b), 4)
        metadata["match_tier"] = "weak"
        return (
            RelationshipType.CONTRADICTS,
            0.5 if unknown else 0.6,
            explanation,
            metadata,
            True,
        )
    if incompatible and comparable and grounded:
        explanation = _explain(
            "contextual_difference", a, b, verdict, ctx, strong, overlap, False
        )
        return (
            RelationshipType.CONTEXTUAL_DIFFERENCE,
            0.75,
            explanation,
            _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
            False,
        )
    if verdict == NumericVerdict.INCOMPARABLE and not incompatible and strong:
        explanation = _explain(
            "related", a, b, verdict, ctx, strong, overlap, True
        )
        return (
            RelationshipType.RELATED,
            0.5,
            explanation,
            _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
            True,
        )
    if comparable and grounded:
        explanation = _explain(
            "related", a, b, verdict, ctx, strong, overlap, False
        )
        return (
            RelationshipType.RELATED,
            0.5,
            explanation,
            _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
            False,
        )
    explanation = _explain(
        "unrelated", a, b, verdict, ctx, strong, overlap, False
    )
    return (
        RelationshipType.UNRELATED,
        0.9,
        explanation,
        _metadata(a, b, verdict, incompatible, unknown, strong, overlap),
        False,
    )


def _metadata(
    a: Fact,
    b: Fact,
    verdict: NumericVerdict,
    incompatible: list[str],
    unknown: list[str],
    strong: bool,
    overlap: bool,
) -> dict:
    """Build the reasoning metadata dict (always with the required keys)."""
    if verdict == NumericVerdict.INCOMPARABLE:
        relative_difference: float | None = None
    elif a.normalized_number == 0.0 and b.normalized_number == 0.0:
        relative_difference = 0.0
    else:
        relative_difference = _relative_difference(
            float(a.normalized_number or 0.0), float(b.normalized_number or 0.0)
        )
    return {
        "numeric_verdict": verdict.value,
        "incompatible_dimensions": list(incompatible),
        "unknown_dimensions": list(unknown),
        "compatible_dimensions": sorted(
            name for name in _DIMENSION_ORDER if name not in incompatible and name not in unknown
        ),
        "strong_match": strong,
        "semantic_overlap": overlap,
        "relative_difference": relative_difference,
    }
