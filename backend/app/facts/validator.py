"""Deterministic draft-to-fact validation (no LLM calls here).

Responsibilities:
- evidence-ID enforcement (unknown IDs → rejection, never stored);
- value-kind resolution and numeric parsing (deterministic only);
- time-expression parsing where unambiguous (never assume jurisdictions);
- confidence passthrough and ambiguity bookkeeping.

Anything semantic (entity/predicate canonicalization, unit conversion)
belongs to the later normalization layer.
"""

import calendar
import re
from datetime import date
from uuid import UUID

from app.models.fact import (
    EstimateStatus,
    EvidenceChunk,
    Fact,
    FactDraft,
    FactStatus,
    TimeKind,
    ValueKind,
)

# Longest-first matching matters ("mn" before "m", "crore" before "cr").
_NUMBER_SUFFIXES: tuple[tuple[str, float], ...] = (
    ("trillion", 1e12),
    ("billion", 1e9),
    ("million", 1e6),
    ("thousand", 1e3),
    ("crore", 1e7),
    ("lakh", 1e5),
    ("lac", 1e5),
    ("mn", 1e6),
    ("cr", 1e7),
    ("bn", 1e9),
    ("b", 1e9),
    ("m", 1e6),
    ("k", 1e3),
    ("t", 1e12),
)

_CURRENCY_PREFIXES = ("₹", "rs.", "rs", "$", "€", "£", "inr", "usd")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_FY_RE = re.compile(r"^fy\s?(\d{2}|\d{4})(?:\s*[-–/]\s*(\d{2}|\d{4}))?$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^q[1-4](\s+fy\s?\d{2,4})?$", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DMY_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$")
_MONTH_YEAR_RE = re.compile(r"^([A-Za-z]+)\s+(\d{4})$")
_RANGE_RE = re.compile(r"^(\d{4})\s*[-–/]\s*(\d{2}|\d{4})$")


def parse_number(text: str) -> float | None:
    """Parse Indian/international numeric strings deterministically.

    Handles currency prefixes, thousand separators, accounting
    parenthetical negatives, and magnitude suffixes (K/Mn/Cr/lakh/B/T).
    Returns None when the text is not deterministically numeric —
    the caller flags ambiguity rather than inventing a value.
    """
    if not isinstance(text, str):
        return None
    s = text.strip().lower().replace(",", "").replace(" ", "")
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    for prefix in _CURRENCY_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.lstrip("+-")
    multiplier = 1.0
    for suffix, factor in _NUMBER_SUFFIXES:
        if s.endswith(suffix):
            core = s[: -len(suffix)]
            if core and any(ch.isdigit() for ch in core):
                multiplier = factor
                s = core
                break
    if s.endswith("%"):
        s = s[:-1]
    try:
        value = float(s)
    except ValueError:
        return None
    if negative:
        value = -value
    return value * multiplier


def parse_time(text: str | None) -> tuple[TimeKind, date | None, date | None]:
    """Parse time expressions only where unambiguous.

    Fiscal years and quarters yield a kind with NO dates: period
    boundaries depend on jurisdiction/convention the extractor must not
    assume. Fully specified dates and month-precision labels yield real
    dates. Anything else is UNKNOWN with the verbatim text preserved.
    """
    if not text or not text.strip():
        return TimeKind.UNKNOWN, None, None
    s = text.strip()
    if _FY_RE.match(s):
        return TimeKind.FISCAL_YEAR, None, None
    if _QUARTER_RE.match(s):
        return TimeKind.QUARTER, None, None
    m = _ISO_DATE_RE.match(s)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return TimeKind.DATE, d, d
        except ValueError:
            return TimeKind.UNKNOWN, None, None
    m = _DMY_RE.match(s)
    if m and m.group(2).lower() in _MONTHS:
        try:
            d = date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
            return TimeKind.DATE, d, d
        except ValueError:
            return TimeKind.UNKNOWN, None, None
    m = _MONTH_YEAR_RE.match(s)
    if m and m.group(1).lower() in _MONTHS:
        year, month = int(m.group(2)), _MONTHS[m.group(1).lower()]
        last = calendar.monthrange(year, month)[1]
        return TimeKind.DATE, date(year, month, 1), date(year, month, last)
    if _RANGE_RE.match(s):
        return TimeKind.RANGE, None, None
    return TimeKind.UNKNOWN, None, None


def resolve_kind(draft: FactDraft) -> tuple[ValueKind, list[str]]:
    """Resolve the value kind deterministically with correction flags."""
    flags: list[str] = []
    text = draft.value_text.strip()
    if text.endswith("%"):
        if draft.value_kind is not None and draft.value_kind != ValueKind.PERCENTAGE:
            flags.append(f"kind_corrected:{draft.value_kind.value}->PERCENTAGE")
        return ValueKind.PERCENTAGE, flags
    if parse_number(text) is not None:
        if draft.value_kind is not None and draft.value_kind not in (
            ValueKind.NUMERIC, ValueKind.PERCENTAGE,
        ):
            flags.append(f"kind_corrected:{draft.value_kind.value}->NUMERIC")
        return ValueKind.NUMERIC, flags
    if draft.value_kind in (ValueKind.NUMERIC, ValueKind.PERCENTAGE):
        flags.append("value_unparseable")
        return draft.value_kind, flags
    return ValueKind.TEXT, flags


def build_fact(
    draft: FactDraft, document_id: UUID, chunk: EvidenceChunk
) -> tuple[Fact | None, str | None]:
    """Validate one draft into a Fact, or reject it with a reason.

    Rejection (return None) happens ONLY for provenance violations:
    unknown evidence IDs. Everything else is preserved with ambiguity
    flags — validation never invents values.
    """
    known = {u.evidence_id for u in chunk.units}
    unknown = [str(i) for i in draft.evidence_ids if i not in known]
    if unknown:
        return None, f"unknown evidence IDs: {', '.join(unknown)}"

    kind, flags = resolve_kind(draft)
    number = parse_number(draft.value_text) if kind != ValueKind.TEXT else None
    if kind != ValueKind.TEXT and number is None:
        flags.append("value_unparseable")
    time_kind, time_start, time_end = parse_time(draft.time_text)
    if draft.time_text and draft.time_text.strip() and time_kind == TimeKind.UNKNOWN:
        flags.append("time_unparsed")

    unit = draft.unit.strip() if draft.unit and draft.unit.strip() else None
    estimate = draft.estimate_status or EstimateStatus.UNKNOWN
    geography = (
        draft.geography.strip()
        if draft.geography and draft.geography.strip()
        else None
    )
    scope = (
        draft.scope_text.strip()
        if draft.scope_text and draft.scope_text.strip()
        else None
    )
    if draft.ambiguous:
        flags.append("llm_flagged_ambiguous")

    return Fact(
        document_id=document_id,
        subject=draft.subject.strip(),
        predicate=draft.predicate.strip(),
        value_kind=kind,
        value_text=draft.value_text.strip(),
        value_number=number,
        unit=unit,
        time_text=draft.time_text.strip() if draft.time_text else None,
        time_kind=time_kind,
        time_start=time_start,
        time_end=time_end,
        scope_text=scope,
        estimate_status=estimate,
        geography=geography,
        context=dict(draft.context),
        evidence_ids=list(draft.evidence_ids),
        extraction_confidence=draft.confidence,
        ambiguity_flags=flags,
        status=FactStatus.AMBIGUOUS if flags else FactStatus.CONFIRMED,
    ), None
