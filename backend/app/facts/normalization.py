"""N-NORM: deterministic normalization engine (no LLM, no network, stdlib only).

``normalize_fact`` populates ONLY the five reserved Fact columns:

- ``normalized_number`` / ``normalized_unit`` (canonical scale + unit),
- ``canonical_subject`` / ``canonical_predicate`` (generic name cleanup),
- ``ambiguity_flags`` (append-only ``norm:`` flags; never invented values).

Raw fields, ``evidence_ids``, ``extraction_confidence`` and ``status`` are
NEVER touched. Unknown units/scales yield ``None`` plus a ``norm:`` flag.
No cross-currency conversion is ever performed.

Canonical unit vocabulary (closed set emitted by this module):

- ``"INR crore"`` — INR values (``Rs``/``Re``/``INR``/``Rs.``/``INR``,
  or Indian scale words ``cr``/``crore``/``lakh``/``lac`` with a
  scale-only unit), canonical scale 1 crore = 1e7.
- ``"USD million"`` / ``"EUR million"`` / ``"GBP million"`` — ``$``/``USD``,
  ``EUR``/``EUR``, ``GBP``/``GBP``; canonical scale 1 million = 1e6.
- ``"million tonnes"`` — ``tonne``/``tonnes`` (value / 1e6),
  ``kg``-family (value / 1e9), ``g``-family (value / 1e12).
- ``"fraction"`` — plain percentages (value / 100) and basis points
  (value * 0.0001).
- ``"percentage_point"`` — ``pp`` / ``percentage point(s)``. NOTE: this is
  deliberately NOT a fraction — a "6.5 pp" move stays ``6.5``, it is NOT
  divided by 100. Percentage points are additive deltas, not ratios.
- cleaned count units — descriptive remainder after stripping standalone
  magnitude tokens (e.g. ``"Mn express parcels"`` -> ``"express parcels"``)
  with the number passed through unchanged (suffix scaling was already
  applied by the validator).
- ``None`` — clean skips (TEXT kind / missing number), scale-only counts
  (e.g. bare ``"Mn"``), and unknown/ambiguous units (with a ``norm:`` flag).

Number parsing (``parse_number``) and time parsing (``parse_time``) are owned
by :mod:`app.facts.validator` and reused here by import only — this module
never re-parses numeric text; it works from ``Fact.value_number``.
"""

from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import repositories
from app.facts.validator import parse_number, parse_time  # noqa: F401 -- reused by contract: validator owns parsing; normalization never re-parses.
from app.models.fact import Fact, ValueKind

# ---------------------------------------------------------------------------
# Canonical units + divisors (generic lookup tables — no dataset-specific logic)
# ---------------------------------------------------------------------------

INR_CANONICAL_UNIT = "INR crore"
USD_CANONICAL_UNIT = "USD million"
EUR_CANONICAL_UNIT = "EUR million"
GBP_CANONICAL_UNIT = "GBP million"
MASS_CANONICAL_UNIT = "million tonnes"
FRACTION_UNIT = "fraction"
PERCENTAGE_POINT_UNIT = "percentage_point"

#: currency code -> (canonical unit, divisor). NO cross-currency conversion.
CURRENCY_CANONICAL: dict[str, tuple[str, float]] = {
    "INR": (INR_CANONICAL_UNIT, 1e7),
    "USD": (USD_CANONICAL_UNIT, 1e6),
    "EUR": (EUR_CANONICAL_UNIT, 1e6),
    "GBP": (GBP_CANONICAL_UNIT, 1e6),
}

#: whole-token (lowercased, dot-stripped) -> currency code.
CURRENCY_TOKENS: dict[str, str] = {
    "₹": "INR",
    "rs": "INR",
    "re": "INR",
    "inr": "INR",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

#: Indian scale words imply INR (canonical scale crore) when no explicit
#: currency is present and the unit carries no other descriptive content
#: (e.g. unit "Cr" / value "8142 Cr" -> INR crore; "lakh people" stays a
#: count of people).
INDIAN_SCALE_TOKENS = frozenset({"cr", "crore", "lakh", "lac"})

#: whole-token mass word -> divisor yielding "million tonnes".
MASS_DIVISORS: dict[str, float] = {
    "tonne": 1e6,
    "tonnes": 1e6,
    "kg": 1e9,
    "kgs": 1e9,
    "kilogram": 1e9,
    "kilograms": 1e9,
    "kilogramme": 1e9,
    "kilogrammes": 1e9,
    "g": 1e12,
    "gram": 1e12,
    "grams": 1e12,
    "gramme": 1e12,
    "grammes": 1e12,
}

#: bare mass tokens too ambiguous to normalize (metric ton? short ton?).
AMBIGUOUS_MASS_TOKENS = frozenset({"t", "ton", "tons"})

#: standalone magnitude tokens stripped from count units (case-insensitive,
#: token-boundary match; validator already applied suffix scaling).
MAGNITUDE_TOKENS = frozenset(
    {
        "k",
        "m",
        "mn",
        "cr",
        "lakh",
        "lac",
        "crore",
        "b",
        "bn",
        "t",
        "thousand",
        "million",
        "billion",
        "trillion",
    }
)

#: currency-like tokens with no canonical scale -> norm:currency_unknown.
UNKNOWN_CURRENCY_TOKENS = frozenset(
    {
        "¥",
        "₩",
        "₽",
        "¢",
        "₺",
        "jpy",
        "yen",
        "cny",
        "yuan",
        "rmb",
        "krw",
        "won",
        "rub",
        "ruble",
        "rouble",
        "chf",
        "aud",
        "cad",
        "hkd",
        "sgd",
        "nzd",
        "mxn",
        "peso",
        "pesos",
        "brl",
        "real",
        "reais",
        "zar",
        "rand",
        "aed",
        "dirham",
        "sar",
        "riyal",
        "riyals",
        "sek",
        "nok",
        "dkk",
        "kr",
        "krone",
        "kroner",
        "krona",
        "kronor",
        "pln",
        "zloty",
        "thb",
        "baht",
        "myr",
        "ringgit",
        "idr",
        "rupiah",
        "php",
        "pkr",
        "bdt",
        "taka",
        "lkr",
        "npr",
        "kes",
        "ngn",
        "naira",
        "egp",
        "try",
        "lira",
        "liras",
        "ils",
        "shekel",
        "shekels",
        "qar",
        "kwd",
        "bhd",
        "omr",
        "jod",
        "twd",
        "ars",
        "clp",
        "cop",
        "pen",
        "uah",
        "ron",
        "huf",
        "forint",
        "czk",
        "koruna",
        "vnd",
        "dong",
        "dollar",
        "dollars",
        "buck",
        "bucks",
        "cent",
        "cents",
        "pence",
        "penny",
        "pennies",
        "paise",
        "paisa",
        "rupee",
        "rupees",
        "pound",
        "pounds",
        "sterling",
        "euro",
        "euros",
        "franc",
        "francs",
    }
)

#: measurement words with no defined conversion -> norm:unit_unknown.
#: (Time words like years/days are deliberately excluded: durations are
#: treated as plain counts. Single letters colliding with magnitude tokens
#: — "m", "k", "b" — are excluded: the magnitude reading wins.)
UNKNOWN_UNIT_TOKENS = frozenset(
    {
        # mass-like
        "lb",
        "lbs",
        "oz",
        "ounce",
        "ounces",
        "quintal",
        "quintals",
        "mt",
        "kt",
        "megaton",
        "megatons",
        "megatonne",
        "megatonnes",
        "kiloton",
        "kilotons",
        "kilotonne",
        "kilotonnes",
        "gt",
        "gigatonne",
        "gigatonnes",
        "stone",
        "stones",
        "grain",
        "grains",
        "carat",
        "carats",
        # length
        "furlong",
        "furlongs",
        "mile",
        "miles",
        "km",
        "kilometre",
        "kilometres",
        "kilometer",
        "kilometers",
        "cm",
        "centimetre",
        "centimetres",
        "centimeter",
        "centimeters",
        "mm",
        "millimetre",
        "millimetres",
        "millimeter",
        "millimeters",
        "metre",
        "metres",
        "meter",
        "meters",
        "ft",
        "feet",
        "foot",
        "inch",
        "inches",
        "yard",
        "yards",
        "nm",
        # volume
        "litre",
        "litres",
        "liter",
        "liters",
        "gallon",
        "gallons",
        "ml",
        "cl",
        "dl",
        "barrel",
        "barrels",
        "pint",
        "pints",
        "quart",
        "quarts",
        # area
        "acre",
        "acres",
        "hectare",
        "hectares",
        "sqm",
        "sqft",
        # energy / power
        "joule",
        "joules",
        "watt",
        "watts",
        "kw",
        "kwh",
        "mw",
        "gw",
        "volt",
        "volts",
        "amp",
        "amps",
        # temperature / angle
        "celsius",
        "fahrenheit",
        "kelvin",
        "degree",
        "degrees",
        # data
        "byte",
        "bytes",
        "bit",
        "bits",
        "kb",
        "mb",
        "gb",
        "tb",
        "pb",
        # time-subunit technical
        "ms",
        "msec",
        "millisecond",
        "milliseconds",
    }
)

#: trailing corporate suffixes stripped from canonical names.
CORPORATE_SUFFIXES = frozenset(
    {
        "ltd",
        "limited",
        "inc",
        "corp",
        "corporation",
        "llc",
        "plc",
        "pvt",
        "private",
    }
)

_TOKEN_RE = re.compile(r"[₹$€£¥₩₽¢₺]|\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_PP_RE = re.compile(r"(?<![a-z0-9])(?:percentage points?|pp)(?![a-z0-9])")
_BP_RE = re.compile(r"(?<![a-z0-9])(?:basis points?|bps?)(?![a-z0-9])")
_MAGNITUDE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(sorted(MAGNITUDE_TOKENS, key=len, reverse=True))
    + r")(?:s)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _tokens(text: str | None) -> list[str]:
    """Lowercase whole-word tokens; currency symbols split off separately."""
    if not text:
        return []
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip(".")
        if token:
            out.append(token)
    return out


def _detect_currency(fact: Fact) -> str | None:
    """Detect a known currency code from unit string, then value_text.

    Whole-token match only (so ``re`` never fires inside ``express``).
    Returns ``"INR"`` / ``"USD"`` / ``"EUR"`` / ``"GBP"`` or ``None``.
    """
    for text in (fact.unit, fact.value_text):
        for token in _tokens(text):
            code = CURRENCY_TOKENS.get(token)
            if code is not None:
                return code
    return None


def _unknown_currency_token(fact: Fact) -> str | None:
    """First currency-like token with no canonical scale, if any."""
    for text in (fact.unit, fact.value_text):
        for token in _tokens(text):
            if token in UNKNOWN_CURRENCY_TOKENS:
                return token
    return None


def _mass_divisor(unit: str | None) -> float | None:
    """Divisor yielding "million tonnes" for a known mass unit, else None."""
    for token in _tokens(unit):
        divisor = MASS_DIVISORS.get(token)
        if divisor is not None:
            return divisor
    return None


def _has_ambiguous_mass(unit: str | None) -> bool:
    """True when the unit carries bare t/ton/tons (tonne? short ton?)."""
    return any(token in AMBIGUOUS_MASS_TOKENS for token in _tokens(unit))


def _unknown_unit_token(unit: str | None) -> str | None:
    """First measurement token with no defined conversion, if any."""
    for token in _tokens(unit):
        if token in UNKNOWN_UNIT_TOKENS:
            return token
    return None


def _has_indian_scale(fact: Fact) -> bool:
    """True when cr/crore/lakh/lac appears in unit or value_text."""
    for text in (fact.unit, fact.value_text):
        for token in _tokens(text):
            if token in INDIAN_SCALE_TOKENS:
                return True
    return False


def _strip_magnitudes(unit: str | None) -> str | None:
    """Remove standalone magnitude tokens; None when nothing remains."""
    if not unit or not unit.strip():
        return None
    cleaned = _MAGNITUDE_RE.sub(" ", unit)
    cleaned = _WS_RE.sub(" ", cleaned).strip(" \t,;:.").strip()
    return cleaned or None


def _normalize_percentage(fact: Fact) -> tuple[float | None, str | None]:
    """Normalize a PERCENTAGE-kind value (value_number already parsed)."""
    value = fact.value_number
    combined = f"{fact.unit or ''} {fact.value_text or ''}".lower()
    if _PP_RE.search(combined):
        # Percentage points are additive deltas, NOT fractions: no /100.
        return value, PERCENTAGE_POINT_UNIT
    if _BP_RE.search(combined):
        return value * 0.0001, FRACTION_UNIT
    return value / 100.0, FRACTION_UNIT


def _normalize_numeric(
    fact: Fact,
) -> tuple[float | None, str | None, str | None]:
    """Normalize a NUMERIC-kind value; third element is a norm: flag or None."""
    value = fact.value_number
    assert value is not None  # caller guarantees a parsed number

    currency = _detect_currency(fact)
    if currency is not None:
        canonical_unit, divisor = CURRENCY_CANONICAL[currency]
        return value / divisor, canonical_unit, None

    divisor = _mass_divisor(fact.unit)
    if divisor is not None:
        return value / divisor, MASS_CANONICAL_UNIT, None

    unknown_currency = _unknown_currency_token(fact)
    if unknown_currency is not None:
        return None, None, f"norm:currency_unknown:{unknown_currency}"

    if _has_ambiguous_mass(fact.unit):
        return None, None, "norm:unit_ambiguous:t"

    unknown_unit = _unknown_unit_token(fact.unit)
    if unknown_unit is not None:
        return None, None, f"norm:unit_unknown:{unknown_unit}"

    if _has_indian_scale(fact) and _strip_magnitudes(fact.unit) is None:
        # Scale-only unit (e.g. "Cr", or no unit with value "8142 Cr"):
        # Indian scales imply the INR canonical scale.
        canonical_unit, inr_divisor = CURRENCY_CANONICAL["INR"]
        return value / inr_divisor, canonical_unit, None

    return value, _strip_magnitudes(fact.unit), None


def _canonical_name(text: str | None) -> str | None:
    """Lowercase + punctuation/whitespace squash, strip leading "the" and
    trailing corporate suffixes (repeatable: "Pvt Ltd" -> "pvt" -> "")."""
    if not text or not text.strip():
        return None
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return None
    previous = None
    while cleaned != previous:
        previous = cleaned
        if cleaned == "the":
            cleaned = ""
        elif cleaned.startswith("the "):
            cleaned = cleaned[4:].strip()
        parts = cleaned.split(" ") if cleaned else []
        if parts and parts[-1] in CORPORATE_SUFFIXES:
            cleaned = " ".join(parts[:-1]).strip()
    return cleaned or None


def normalize_fact(fact: Fact) -> Fact:
    """Return a copy of ``fact`` with ONLY the five reserved fields changed.

    The input object is never mutated (all lists are rebuilt). TEXT-kind
    facts and facts with ``value_number is None`` keep ``None`` normalized
    fields with no flags (clean skip, not ambiguity). Time, scope,
    estimate, geography and context pass through byte-identical.
    """
    normalized_number: float | None = None
    normalized_unit: str | None = None
    new_flags: list[str] = []

    if fact.value_kind != ValueKind.TEXT and fact.value_number is not None:
        if fact.value_kind == ValueKind.PERCENTAGE:
            normalized_number, normalized_unit = _normalize_percentage(fact)
        else:
            normalized_number, normalized_unit, flag = _normalize_numeric(fact)
            if flag is not None:
                new_flags.append(flag)

    updated = fact.model_copy(deep=True)
    updated.normalized_number = normalized_number
    updated.normalized_unit = normalized_unit
    updated.canonical_subject = _canonical_name(fact.subject)
    updated.canonical_predicate = _canonical_name(fact.predicate)
    updated.ambiguity_flags = list(fact.ambiguity_flags or []) + new_flags
    return updated


class NormalizationReport(BaseModel):
    """Per-document normalization outcome."""

    document_id: str
    facts_processed: int = 0
    facts_normalized: int = 0  # normalized_number set OR canonical name set
    facts_flagged: int = 0  # any ambiguity flag starting with "norm:"
    errors: list[str] = []


def normalize_document_facts(session: Session, document_id: str) -> NormalizationReport:
    """Normalize every fact of a document; never raise for bad data.

    Lists facts via ``repositories.list_facts_for_document``, normalizes
    each with :func:`normalize_fact`, and persists the reserved columns via
    ``repositories.update_fact_normalization(session, updated)``. Per-fact
    failures are recorded as error strings and the original row is left
    untouched.
    """
    report = NormalizationReport(document_id=str(document_id))
    try:
        doc_uuid = document_id if isinstance(document_id, UUID) else UUID(str(document_id))
    except (ValueError, AttributeError, TypeError) as exc:
        report.errors.append(f"invalid document_id {document_id!r}: {exc}")
        return report

    try:
        facts = repositories.list_facts_for_document(session, doc_uuid)
    except Exception as exc:  # defensive: bad data must not raise
        report.errors.append(f"list_facts_for_document failed: {type(exc).__name__}: {exc}")
        return report

    for fact in facts:
        report.facts_processed += 1
        try:
            updated = normalize_fact(fact)
            repositories.update_fact_normalization(session, updated)
        except Exception as exc:  # per-fact containment; row untouched
            fid = getattr(fact, "id", "?")
            report.errors.append(f"fact {fid}: {type(exc).__name__}: {exc}")
            continue
        if (
            updated.normalized_number is not None
            or updated.canonical_subject is not None
            or updated.canonical_predicate is not None
        ):
            report.facts_normalized += 1
        if any(
            flag.startswith("norm:") for flag in (updated.ambiguity_flags or [])
        ):
            report.facts_flagged += 1
    return report
