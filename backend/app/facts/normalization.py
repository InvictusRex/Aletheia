"""N-NORM: deterministic normalization engine (no LLM, no network, stdlib only).

``normalize_fact`` populates ONLY the five reserved Fact columns:

- ``normalized_number`` / ``normalized_unit`` (canonical scale + unit),
- ``canonical_subject`` / ``canonical_predicate`` (generic name cleanup),
- ``ambiguity_flags`` (``norm:`` flags recomputed from scratch; never
  invented values).

Idempotent by construction: each run REPLACES the ``norm:`` flags it owns
rather than appending to them, so normalizing an already-normalized fact
is a no-op and a changed rule cannot leave a stale flag behind. Flags
owned by the validator (no ``norm:`` prefix) are preserved verbatim.

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
  magnitude tokens (e.g. ``"Mn express parcels"`` -> ``"express parcels"``).
  The magnitude is applied to the number here ONLY when the value text did
  not already carry one: an extractor may write the scale either in the
  value (``"740 Mn"``, already scaled by the validator) or in the unit
  (value ``"740"``, unit ``"Mn express parcels"``), and only the second
  form still needs scaling.
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
from app.facts.canonicalization import apply_aliases, build_alias_map
from app.facts.validator import parse_number, parse_time  # noqa: F401 -- reused by contract: validator owns parsing; normalization never re-parses.
from app.models.fact import Comparability, Fact, ValueKind

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

#: standalone magnitude token -> its multiplier. Applied ONLY when the
#: magnitude lives in the unit and not in the value text: an extractor may
#: write either "US$ 14 billion" (validator already scaled it) or value
#: "14.0" with unit "US$ billion" (nothing scaled it yet). Without this the
#: second form is off by the whole magnitude.
MAGNITUDE_SCALE: dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "lakh": 1e5,
    "lac": 1e5,
    "cr": 1e7,
    "crore": 1e7,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

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
#: A unit that means "this number is a percentage" and nothing else.
#: Anchored: "% of gdp" is a share of a named base, not a bare percent,
#: and must keep its descriptive unit rather than silently becoming a
#: dimensionless fraction.
_PERCENT_UNIT_RE = re.compile(r"%|per ?cent(?:age)?|percent(?:age)?")
_MAGNITUDE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(sorted(MAGNITUDE_TOKENS, key=len, reverse=True))
    + r")(?:s)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)

#: Same tokens, but a digit may sit immediately before them, because the
#: validator scales a suffix written flush against the number ("1,429K").
#: Used only to detect whether the VALUE already carries its magnitude;
#: stripping still uses the stricter whole-token form above.
_VALUE_MAGNITUDE_RE = re.compile(
    r"(?<![A-Za-z])(?:"
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


def _unit_scale(fact: Fact) -> float:
    """Multiplier for a magnitude carried by the unit rather than the value.

    Returns 1.0 whenever the value text already carries a magnitude token,
    because the validator scaled it there and applying it twice would be a
    million-fold error in the other direction.
    """
    if _VALUE_MAGNITUDE_RE.search(fact.value_text or ""):
        return 1.0
    for token in _tokens(fact.unit):
        scale = MAGNITUDE_SCALE.get(token)
        if scale is not None:
            return scale
    return 1.0


def _normalize_numeric(
    fact: Fact,
) -> tuple[float | None, str | None, str | None]:
    """Normalize a NUMERIC-kind value; third element is a norm: flag or None."""
    value = fact.value_number
    assert value is not None  # caller guarantees a parsed number
    value *= _unit_scale(fact)

    # A percent marker may live in the unit rather than the value text
    # ("1.7" + unit "%"), in which case resolve_kind saw no "%" and left
    # the fact NUMERIC. Canonicalize it here so it compares against a
    # PERCENTAGE-kind fact stating the same thing.
    unit_text = (fact.unit or "").lower()
    if unit_text:
        if _PP_RE.search(unit_text):
            return value, PERCENTAGE_POINT_UNIT, None
        if _BP_RE.search(unit_text):
            return value * 0.0001, FRACTION_UNIT, None
        if _PERCENT_UNIT_RE.fullmatch(unit_text.strip()):
            return value / 100.0, FRACTION_UNIT, None

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


NORM_FLAG_PREFIX = "norm:"


def _replace_norm_flags(
    existing: list[str] | None, fresh: list[str]
) -> list[str]:
    """Drop the flags this module owns, then append the freshly computed ones.

    Keeps validator-owned flags in their original order and positions
    ``norm:`` flags last. Re-running normalization therefore converges
    instead of growing the list on every pass.
    """
    preserved = [
        flag
        for flag in (existing or [])
        if not str(flag).startswith(NORM_FLAG_PREFIX)
    ]
    return preserved + fresh


def normalize_fact(fact: Fact, aliases: dict[str, str] | None = None) -> Fact:
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
    # Syntactic cleanup first, then corpus-mined abbreviation expansion,
    # so "GFCF" and "Gross Fixed Capital Formation" converge on one
    # string and the matching layer can see them as the same claim.
    resolved = aliases or {}
    updated.canonical_subject = apply_aliases(_canonical_name(fact.subject), resolved)
    updated.canonical_predicate = apply_aliases(
        _canonical_name(fact.predicate), resolved
    )
    updated.ambiguity_flags = _replace_norm_flags(fact.ambiguity_flags, new_flags)
    return updated


class NormalizationReport(BaseModel):
    """Per-document normalization outcome."""

    document_id: str
    facts_processed: int = 0
    facts_normalized: int = 0  # normalized_number set OR canonical name set
    facts_flagged: int = 0  # any ambiguity flag starting with "norm:"
    aliases_applied: int = 0  # abbreviation definitions mined from the corpus
    comparable: int = 0  # number + canonical unit + typed period
    partially_comparable: int = 0  # missing one of the three
    not_comparable: int = 0  # cannot take part in numeric reasoning
    missing_counts: dict[str, int] = {}  # what the gaps actually are
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

    try:
        aliases = build_alias_map(repositories.list_evidence_texts(session))
    except Exception as exc:  # alias mining must never fail normalization
        report.errors.append(f"alias mining failed: {type(exc).__name__}: {exc}")
        aliases = {}
    report.aliases_applied = len(aliases)

    for fact in facts:
        report.facts_processed += 1
        try:
            updated = normalize_fact(fact, aliases)
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
        verdict = updated.comparability
        if verdict == Comparability.COMPARABLE:
            report.comparable += 1
        elif verdict == Comparability.PARTIAL:
            report.partially_comparable += 1
        else:
            report.not_comparable += 1
        for gap in updated.missing_for_comparison:
            report.missing_counts[gap] = report.missing_counts.get(gap, 0) + 1
    return report
