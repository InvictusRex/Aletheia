"""Corpus-derived alias canonicalization (deterministic, no LLM, no network).

The problem this solves: two documents state the same claim in different
words -- "GFCF" in one and "Gross Fixed Capital Formation" in the other --
so ``canonical_subject`` / ``canonical_predicate`` never match and neither
corroboration nor contradiction is reachable between them.

The aliases are MINED FROM EACH DOCUMENT'S OWN TEXT rather than read from
a fixed vocabulary, because a table of terms built from one corpus cannot
generalize to a PDF the system has never seen. Technical documents define
their own abbreviations in a standard way::

    Gross Fixed Capital Formation (GFCF)
    Consumer Price Index ("CPI")

so the corpus carries its own glossary. Mining it keeps the layer
document-agnostic: a new upload contributes its own definitions and needs
no code change.

Acceptance rule. A candidate is kept only when the abbreviation's letters
align with the initials of exactly as many trailing words. Anchoring at
the END is what discards the surrounding prose that a naive match drags
in ("Although real gross domestic product (GDP)" yields "gross domestic
product", not the sentence opener).

What this deliberately does NOT do: invent equivalences. Every alias is
a definition the document itself states, so the mapping is traceable to
source text rather than to a model's opinion or a curated list.
"""

from __future__ import annotations

import re
from collections import Counter

__all__ = [
    "MIN_ABBREVIATION_LENGTH",
    "SELF_REFERENCE_NOUNS",
    "apply_aliases",
    "build_alias_map",
    "build_self_reference_map",
    "mine_alias_pairs",
    "mine_self_reference",
]

#: Single letters are far too ambiguous to treat as a definition.
MIN_ABBREVIATION_LENGTH = 2

#: Longest plausible spelled-out form, in words.
_MAX_LONG_FORM_WORDS = 8

#: "Long Form (ABBR)" with optional quoting around the abbreviation.
_DEFINITION_RE = re.compile(
    r"([A-Za-z][A-Za-z&.\-]*(?:\s+[A-Za-z&.\-]+){0,%d})"
    r"\s*\(\s*[“‘\"']?([A-Z][A-Za-z&.\-]{1,9})[”’\"']?\s*\)"
    % _MAX_LONG_FORM_WORDS
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _initials(words: list[str]) -> str:
    return "".join(word[0] for word in words if word[:1].isalpha()).upper()


def _aligned_long_form(long_form: str, abbreviation: str) -> str | None:
    """Trailing words whose initials spell the abbreviation, else None.

    Anchoring at the end is what separates the definition from the prose
    that happens to precede it.
    """
    letters = [c for c in abbreviation.upper() if c.isalpha()]
    words = long_form.split()
    if len(letters) < MIN_ABBREVIATION_LENGTH or len(words) < len(letters):
        return None
    candidate = words[-len(letters) :]
    if _initials(candidate) != "".join(letters):
        return None
    return " ".join(candidate)


def mine_alias_pairs(texts: list[str]) -> list[tuple[str, str]]:
    """Return ``(abbreviation, long form)`` definitions stated in ``texts``.

    Pure function. Output is deduplicated and ordered by how often the
    definition appears, then alphabetically, so the result is stable for
    a given corpus regardless of input ordering.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for text in texts:
        if not text or "(" not in text:
            continue
        for long_form, abbreviation in _DEFINITION_RE.findall(text.replace("\n", " ")):
            aligned = _aligned_long_form(long_form, abbreviation)
            if aligned is None:
                continue
            counts[(_clean(abbreviation), _clean(aligned))] += 1
    return [pair for pair, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def build_alias_map(texts: list[str]) -> dict[str, str]:
    """Map each abbreviation to the single long form the corpus favours.

    A term defined inconsistently keeps its most frequent reading; ties
    break alphabetically so the map is deterministic. An abbreviation
    that also appears as somebody else's long form is dropped, because
    resolving it either way would be a guess.
    """
    pairs = mine_alias_pairs(texts)
    chosen: dict[str, str] = {}
    for abbreviation, long_form in pairs:
        chosen.setdefault(abbreviation, long_form)
    long_forms = set(chosen.values())
    return {
        abbreviation: long_form
        for abbreviation, long_form in chosen.items()
        if abbreviation not in long_forms
    }


def _collapse_adjacent_repeats(tokens: list[str]) -> list[str]:
    """Drop an n-gram that immediately repeats the one before it.

    A source often writes both spellings together ("Gross Fixed Capital
    Formation (GFCF)"), so expanding the abbreviation in place would
    state the same phrase twice. Longest runs collapse first so the
    whole phrase goes rather than one shared word.
    """
    result = list(tokens)
    changed = True
    while changed:
        changed = False
        for size in range(len(result) // 2, 0, -1):
            for start in range(len(result) - 2 * size + 1):
                if result[start : start + size] == result[start + size : start + 2 * size]:
                    del result[start + size : start + 2 * size]
                    changed = True
                    break
            if changed:
                break
    return result


def apply_aliases(name: str | None, aliases: dict[str, str]) -> str | None:
    """Rewrite abbreviations inside an already-cleaned canonical name.

    Whole-token replacement only, so "cpi" expands while "cpio" does not.
    A name that is exactly a long form is returned unchanged, which is
    what makes the two spellings converge on one string. Returns ``None``
    for empty input so callers can pass a missing name straight through.
    """
    if not name or not name.strip():
        return None
    if not aliases:
        return name
    tokens = _WORD_RE.findall(name.lower())
    if not tokens:
        return name
    rewritten: list[str] = []
    for token in tokens:
        rewritten.extend(aliases.get(token, token).split())
    collapsed = _collapse_adjacent_repeats(rewritten)
    return _WS_RE.sub(" ", " ".join(collapsed)).strip() or name


#: Generic nouns an organisation uses for itself. These are ordinary
#: English reference words, not names from any particular corpus, so a
#: filing by any issuer uses the same handful.
SELF_REFERENCE_NOUNS = frozenset(
    {
        "company",
        "group",
        "bank",
        "corporation",
        "issuer",
        "fund",
        "trust",
        "partnership",
        "firm",
        "society",
        "authority",
        "parent",
    }
)

#: Pronouns and possessives a filing uses for itself.
_SELF_PRONOUNS = frozenset({"we", "us", "our", "ours"})

#: ``Some Entity Name ("Company")`` / ``Some Entity Name (the "Company")``.
_SELF_DEFINITION_RE = re.compile(
    r"([A-Z][A-Za-z&.,\-]*(?:\s+[A-Za-z&.,\-]+){0,6})"
    r"\s*\(\s*(?:the\s+)?[“‘\"']?"
    r"(%s)"
    r"[”’\"']?\s*(?:/|\))" % "|".join(sorted(SELF_REFERENCE_NOUNS))
    ,
    re.IGNORECASE,
)


def _trailing_proper_name(name: str) -> str | None:
    """The trailing run of capitalised words, which is the entity itself.

    A definition is usually embedded in prose ("approved by the
    shareholders of Delhivery Limited (the “Company”)"), and only the
    tail of it names the organisation. Anchoring on the trailing
    capitalised run drops the verb phrase without needing to know any
    particular company or verb.
    """
    words = name.split()
    kept: list[str] = []
    for word in reversed(words):
        if word[:1].isupper():
            kept.append(word)
            continue
        break
    if not kept:
        return None
    return " ".join(reversed(kept))


def mine_self_reference(texts: list[str]) -> str | None:
    """The entity a document calls itself, from its own definition.

    A filing states this explicitly -- ``Delhivery Limited ("Company")``
    -- so the issuer is mined the same way abbreviations are, rather
    than guessed from a filename or a title. Returns the most frequently
    defined name, or ``None`` when the document never names itself.

    Scope is deliberately ONE document: every filing's "the Company" is
    a different company, so this map must never be shared across a
    corpus the way an abbreviation glossary can be.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        if not text or "(" not in text:
            continue
        for name, _noun in _SELF_DEFINITION_RE.findall(text.replace("\n", " ")):
            proper = _trailing_proper_name(name)
            if proper:
                counts[_clean(proper)] += 1
    if not counts:
        return None
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def build_self_reference_map(
    texts: list[str], canonical_issuer: str | None
) -> dict[str, str]:
    """Map self-reference words to the document's own canonical issuer.

    Without this every "the Company" and "We" is an anonymous entity
    that can never match the issuer's real name in another document --
    and worse, two different issuers both reduce to "company" and look
    like the same entity.
    """
    if not canonical_issuer or not canonical_issuer.strip():
        return {}
    _ = texts  # the caller mines the issuer; kept for call-site symmetry
    target = canonical_issuer.strip()
    return {
        word: target
        for word in sorted(SELF_REFERENCE_NOUNS | _SELF_PRONOUNS)
        if word != target
    }
