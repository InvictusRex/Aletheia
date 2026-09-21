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
    "apply_aliases",
    "build_alias_map",
    "mine_alias_pairs",
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
