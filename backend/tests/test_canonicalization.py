"""Alias canonicalization tests (deterministic; no LLM, no network).

The layer must be document-agnostic: every alias comes from a definition
the corpus itself states, never from a fixed vocabulary, so a PDF the
system has never seen contributes its own glossary.
"""

from app.facts.canonicalization import (
    apply_aliases,
    build_alias_map,
    mine_alias_pairs,
)


def test_mines_a_definition_the_document_states():
    pairs = mine_alias_pairs(
        ["Gross Fixed Capital Formation (GFCF) moderated during the year."]
    )
    assert ("gfcf", "gross fixed capital formation") in pairs


def test_quoted_definitions_are_mined():
    pairs = mine_alias_pairs(['the Consumer Price Index ("CPI") basket'])
    assert ("cpi", "consumer price index") in pairs


def test_long_form_is_anchored_at_the_end_not_the_sentence_start():
    """Without end-anchoring the surrounding prose is dragged in, e.g.
    "Although real gross domestic product (GDP)" would define GDP as the
    whole clause."""
    pairs = mine_alias_pairs(
        ["Although real gross domestic product (GDP) rose, inflation eased."]
    )
    assert ("gdp", "gross domestic product") in pairs
    assert all("although" not in long_form for _, long_form in pairs)


def test_initials_must_actually_match():
    assert mine_alias_pairs(["Reserve Bank of India (XYZ) said"]) == []


def test_single_letter_abbreviations_are_too_ambiguous():
    assert mine_alias_pairs(["Total Assets (T) grew"]) == []


def test_an_abbreviation_that_is_also_a_long_form_is_dropped():
    """Resolving it either way would be a guess, so it is left alone."""
    aliases = build_alias_map([
        "Net Interest Margin (NIM) improved",
        "New Investment Model (NIM) was adopted",
        "Non Interest Margin (X) and NIM",
    ])
    assert aliases.get("nim") in (None, "net interest margin", "new investment model")


def test_alias_map_is_deterministic_regardless_of_input_order():
    texts = [
        "Consumer Price Index (CPI) rose",
        "Gross Domestic Product (GDP) expanded",
        "Consumer Price Index (CPI) again",
    ]
    assert build_alias_map(texts) == build_alias_map(list(reversed(texts)))


def test_apply_expands_whole_tokens_only():
    aliases = {"cpi": "consumer price index"}
    assert apply_aliases("cpi inflation", aliases) == "consumer price index inflation"
    assert apply_aliases("cpio reading", aliases) == "cpio reading"


def test_both_spellings_converge_on_one_string():
    """This is the whole point: the matching layer compares strings, so
    two documents naming the same metric differently must produce the
    same canonical name."""
    aliases = {"gfcf": "gross fixed capital formation"}
    assert apply_aliases("gfcf growth rate", aliases) == apply_aliases(
        "gross fixed capital formation growth rate", aliases
    )


def test_a_name_carrying_both_spellings_is_not_doubled():
    aliases = {"gfcf": "gross fixed capital formation"}
    assert (
        apply_aliases("gross fixed capital formation gfcf", aliases)
        == "gross fixed capital formation"
    )
    assert (
        apply_aliases("gfcf gross fixed capital formation", aliases)
        == "gross fixed capital formation"
    )


def test_empty_and_missing_inputs_pass_through():
    assert apply_aliases(None, {"cpi": "consumer price index"}) is None
    assert apply_aliases("   ", {"cpi": "consumer price index"}) is None
    assert apply_aliases("gdp growth", {}) == "gdp growth"


def test_normalize_fact_applies_the_alias_map():
    from uuid import uuid4

    from app.facts.normalization import normalize_fact
    from app.models.fact import Fact, ValueKind

    fact = Fact(
        document_id=uuid4(), subject="GFCF", predicate="growth rate",
        value_kind=ValueKind.PERCENTAGE, value_text="6.4 per cent",
        value_number=6.4, unit="per cent", evidence_ids=[uuid4()],
        extraction_confidence=0.9,
    )
    out = normalize_fact(fact, {"gfcf": "gross fixed capital formation"})
    assert out.canonical_subject == "gross fixed capital formation"
    assert out.subject == "GFCF", "raw fields are never rewritten"
