"""Comparability tests: what the system can actually reason about.

A fact missing a unit or a period is not merely lower quality -- it
cannot support the comparison this system exists to perform, and one
missing a period actively produces false contradictions because an
unknown period never blocks a disagreement.
"""

from uuid import uuid4

import pytest

from app.matching.compare import classify
from app.models import Comparability, RelationshipType
from app.models.fact import Fact, TimeKind, ValueKind


def _fact(**overrides):
    kwargs = dict(
        document_id=uuid4(), subject="India", predicate="real GDP growth",
        canonical_subject="india", canonical_predicate="real gdp growth",
        value_kind=ValueKind.PERCENTAGE, value_text="6.5 per cent",
        value_number=6.5, normalized_number=0.065, normalized_unit="fraction",
        time_text="FY24", time_kind=TimeKind.FISCAL_YEAR,
        evidence_ids=[uuid4()], extraction_confidence=0.9,
    )
    kwargs.update(overrides)
    return Fact(**kwargs)


def test_all_three_dimensions_make_a_fact_comparable():
    assert _fact().comparability == Comparability.COMPARABLE
    assert _fact().missing_for_comparison == []


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"time_kind": TimeKind.UNKNOWN, "time_text": None}, ["period"]),
        ({"normalized_unit": None}, ["unit"]),
        ({"normalized_unit": "   "}, ["unit"]),
    ],
)
def test_one_missing_dimension_is_partial(overrides, missing):
    fact = _fact(**overrides)
    assert fact.comparability == Comparability.PARTIAL
    assert fact.missing_for_comparison == missing


def test_a_fact_without_a_number_is_not_comparable():
    fact = _fact(normalized_number=None, normalized_unit=None,
                 time_kind=TimeKind.UNKNOWN, time_text=None)
    assert fact.comparability == Comparability.NOT_COMPARABLE
    assert fact.missing_for_comparison == ["number", "unit", "period"]


def test_comparability_is_derived_not_stored():
    """Persisting it would let it drift from the columns it summarises."""
    fact = _fact()
    assert fact.comparability == Comparability.COMPARABLE
    updated = fact.model_copy(update={"time_kind": TimeKind.UNKNOWN})
    assert updated.comparability == Comparability.PARTIAL


def test_comparability_is_exposed_in_the_serialized_fact():
    dumped = _fact().model_dump()
    assert dumped["comparability"] == Comparability.COMPARABLE
    assert dumped["missing_for_comparison"] == []


def test_a_contradiction_needs_both_sides_comparable():
    """The gate's live contribution is the period dimension.

    A missing unit is already caught upstream -- compare_numeric returns
    INCOMPARABLE, so the pair never reaches the contradiction branch.
    A missing period is not caught anywhere else, which is exactly how
    unknown periods used to manufacture disagreements.
    """
    a = _fact()
    b = _fact(document_id=uuid4(), value_text="9.9 per cent",
              value_number=9.9, normalized_number=0.099,
              time_text=None, time_kind=TimeKind.UNKNOWN)
    assert b.comparability == Comparability.PARTIAL
    rtype, _c, _e, meta, _n = classify(a, b)
    assert rtype != RelationshipType.CONTRADICTS
    assert meta.get("withheld_verdict") == "CONTRADICTS"
    assert "not comparable" in meta["withheld_reason"]
    assert "period" in meta["withheld_reason"]


def test_a_missing_unit_is_incomparable_before_the_gate_is_reached():
    a = _fact()
    b = _fact(document_id=uuid4(), value_text="9.9 per cent",
              value_number=9.9, normalized_number=0.099, normalized_unit=None)
    assert b.comparability == Comparability.PARTIAL
    assert classify(a, b)[0] != RelationshipType.CONTRADICTS


def test_a_contradiction_needs_symmetric_scope():
    """"standalone" against an unstated scope may be two different things."""
    a = _fact(scope_text="standalone")
    b = _fact(document_id=uuid4(), value_text="9.9 per cent",
              value_number=9.9, normalized_number=0.099, scope_text=None)
    rtype, _c, _e, meta, _n = classify(a, b)
    assert rtype != RelationshipType.CONTRADICTS
    assert meta["withheld_reason"] == "scope stated on only one side"


def test_a_fully_qualified_disagreement_still_contradicts():
    a = _fact(scope_text="consolidated")
    b = _fact(document_id=uuid4(), value_text="9.9 per cent",
              value_number=9.9, normalized_number=0.099,
              scope_text="consolidated")
    assert classify(a, b)[0] == RelationshipType.CONTRADICTS
