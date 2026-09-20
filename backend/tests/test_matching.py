"""R-TST: matching-layer tests (compare, discovery, service, judgment, API).

Conventions:
- Synthetic Facts are built directly in code with neutral wording
  ("Alpha metric", "Widget output"); two fake document_ids per pair.
- Facts go through ``save_facts`` into the shared ``db_session`` fixture
  (file-backed SQLite, full schema). No PDFs are needed except via the
  lightweight ingestion-bundle helper for API tests (no PDF bytes).
- LLM access uses duck-typed fake transports only (``complete_text``),
  no network, no API keys. Embeddings are unavailable here
  (sentence-transformers not installed) so the lexical path is exercised.
- Production code is never modified from this module.
"""

from __future__ import annotations

import json
from datetime import date
from uuid import uuid4

from app.core.config import settings
from app.db.repositories import (
    get_fact,
    get_relationship,
    list_all_facts,
    list_relationships_for_document,
    save_facts,
    save_ingestion,
)
from app.matching.compare import (
    RELATIVE_TOLERANCE,
    claims_comparable,
    classify,
    compare_context,
    compare_numeric,
    semantic_overlap,
    strong_match,
)
from app.matching.embeddings import lexical_score
from app.matching.service import (
    discover_candidates,
    get_fact_embedder,
    run_matching_for_document,
)
from app.models import (
    DimensionVerdict,
    Document,
    EstimateStatus,
    EvidenceType,
    EvidenceUnit,
    ExtractionMethod,
    Fact,
    IngestionStatus,
    NumericVerdict,
    Page,
    QualityVerdict,
    RelationshipStatus,
    RelationshipType,
    TimeKind,
    ValueKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_fact(document_id, evidence_ids, **overrides):
    """Neutral synthetic Fact with corroboration-friendly defaults."""
    kwargs = {
        "document_id": document_id,
        "subject": "Alpha metric",
        "canonical_subject": "alpha_metric",
        "predicate": "reported total",
        "canonical_predicate": "reported_total",
        "value_kind": ValueKind.NUMERIC,
        "value_text": "5000",
        "value_number": 5000.0,
        "unit": "count",
        "normalized_number": 5000.0,
        "normalized_unit": "count",
        "time_text": "FY24",
        "time_kind": TimeKind.FISCAL_YEAR,
        "scope_text": "Scope Alpha",
        "estimate_status": EstimateStatus.ACTUAL,
        "geography": "Region Alpha",
        "evidence_ids": list(evidence_ids),
        "extraction_confidence": 0.8,
    }
    kwargs.update(overrides)
    return Fact(**kwargs)


def _seed_doc_with_evidence(db_session, filename):
    """Minimal ingestion bundle (no PDF bytes) so API routes see the doc."""
    doc = Document(
        id=uuid4(),
        filename=filename,
        file_sha256="0" * 64,
        page_count=1,
        ingestion_status=IngestionStatus.COMPLETED,
    )
    page = Page(
        document_id=doc.id,
        pdf_page_number=0,
        width=100.0,
        height=100.0,
        quality_verdict=QualityVerdict.GOOD,
        extraction_quality=0.9,
    )
    ev = EvidenceUnit(
        id=uuid4(),
        document_id=doc.id,
        pdf_page_number=0,
        type=EvidenceType.TEXT,
        text="Synthetic neutral evidence sentence.",
        extraction_method=ExtractionMethod.PYMUPDF,
    )
    save_ingestion(db_session, doc, [page], [ev])
    db_session.commit()
    return doc, ev


def _run(db_session, document_id, transport=None):
    report = run_matching_for_document(
        db_session, str(document_id), embedder=None, transport=transport
    )
    db_session.commit()
    return report


class _CannedTransport:
    """Duck-typed judgment transport returning canned raw model text."""

    def __init__(self, text):
        self._text = text
        self.prompts: list[str] = []

    def complete_text(self, prompt):
        self.prompts.append(prompt)
        return self._text


def _ambiguous_pair(doc_a, doc_b):
    """Overlap-only + equivalent numerics -> needs_llm provisional RELATED.

    Canonicals are absent (no strong match) but raw subject/predicate
    tokens are identical (semantic overlap); normalized values agree and
    every known context dimension is compatible.
    """
    shared = {
        "canonical_subject": None,
        "canonical_predicate": None,
        "normalized_number": 5000.0,
        "normalized_unit": "count",
        "estimate_status": EstimateStatus.UNKNOWN,
        "geography": None,
        "scope_text": None,
    }
    return (
        _mk_fact(doc_a, [uuid4()], **shared),
        _mk_fact(doc_b, [uuid4()], **shared),
    )


# ---------------------------------------------------------------------------
# 0. Tolerance contract sanity
# ---------------------------------------------------------------------------


def test_relative_tolerance_value_and_boundary():
    assert RELATIVE_TOLERANCE == 0.01
    doc_a, doc_b = uuid4(), uuid4()
    base = {"canonical_subject": "alpha_metric",
            "canonical_predicate": "reported_total",
            "normalized_unit": "count"}
    at_edge = (
        _mk_fact(doc_a, [uuid4()], normalized_number=10000.0, **base),
        _mk_fact(doc_b, [uuid4()], normalized_number=10100.0,
                 value_text="10100", value_number=10100.0, **base),
    )
    # 100/10100 ~= 0.0099 <= 0.01 -> APPROXIMATE
    assert compare_numeric(*at_edge) == NumericVerdict.APPROXIMATE
    beyond = (
        _mk_fact(doc_a, [uuid4()], normalized_number=10000.0, **base),
        _mk_fact(doc_b, [uuid4()], normalized_number=10200.0,
                 value_text="10200", value_number=10200.0, **base),
    )
    # 200/10200 ~= 0.0196 > 0.01 -> DIFFERENT
    assert compare_numeric(*beyond) == NumericVerdict.DIFFERENT


# ---------------------------------------------------------------------------
# 1. Corroboration: exact + approximate, compatible FY context
# ---------------------------------------------------------------------------


def test_classify_corroborates_exact_and_approximate():
    doc_a, doc_b = uuid4(), uuid4()
    exact_b = _mk_fact(doc_b, [uuid4()])
    rtype, conf, _expl, _meta, needs_llm = classify(
        _mk_fact(doc_a, [uuid4()]), exact_b
    )
    assert rtype == RelationshipType.CORROBORATES
    assert conf == 0.95
    assert needs_llm is False

    approx = _mk_fact(
        doc_b, [uuid4()], value_text="5020", value_number=5020.0,
        normalized_number=5020.0,
    )
    rtype, conf, _expl, meta, needs_llm = classify(
        _mk_fact(doc_a, [uuid4()]), approx
    )
    assert rtype == RelationshipType.CORROBORATES
    assert conf == 0.9
    assert meta["numeric_verdict"] == NumericVerdict.APPROXIMATE.value
    assert needs_llm is False


def test_service_corroborates_resolved_high_confidence(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    save_facts(db_session, [
        _mk_fact(doc_a, [uuid4()]),
        _mk_fact(doc_b, [uuid4()]),
    ])
    db_session.commit()
    report = _run(db_session, doc_a)
    assert len(report.relationships) == 1
    rel = report.relationships[0]
    assert rel.relationship_type == RelationshipType.CORROBORATES
    assert rel.confidence >= 0.9
    assert rel.status == RelationshipStatus.RESOLVED
    assert rel.explanation  # non-empty by contract


# ---------------------------------------------------------------------------
# 2. Corroboration via wording variant (canonicals decide, not strings)
# ---------------------------------------------------------------------------


def test_wording_variant_still_corroborates():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="ALPHA METRIC (revised label)",
        predicate="total reported output",
    )
    assert fact_a.subject != fact_b.subject
    assert fact_a.predicate != fact_b.predicate
    assert strong_match(fact_a, fact_b) is True  # canonicals agree
    rtype, conf, _expl, _meta, _needs = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CORROBORATES
    assert conf >= 0.9


# ---------------------------------------------------------------------------
# 3. Quantity corroboration across raw scales
# ---------------------------------------------------------------------------


def test_scaled_counts_corroborate():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()], value_text="1,429,000", value_number=1429000.0,
        normalized_number=1429000.0,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()], value_text="1.429M", value_number=1429000.0,
        normalized_number=1429000.0,
    )
    assert compare_numeric(fact_a, fact_b) == NumericVerdict.EXACT
    rtype, conf, _expl, _meta, _needs = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CORROBORATES
    assert conf >= 0.9


# ---------------------------------------------------------------------------
# 4. Contradiction: materially different values, same context
# ---------------------------------------------------------------------------


def test_contradicts_on_materially_different_values():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()], normalized_number=1000.0,
                      value_text="1000", value_number=1000.0)
    fact_b = _mk_fact(doc_b, [uuid4()], normalized_number=2000.0,
                      value_text="2000", value_number=2000.0)
    assert compare_numeric(fact_a, fact_b) == NumericVerdict.DIFFERENT
    assert strong_match(fact_a, fact_b) is True
    ctx = compare_context(fact_a, fact_b)
    assert ctx.incompatible == []
    assert ctx.unknown == []
    rtype, conf, _expl, meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTRADICTS
    assert conf == 0.85
    assert needs_llm is False
    assert meta["unknown_dimensions"] == []


# ---------------------------------------------------------------------------
# 5. Contradiction with unknown context: capped at 0.7, unknowns recorded
# ---------------------------------------------------------------------------


def test_contradicts_capped_with_unknown_context():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()], normalized_number=1000.0,
                      value_text="1000", value_number=1000.0)
    fact_b = _mk_fact(
        doc_b, [uuid4()], normalized_number=2000.0,
        value_text="2000", value_number=2000.0, geography=None,
    )
    rtype, conf, _expl, meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTRADICTS  # unknown != incompatible
    assert conf == 0.7
    assert "geography" in meta["unknown_dimensions"]
    assert needs_llm is False


# ---------------------------------------------------------------------------
# 6. Contextual difference vs unknown-context pairs
# ---------------------------------------------------------------------------


def test_contextual_difference_on_estimate_and_geography():
    doc_a, doc_b = uuid4(), uuid4()
    est_b = _mk_fact(doc_b, [uuid4()],
                     estimate_status=EstimateStatus.ESTIMATE)
    rtype, conf, expl, meta, needs_llm = classify(
        _mk_fact(doc_a, [uuid4()]), est_b
    )
    assert rtype == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert conf == 0.75
    assert "estimate_status" in meta["incompatible_dimensions"]
    assert "estimate" in expl.lower()
    assert needs_llm is False

    geo_b = _mk_fact(doc_b, [uuid4()], geography="Region Beta")
    rtype, _conf, _expl, meta, _needs = classify(
        _mk_fact(doc_a, [uuid4()]), geo_b
    )
    assert rtype == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert "geography" in meta["incompatible_dimensions"]


def test_differing_values_with_unknown_context_not_contextual():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()], normalized_number=1000.0,
                      value_text="1000", value_number=1000.0)
    fact_b = _mk_fact(
        doc_b, [uuid4()], normalized_number=2000.0,
        value_text="2000", value_number=2000.0,
        estimate_status=EstimateStatus.UNKNOWN,
    )
    rtype, conf, _expl, meta, _needs = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTRADICTS  # capped, not contextual
    assert rtype != RelationshipType.CONTEXTUAL_DIFFERENCE
    assert conf == 0.7
    assert "estimate_status" in meta["unknown_dimensions"]


# ---------------------------------------------------------------------------
# 7. Related: partial overlap, incomparable values
# ---------------------------------------------------------------------------


def test_related_on_incomparable_values_with_overlap():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(
        doc_b, [uuid4()], value_kind=ValueKind.TEXT, value_text="high",
        value_number=None, unit=None,
        normalized_number=None, normalized_unit=None,
    )
    assert compare_numeric(fact_a, fact_b) == NumericVerdict.INCOMPARABLE
    assert strong_match(fact_a, fact_b) is False  # kinds differ
    assert semantic_overlap(fact_a, fact_b) is True  # shared canonicals
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.RELATED
    assert conf == 0.5
    assert needs_llm is False


# ---------------------------------------------------------------------------
# 8. Unrelated: disjoint pair + never persisted
# ---------------------------------------------------------------------------


def test_unrelated_disjoint_pair_never_persisted(db_session, monkeypatch):
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(
        doc_b, [uuid4()], subject="Zebra harvest",
        canonical_subject="zebra_harvest", predicate="observed yield",
        canonical_predicate="observed_yield", value_text="7",
        value_number=7.0, normalized_number=7.0, normalized_unit="count",
        time_text=None, time_kind=TimeKind.UNKNOWN,
        geography=None, scope_text=None,
    )
    assert semantic_overlap(fact_a, fact_b) is False
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert conf == 0.9
    assert needs_llm is False

    # Service level: floor 0 so the disjoint pair is evaluated, not skipped.
    monkeypatch.setattr(settings, "matching_similarity_floor", 0.0)
    save_facts(db_session, [fact_a, fact_b])
    db_session.commit()
    report = _run(db_session, doc_a)
    assert report.candidates_evaluated == 1
    assert report.unrelated_count == 1
    assert report.relationships == []
    assert list_relationships_for_document(db_session, doc_a) == []


# ---------------------------------------------------------------------------
# 9. Missing canonicals: unknown passthrough keeps the candidate
# ---------------------------------------------------------------------------


def test_missing_canonicals_survive_prefilter():
    doc_a, doc_b = uuid4(), uuid4()
    shared = {
        "canonical_subject": None,
        "canonical_predicate": None,
        "value_number": None,
        "normalized_number": None,
        "normalized_unit": None,
        "unit": None,
    }
    fact_a = _mk_fact(doc_a, [uuid4()], **shared)
    fact_b = _mk_fact(doc_b, [uuid4()], **shared)
    assert strong_match(fact_a, fact_b) is False
    assert lexical_score(fact_a, fact_b) == 1.0  # strong raw overlap
    pool = [fact_a, fact_b]
    cands = discover_candidates([fact_a], pool, {}, top_k=10, floor=0.25)
    pair_ids = {(a, b) for a, b, _s in cands}
    ordered = tuple(sorted((fact_a.id, fact_b.id), key=str))
    assert ordered in pair_ids


# ---------------------------------------------------------------------------
# 10. Embeddings unavailable: lexical path still classifies
# ---------------------------------------------------------------------------


def test_embedder_unavailable_and_lexical_path_classifies(db_session):
    assert get_fact_embedder() is None  # sentence-transformers absent here
    doc_a, doc_b = uuid4(), uuid4()
    save_facts(db_session, [
        _mk_fact(doc_a, [uuid4()]),
        _mk_fact(doc_b, [uuid4()]),
    ])
    db_session.commit()
    report = run_matching_for_document(
        db_session, str(doc_a), embedder=None, transport=None
    )
    db_session.commit()
    assert len(report.relationships) == 1
    assert report.relationships[0].relationship_type == (
        RelationshipType.CORROBORATES
    )


# ---------------------------------------------------------------------------
# 11. Similarity never classifies: unit mismatch is not corroboration
# ---------------------------------------------------------------------------


def test_unit_mismatch_never_corroborates_or_contradicts():
    doc_a, doc_b = uuid4(), uuid4()
    base = {
        "canonical_subject": None,  # raw words shared -> high lexical score
        "canonical_predicate": None,
        "normalized_number": 100.0,
        "value_text": "100",
        "value_number": 100.0,
    }
    cash_a = _mk_fact(doc_a, [uuid4()], normalized_unit="inr_crore",
                      unit="crore", **base)
    heads_b = _mk_fact(doc_b, [uuid4()], normalized_unit="headcount",
                       unit="heads", **base)
    assert lexical_score(cash_a, heads_b) >= 0.9  # high similarity...
    assert compare_numeric(cash_a, heads_b) == NumericVerdict.INCOMPARABLE
    rtype, _c, _e, _m, _n = classify(cash_a, heads_b)
    assert rtype not in (
        RelationshipType.CORROBORATES, RelationshipType.CONTRADICTS,
    )

    # Unit present-vs-missing lands RELATED (unknown, not incompatible).
    nounit_b = _mk_fact(doc_b, [uuid4()], normalized_unit=None, unit=None,
                        **base)
    rtype2, _c2, _e2, _m2, _n2 = classify(cash_a, nounit_b)
    assert rtype2 == RelationshipType.RELATED
    assert rtype2 not in (
        RelationshipType.CORROBORATES, RelationshipType.CONTRADICTS,
    )


# ---------------------------------------------------------------------------
# 12. FY vs calendar time kinds never corroborate on time grounds
# ---------------------------------------------------------------------------


def test_fiscal_year_vs_calendar_never_corroborates():
    doc_a, doc_b = uuid4(), uuid4()
    fiscal = _mk_fact(doc_a, [uuid4()])
    cal_date = _mk_fact(
        doc_b, [uuid4()], time_text="2024-05-17",
        time_kind=TimeKind.DATE, time_start=date(2024, 5, 17),
        time_end=date(2024, 5, 17),
    )
    ctx = compare_context(fiscal, cal_date)
    assert ctx.dimensions["time"] == DimensionVerdict.INCOMPATIBLE
    rtype, _c, _e, _m, _n = classify(fiscal, cal_date)
    assert rtype != RelationshipType.CORROBORATES

    cal_range = _mk_fact(
        doc_b, [uuid4()], time_text="2024-04-01 to 2024-06-30",
        time_kind=TimeKind.RANGE, time_start=date(2024, 4, 1),
        time_end=date(2024, 6, 30),
    )
    ctx2 = compare_context(fiscal, cal_range)
    assert ctx2.dimensions["time"] == DimensionVerdict.INCOMPATIBLE
    rtype2, _c2, _e2, _m2, _n2 = classify(fiscal, cal_range)
    assert rtype2 != RelationshipType.CORROBORATES


# ---------------------------------------------------------------------------
# 13. LLM unavailable on ambiguous pair: provisional RELATED, NEEDS_REVIEW
# ---------------------------------------------------------------------------


def test_ambiguous_pair_without_llm_is_provisional(db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "")
    doc_a, doc_b = uuid4(), uuid4()
    fact_a, fact_b = _ambiguous_pair(doc_a, doc_b)
    _t, _c, _e, _m, needs_llm = classify(fact_a, fact_b)
    assert needs_llm is True  # precondition: genuinely ambiguous
    save_facts(db_session, [fact_a, fact_b])
    db_session.commit()
    report = _run(db_session, doc_a, transport=None)
    assert len(report.relationships) == 1
    rel = report.relationships[0]
    assert rel.relationship_type == RelationshipType.RELATED  # provisional kept
    assert rel.status == RelationshipStatus.NEEDS_REVIEW  # not certain
    assert rel.confidence <= 0.5
    assert rel.reasoning_metadata.get("needs_review_reason") == "llm_unavailable"


# ---------------------------------------------------------------------------
# 14. LLM judgment success via fake transport is adopted
# ---------------------------------------------------------------------------


def test_fake_llm_judgment_adopted(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    fact_a, fact_b = _ambiguous_pair(doc_a, doc_b)
    save_facts(db_session, [fact_a, fact_b])
    db_session.commit()
    canned = json.dumps({
        "relationship_type": "CONTEXTUAL_DIFFERENCE",
        "confidence": 0.82,
        "explanation": "synthetic judgment explanation",
    })
    fake = _CannedTransport(canned)
    report = _run(db_session, doc_a, transport=fake)
    assert fake.prompts and all(p.strip() for p in fake.prompts)
    assert len(report.relationships) == 1
    rel = report.relationships[0]
    assert rel.relationship_type == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert rel.confidence == 0.82
    assert rel.explanation == "synthetic judgment explanation"
    assert rel.status == RelationshipStatus.RESOLVED


# ---------------------------------------------------------------------------
# 15. Malformed LLM output: provisional preserved, no exception
# ---------------------------------------------------------------------------


def test_malformed_llm_output_stays_provisional(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    save_facts(db_session, list(_ambiguous_pair(doc_a, doc_b)))
    db_session.commit()
    fake = _CannedTransport("this is not json{{{garbage")
    report = _run(db_session, doc_a, transport=fake)  # must not raise
    assert len(report.relationships) == 1
    rel = report.relationships[0]
    assert rel.relationship_type == RelationshipType.RELATED
    assert rel.status == RelationshipStatus.NEEDS_REVIEW
    assert rel.confidence <= 0.5
    reason = rel.reasoning_metadata.get("needs_review_reason", "")
    assert "LLMMalformedError" in reason


# ---------------------------------------------------------------------------
# 16. Provenance round-trip: relationship -> facts -> evidence ids
# ---------------------------------------------------------------------------


def test_provenance_round_trip(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    ev_a, ev_b = uuid4(), uuid4()
    save_facts(db_session, [
        _mk_fact(doc_a, [ev_a]),
        _mk_fact(doc_b, [ev_b]),
    ])
    db_session.commit()
    report = _run(db_session, doc_a)
    assert len(report.relationships) == 1
    stored = get_relationship(db_session, report.relationships[0].id)
    assert stored is not None
    assert stored.relationship_type == RelationshipType.CORROBORATES
    fact_a = get_fact(db_session, stored.fact_a_id)
    fact_b = get_fact(db_session, stored.fact_b_id)
    assert fact_a is not None and fact_b is not None
    assert fact_a.evidence_ids and fact_b.evidence_ids
    assert set(fact_a.evidence_ids) | set(fact_b.evidence_ids) == {ev_a, ev_b}


# ---------------------------------------------------------------------------
# 17. Dedup: second run persists nothing new; same-doc facts never pair
# ---------------------------------------------------------------------------


def test_second_run_persists_nothing_new(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    save_facts(db_session, [
        _mk_fact(doc_a, [uuid4()]),
        _mk_fact(doc_b, [uuid4()]),
    ])
    db_session.commit()
    first = _run(db_session, doc_a)
    assert len(first.relationships) == 1
    assert first.skipped_existing == 0
    second = _run(db_session, doc_a)
    assert len(second.relationships) == 0
    assert second.skipped_existing > 0
    rows = list_relationships_for_document(db_session, doc_a)
    assert len(rows) == 1


def test_same_document_facts_never_pair(db_session):
    doc_a = uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(doc_a, [uuid4()])
    pool = [fact_a, fact_b]
    assert discover_candidates([fact_a, fact_b], pool, {}, top_k=10,
                               floor=0.0) == []
    save_facts(db_session, pool)
    db_session.commit()
    report = _run(db_session, doc_a)
    assert report.candidates_evaluated == 0
    assert report.relationships == []


# ---------------------------------------------------------------------------
# 18. Top-k boundedness: no all-pairs
# ---------------------------------------------------------------------------


def test_discovery_is_top_k_bounded():
    doc_a = uuid4()
    query = _mk_fact(doc_a, [uuid4()])
    pool = [query] + [_mk_fact(uuid4(), [uuid4()]) for _ in range(12)]
    cands = discover_candidates([query], pool, {}, top_k=3, floor=0.0)
    assert 0 < len(cands) <= 3


# ---------------------------------------------------------------------------
# 19. API: trigger, filtered list, detail, 404s
# ---------------------------------------------------------------------------


def _seed_api_pair(api_client, db_session):
    doc_a, ev_a = _seed_doc_with_evidence(db_session, "api-a.pdf")
    doc_b, ev_b = _seed_doc_with_evidence(db_session, "api-b.pdf")
    save_facts(db_session, [
        _mk_fact(doc_a.id, [ev_a.id]),
        _mk_fact(doc_b.id, [ev_b.id]),
    ])
    db_session.commit()
    return doc_a, doc_b, ev_a, ev_b


def test_api_trigger_list_detail_and_404s(api_client, db_session):
    doc_a, doc_b, ev_a, ev_b = _seed_api_pair(api_client, db_session)

    post = api_client.post(f"/documents/{doc_a.id}/relationships")
    assert post.status_code == 200
    body = post.json()
    assert len(body["relationships"]) == 1
    rel_body = body["relationships"][0]
    assert rel_body["relationship_type"] == "CORROBORATES"
    assert rel_body["confidence"] >= 0.9
    assert rel_body["status"] == "RESOLVED"
    rel_id = rel_body["id"]

    got = api_client.get(
        f"/documents/{doc_a.id}/relationships",
        params={"relationship_type": "CORROBORATES", "min_confidence": 0.9},
    )
    assert got.status_code == 200
    items = got.json()
    assert len(items) == 1
    assert items[0]["id"] == rel_id

    empty = api_client.get(
        f"/documents/{doc_a.id}/relationships",
        params={"relationship_type": "CORROBORATES", "min_confidence": 0.99},
    )
    assert empty.status_code == 200
    assert empty.json() == []

    wrong_type = api_client.get(
        f"/documents/{doc_a.id}/relationships",
        params={"relationship_type": "CONTRADICTS"},
    )
    assert wrong_type.status_code == 200
    assert wrong_type.json() == []

    detail = api_client.get(f"/relationships/{rel_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["relationship"]["id"] == rel_id
    fact_ids = {payload["fact_a"]["id"], payload["fact_b"]["id"]}
    assert len(fact_ids) == 2
    assert payload["evidence_a"]  # evidence lists populated
    assert payload["evidence_b"]
    seen_evidence = {e["id"] for e in payload["evidence_a"]}
    seen_evidence |= {e["id"] for e in payload["evidence_b"]}
    assert {str(ev_a.id), str(ev_b.id)} <= seen_evidence

    missing = str(uuid4())
    assert api_client.post(f"/documents/{missing}/relationships").status_code == 404
    assert api_client.get(f"/documents/{missing}/relationships").status_code == 404
    assert api_client.get(f"/relationships/{missing}").status_code == 404


def test_canonical_only_link_payables_vs_scrip_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="accounts payable days outstanding reported",
        predicate="days payable outstanding reported",
        canonical_subject="acme_corp",
        canonical_predicate="payables_days",
        value_text="45", value_number=45.0,
        normalized_number=45.0, normalized_unit="days", unit="days",
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="scrip code listing reported",
        predicate="identifier code reported",
        canonical_subject="acme_corp",
        canonical_predicate="scrip_code",
        value_text="45", value_number=45.0,
        normalized_number=45.0, normalized_unit="days", unit="days",
    )
    assert semantic_overlap(fact_a, fact_b) is True
    assert claims_comparable(fact_a, fact_b) is False
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert conf == 0.9
    assert needs_llm is False


def test_canonical_only_link_website_vs_related_party_share_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="company website address reported",
        predicate="official web address reported",
        canonical_subject="acme_corp",
        canonical_predicate="website",
        value_text="1", value_number=1.0,
        normalized_number=1.0, normalized_unit="count",
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="related party share reported",
        predicate="shareholding percentage reported",
        canonical_subject="acme_corp",
        canonical_predicate="related_party_share",
        value_text="1", value_number=1.0,
        normalized_number=1.0, normalized_unit="count",
    )
    assert semantic_overlap(fact_a, fact_b) is True
    assert claims_comparable(fact_a, fact_b) is False
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert conf == 0.9
    assert needs_llm is False


def test_canonical_only_link_communication_date_vs_awareness_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="date of communication reported",
        predicate="notice issue date reported",
        canonical_subject="acme_corp",
        canonical_predicate="communication_date",
        value_text="2", value_number=2.0,
        normalized_number=2.0, normalized_unit="count",
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="awareness coverage reported",
        predicate="population reached reported",
        canonical_subject="acme_corp",
        canonical_predicate="awareness_coverage",
        value_text="2", value_number=2.0,
        normalized_number=2.0, normalized_unit="count",
    )
    assert semantic_overlap(fact_a, fact_b) is True
    assert claims_comparable(fact_a, fact_b) is False
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert conf == 0.9
    assert needs_llm is False


def test_macro_gdp_pair_stays_contextual_difference_capable():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="real GDP growth",
        predicate="projected",
        canonical_subject="gdp_growth",
        canonical_predicate="projected_growth",
        value_text="3", value_number=3.0,
        normalized_number=3.0, normalized_unit="percent", unit="percent",
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="real gross domestic product (GDP) growth",
        predicate="estimated growth rate",
        canonical_subject="gdp_growth",
        canonical_predicate="estimated_growth",
        value_text="3", value_number=3.0,
        normalized_number=3.0, normalized_unit="percent", unit="percent",
    )
    assert claims_comparable(fact_a, fact_b) is True
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert conf == 0.75
    assert needs_llm is False


def test_ebitda_pair_stays_corroboration_class():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="EBITDA",
        predicate="annual value",
        canonical_subject="ebitda",
        canonical_predicate="annual_value",
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="EBITDA",
        predicate="annual value",
        canonical_subject="ebitda",
        canonical_predicate="annual_value",
    )
    assert claims_comparable(fact_a, fact_b) is True
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CORROBORATES
    assert conf == 0.95
    assert needs_llm is False


def test_near_synonym_predicates_stay_comparable():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        predicate="revenue from services",
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        predicate="FY24 services revenue",
        geography="Region Beta",
    )
    assert claims_comparable(fact_a, fact_b) is True
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert conf == 0.75
    assert needs_llm is False


def test_scaffolding_token_overlap_is_not_comparable():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="FY24 EBITDA",
        predicate="value",
        canonical_subject=None,
        canonical_predicate=None,
        value_text="Rs. 127 Cr",
        value_number=127.0,
        normalized_number=127.0,
        normalized_unit="INR crore",
        unit="INR crore",
        time_text="FY24",
        time_kind=TimeKind.FISCAL_YEAR,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="software",
        predicate="value",
        canonical_subject=None,
        canonical_predicate=None,
        value_text="Rs. 2418.03 million",
        value_number=2418.03,
        normalized_number=241.803,
        normalized_unit="INR crore",
        unit="INR crore",
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
    )
    assert claims_comparable(fact_a, fact_b) is False
    rtype, _conf, _expl, meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert meta["numeric_verdict"] == "DIFFERENT"
    assert needs_llm is False


def test_shared_company_token_is_not_comparable():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="Delhivery",
        predicate="has",
        canonical_subject=None,
        canonical_predicate=None,
        value_kind=ValueKind.TEXT,
        value_text="technology-driven systems",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="Delhivery",
        predicate="maintains",
        canonical_subject=None,
        canonical_predicate=None,
        value_kind=ValueKind.TEXT,
        value_text="contingency and redundancy",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    rtype, _conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert needs_llm is False


def test_role_vs_identifier_same_entity_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="Mr. Sahil Barua",
        canonical_subject="sahil_barua",
        predicate="Role",
        canonical_predicate="role",
        value_kind=ValueKind.TEXT,
        value_text="Managing Director & Chief Executive Officer",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="Sahil Barua",
        canonical_subject="sahil_barua",
        predicate="DIN",
        canonical_predicate="din",
        value_text="05131571",
        value_number=5131571.0,
        normalized_number=5131571.0,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    rtype, _conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert needs_llm is False


def test_tenure_vs_role_text_pair_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="Deepak Kapoor",
        canonical_subject="deepak_kapoor",
        predicate="Period of Directorship",
        canonical_predicate="directorship_period",
        value_kind=ValueKind.TEXT,
        value_text="Since November 22, 2017",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="Deepak Kapoor",
        canonical_subject="deepak_kapoor",
        predicate="role",
        canonical_predicate="role",
        value_kind=ValueKind.TEXT,
        value_text="Chairperson & Non-Executive Independent Director",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    rtype, _conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert needs_llm is False


def test_revenue_growth_vs_unrelated_percentage_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="Revenue from services",
        canonical_subject=None,
        predicate="growth",
        canonical_predicate=None,
        value_text="36%",
        value_number=36.0,
        normalized_number=0.36,
        normalized_unit="fraction",
        unit="%",
        time_text="QoQ",
        time_kind=TimeKind.QUARTER,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="private consumption expenditure growth",
        canonical_subject=None,
        predicate="fell from",
        canonical_predicate=None,
        value_text="6.8%",
        value_number=6.8,
        normalized_number=0.068,
        normalized_unit="fraction",
        unit="%",
        time_text="in FY23",
        time_kind=TimeKind.FISCAL_YEAR,
        geography=None,
        scope_text=None,
    )
    assert claims_comparable(fact_a, fact_b) is False
    rtype, _conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert needs_llm is False


def test_identifier_pair_with_canonicals_stays_llm_provisional():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_b, [uuid4()],
        subject="Deepak Kapoor",
        canonical_subject="deepak_kapoor",
        predicate="DIN",
        canonical_predicate="din",
        value_text="00162957",
        value_number=162957.0,
        normalized_number=162957.0,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_a, [uuid4()],
        subject="Deepak Kapoor",
        canonical_subject="deepak_kapoor",
        predicate="DIN",
        canonical_predicate="din",
        value_text="00162957",
        value_number=162957.0,
        normalized_number=162957.0,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    rtype, conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.RELATED
    assert conf == 0.5
    assert needs_llm is True


def test_matching_rerun_persists_nothing_new(db_session):
    from app.db.models import RelationshipRow

    doc_a, doc_b = uuid4(), uuid4()
    save_facts(db_session, list(_ambiguous_pair(doc_a, doc_b)))
    db_session.commit()
    canned = json.dumps({
        "relationship_type": "CORROBORATES",
        "confidence": 0.9,
        "explanation": "synthetic corroboration",
    })
    first = _run(db_session, doc_a, transport=_CannedTransport(canned))
    assert len(first.relationships) == 1
    assert first.skipped_existing == 0
    second = _run(db_session, doc_a, transport=_CannedTransport(canned))
    assert second.relationships == []
    assert second.skipped_existing == 1
    assert db_session.query(RelationshipRow).count() == 1


def test_same_predicate_different_entities_text_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="Sahil Barua",
        canonical_subject="sahil_barua",
        predicate="role",
        canonical_predicate="role",
        value_kind=ValueKind.TEXT,
        value_text="Managing Director",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="Deepak Kapoor",
        canonical_subject="deepak_kapoor",
        predicate="role",
        canonical_predicate="role",
        value_kind=ValueKind.TEXT,
        value_text="Chairperson",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    assert claims_comparable(fact_a, fact_b) is True
    rtype, _conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert needs_llm is False


def test_scaffolding_only_overlap_is_unrelated():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(
        doc_a, [uuid4()],
        subject="Alpha",
        canonical_subject=None,
        predicate="reported status",
        canonical_predicate=None,
        value_kind=ValueKind.TEXT,
        value_text="complete",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        subject="Beta",
        canonical_subject=None,
        predicate="reported status",
        canonical_predicate=None,
        value_kind=ValueKind.TEXT,
        value_text="pending",
        value_number=None,
        normalized_number=None,
        normalized_unit=None,
        unit=None,
        time_text=None,
        time_kind=TimeKind.UNKNOWN,
        geography=None,
        scope_text=None,
    )
    assert claims_comparable(fact_a, fact_b) is False
    rtype, _conf, _expl, _meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.UNRELATED
    assert needs_llm is False


def test_same_measure_different_geography_is_contextual_difference():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        value_text="6000",
        value_number=6000.0,
        normalized_number=6000.0,
        geography="Region Beta",
    )
    rtype, conf, _expl, meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert conf == 0.75
    assert "geography" in meta["incompatible_dimensions"]
    assert needs_llm is False


def test_same_measure_different_period_is_contextual_difference():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        value_text="6000",
        value_number=6000.0,
        normalized_number=6000.0,
        time_text="FY23",
    )
    rtype, conf, _expl, meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTEXTUAL_DIFFERENCE
    assert conf == 0.75
    assert "time" in meta["incompatible_dimensions"]
    assert needs_llm is False


def test_same_claim_different_values_is_contradiction():
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(
        doc_b, [uuid4()],
        value_text="9000",
        value_number=9000.0,
        normalized_number=9000.0,
    )
    rtype, conf, _expl, meta, needs_llm = classify(fact_a, fact_b)
    assert rtype == RelationshipType.CONTRADICTS
    assert conf == 0.85
    assert meta["incompatible_dimensions"] == []
    assert needs_llm is False


# ---------------------------------------------------------------------------
# compare_with bounds the candidate side to the documents the user selected
# ---------------------------------------------------------------------------


def test_compare_with_bounds_candidates_to_selected_documents(db_session):
    doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
    save_facts(db_session, [
        _mk_fact(doc_a, [uuid4()]),
        _mk_fact(doc_b, [uuid4()]),
        _mk_fact(doc_c, [uuid4()]),
    ])
    db_session.commit()

    report = run_matching_for_document(
        db_session,
        str(doc_a),
        embedder=None,
        transport=None,
        compare_with=[doc_b],
    )
    db_session.commit()

    partners = set()
    for rel in report.relationships:
        partners.add(str(rel.fact_a_id))
        partners.add(str(rel.fact_b_id))
    stored = {
        str(f.document_id)
        for f in list_all_facts(db_session)
        if str(f.id) in partners
    }
    assert stored == {str(doc_a), str(doc_b)}
    assert str(doc_c) not in stored


def test_compare_with_none_still_compares_against_everything(db_session):
    doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
    save_facts(db_session, [
        _mk_fact(doc_a, [uuid4()]),
        _mk_fact(doc_b, [uuid4()]),
        _mk_fact(doc_c, [uuid4()]),
    ])
    db_session.commit()

    report = run_matching_for_document(
        db_session, str(doc_a), embedder=None, transport=None
    )
    db_session.commit()

    partners = set()
    for rel in report.relationships:
        partners.add(str(rel.fact_a_id))
        partners.add(str(rel.fact_b_id))
    stored = {
        str(f.document_id)
        for f in list_all_facts(db_session)
        if str(f.id) in partners
    }
    assert stored == {str(doc_a), str(doc_b), str(doc_c)}


# ---------------------------------------------------------------------------
# recompute: an ordinary rerun never rewrites a stored verdict; an explicit
# recompute re-classifies in place when the underlying facts have changed
# ---------------------------------------------------------------------------


def _rerun(db_session, doc_id, **kwargs):
    report = run_matching_for_document(
        db_session, str(doc_id), embedder=None, transport=None, **kwargs
    )
    db_session.commit()
    return report


def test_rerun_without_recompute_keeps_the_stored_verdict(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(doc_b, [uuid4()])
    save_facts(db_session, [fact_a, fact_b])
    db_session.commit()

    first = _rerun(db_session, doc_a)
    assert len(first.relationships) == 1
    original = first.relationships[0]
    assert original.relationship_type == RelationshipType.CORROBORATES

    # The fact now says something materially different.
    from app.db.repositories import update_fact_normalization

    fact_b.normalized_number = -999.0
    update_fact_normalization(db_session, fact_b)
    db_session.commit()

    second = _rerun(db_session, doc_a)
    assert second.skipped_existing == 1
    assert second.recomputed == 0
    stored = list_relationships_for_document(db_session, doc_a)
    assert stored[0].relationship_type == RelationshipType.CORROBORATES


def test_recompute_reclassifies_in_place_and_keeps_the_row_id(db_session):
    doc_a, doc_b = uuid4(), uuid4()
    fact_a = _mk_fact(doc_a, [uuid4()])
    fact_b = _mk_fact(doc_b, [uuid4()])
    save_facts(db_session, [fact_a, fact_b])
    db_session.commit()

    first = _rerun(db_session, doc_a)
    row_id = list_relationships_for_document(db_session, doc_a)[0].id
    assert first.relationships[0].relationship_type == RelationshipType.CORROBORATES

    from app.db.repositories import update_fact_normalization

    fact_b.normalized_number = -999.0
    update_fact_normalization(db_session, fact_b)
    db_session.commit()

    second = _rerun(db_session, doc_a, recompute=True)
    assert second.recomputed == 1
    assert second.skipped_existing == 0

    stored = list_relationships_for_document(db_session, doc_a)
    assert len(stored) == 1, "recompute must update in place, not duplicate"
    assert stored[0].id == row_id
    assert stored[0].relationship_type != RelationshipType.CORROBORATES
