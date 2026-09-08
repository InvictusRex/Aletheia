"""N-TST: normalization tests (deterministic; no LLM, no network).

Covers ``app.facts.normalization`` against its actual implementation:

- ``normalize_fact`` populates ONLY the five reserved Fact columns
  (``normalized_number`` / ``normalized_unit`` / ``canonical_subject`` /
  ``canonical_predicate`` / ``ambiguity_flags`` with ``norm:`` additions).
- ``NormalizationReport`` / ``normalize_document_facts`` service behaviour.
- ``POST /documents/{id}/normalize`` endpoint behaviour.

Conventions:
- Synthetic ``Fact`` objects are built directly in code (neutral wording,
  no production dataset references).
- Service/endpoint integration reuses the ``run_ingestion`` +
  ``save_ingestion`` pattern with in-memory PyMuPDF PDFs, and persists
  facts via ``app.db.repositories.save_facts`` into the shared
  ``db_session`` fixture (file-backed SQLite, full schema).
"""

from __future__ import annotations

import uuid
from datetime import date
from uuid import uuid4

import pymupdf
import pytest

from app.db.repositories import list_facts_for_document, save_facts, save_ingestion
from app.extraction.pipeline import run_ingestion
from app.facts.normalization import (
    NormalizationReport,
    normalize_document_facts,
    normalize_fact,
)
from app.facts.validator import build_fact
from app.models import (
    ChunkUnit,
    EstimateStatus,
    EvidenceChunk,
    Fact,
    FactDraft,
    FactStatus,
    IngestionStatus,
    TimeKind,
    ValueKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact(**overrides) -> Fact:
    """Build a neutral synthetic Fact directly in code."""
    kwargs = {
        "document_id": uuid4(),
        "subject": "Synthetic subject",
        "predicate": "reported total",
        "value_kind": ValueKind.NUMERIC,
        "value_text": "740",
        "value_number": 740.0,
        "unit": "units",
        "evidence_ids": [uuid4()],
        "extraction_confidence": 0.8,
    }
    kwargs.update(overrides)
    return Fact(**kwargs)


def _make_text_pdf(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _ingest(db_session, filename: str, pdf_bytes: bytes):
    """Run ingestion and persist the bundle; return (document, pages, evidence)."""
    document, pages, evidence = run_ingestion(filename, pdf_bytes)
    assert document.ingestion_status == IngestionStatus.COMPLETED
    assert evidence, "precondition: ingestion must produce evidence"
    save_ingestion(db_session, document, pages, evidence)
    db_session.commit()
    return document, pages, evidence


# ---------------------------------------------------------------------------
# 1. Integers / decimals / commas: value_number passthrough, unit cleaned
# ---------------------------------------------------------------------------


def test_integer_with_commas_passes_through_with_cleaned_unit():
    fact = _fact(value_text="81,415", value_number=81415.0, unit="units")
    out = normalize_fact(fact)
    assert out.normalized_number == pytest.approx(81415.0)
    assert out.normalized_unit == "units"
    assert not any(f.startswith("norm:") for f in out.ambiguity_flags)


def test_decimal_value_passes_through():
    fact = _fact(value_text="81.415", value_number=81.415, unit="units")
    out = normalize_fact(fact)
    assert out.normalized_number == pytest.approx(81.415)
    assert out.normalized_unit == "units"


# ---------------------------------------------------------------------------
# 2. Negatives: sign preserved
# ---------------------------------------------------------------------------


def test_negative_parenthesised_value_sign_preserved():
    # "(1,234)" parses (validator-owned) to -1234.0; normalization keeps sign.
    fact = _fact(value_text="(1,234)", value_number=-1234.0, unit="units")
    out = normalize_fact(fact)
    assert out.normalized_number == pytest.approx(-1234.0)
    assert out.normalized_unit == "units"


# ---------------------------------------------------------------------------
# 3. Percentages vs percentage points: DISTINGUISHABLE
# ---------------------------------------------------------------------------


def test_percent_fraction_vs_percentage_point_distinguishable():
    pct = _fact(
        value_text="6.5%",
        value_number=6.5,
        unit="%",
        value_kind=ValueKind.PERCENTAGE,
    )
    pp = _fact(
        value_text="6.5 pp",
        value_number=6.5,
        unit="pp",
        value_kind=ValueKind.PERCENTAGE,
    )
    pp_words = _fact(
        value_text="up 6.5 percentage points",
        value_number=6.5,
        unit="percentage points",
        value_kind=ValueKind.PERCENTAGE,
    )
    out_pct = normalize_fact(pct)
    out_pp = normalize_fact(pp)
    out_pp_words = normalize_fact(pp_words)

    assert (out_pct.normalized_number, out_pct.normalized_unit) == (
        pytest.approx(0.065),
        "fraction",
    )
    assert out_pp.normalized_number == pytest.approx(6.5)
    assert out_pp.normalized_unit == "percentage_point"
    assert out_pp_words.normalized_number == pytest.approx(6.5)
    assert out_pp_words.normalized_unit == "percentage_point"

    # The two readings must be distinguishable: different unit AND number.
    assert out_pct.normalized_unit != out_pp.normalized_unit
    assert out_pct.normalized_number != pytest.approx(out_pp.normalized_number)


# ---------------------------------------------------------------------------
# 4. Scaling: K / Mn / Cr / lakh / B / T
# ---------------------------------------------------------------------------


def test_inr_magnitude_scales_to_inr_crore():
    # "Rs Mn" with validator-scaled value 81415000000.0 -> /1e7.
    fact = _fact(
        value_text="Rs 81415 Mn", value_number=81415000000.0, unit="Rs Mn"
    )
    out = normalize_fact(fact)
    assert out.normalized_number == pytest.approx(8141.5)
    assert out.normalized_unit == "INR crore"


@pytest.mark.parametrize(
    ("value_text", "value_number", "unit"),
    [
        ("1,429K tonnes", 1429000.0, "K tonnes"),
        ("1.4 Mn tonnes", 1400000.0, "Mn tonnes"),
    ],
)
def test_mass_magnitudes_scale_to_million_tonnes(value_text, value_number, unit):
    fact = _fact(value_text=value_text, value_number=value_number, unit=unit)
    out = normalize_fact(fact)
    assert out.normalized_unit == "million tonnes"
    assert out.normalized_number == pytest.approx(value_number / 1e6)


def test_equivalent_scaled_magnitudes_share_canonical_unit():
    # Same order of magnitude expressed as Mn vs Cr -> same canonical unit.
    via_mn = _fact(
        value_text="Rs 81415 Mn", value_number=81415000000.0, unit="Rs Mn"
    )
    via_cr = _fact(
        value_text="Rs 8142 Cr", value_number=81420000000.0, unit="Rs Cr"
    )
    out_mn = normalize_fact(via_mn)
    out_cr = normalize_fact(via_cr)
    assert out_mn.normalized_unit == out_cr.normalized_unit == "INR crore"
    assert out_mn.normalized_number == pytest.approx(8141.5)
    assert out_cr.normalized_number == pytest.approx(8142.0)


# ---------------------------------------------------------------------------
# 5. Currency: INR / USD canonical scales, unknown tokens, no cross-convert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["Rs", "Rs.", "INR", "\u20b9"])
def test_inr_currency_tokens_canonicalise_to_inr_crore(unit):
    fact = _fact(value_text=f"{unit} 100", value_number=100.0, unit=unit)
    out = normalize_fact(fact)
    assert out.normalized_unit == "INR crore"
    assert out.normalized_number == pytest.approx(100.0 / 1e7)


@pytest.mark.parametrize("unit", ["$", "USD"])
def test_usd_currency_tokens_canonicalise_to_usd_million(unit):
    fact = _fact(value_text="5 million", value_number=5000000.0, unit=unit)
    out = normalize_fact(fact)
    assert out.normalized_unit == "USD million"
    assert out.normalized_number == pytest.approx(5.0)


def test_usd_never_cross_converts_to_inr():
    fact = _fact(value_text="$5 million", value_number=5000000.0, unit="$")
    out = normalize_fact(fact)
    assert out.normalized_unit == "USD million"
    assert out.normalized_unit != "INR crore"


def test_unknown_currency_token_yields_none_plus_flag():
    fact = _fact(value_text="100", value_number=100.0, unit="JPY")
    out = normalize_fact(fact)
    assert out.normalized_number is None
    assert out.normalized_unit is None
    assert any(f.startswith("norm:") for f in out.ambiguity_flags)
    assert any("currency_unknown" in f for f in out.ambiguity_flags)


# ---------------------------------------------------------------------------
# 6. Unknown / ambiguous units; raw fields untouched
# ---------------------------------------------------------------------------


def test_unknown_unit_yields_none_plus_flag_with_raw_untouched():
    fact = _fact(value_text="3", value_number=3.0, unit="furlongs")
    snapshot = fact.model_dump()
    out = normalize_fact(fact)
    assert out.normalized_number is None
    assert out.normalized_unit is None
    assert any(f.startswith("norm:") for f in out.ambiguity_flags)
    assert any("unit_unknown" in f for f in out.ambiguity_flags)
    # Raw fields untouched: only reserved columns may change.
    for field in ("subject", "predicate", "value_text", "value_number", "unit"):
        assert getattr(out, field) == snapshot[field]


def test_bare_t_is_ambiguous_with_flag():
    fact = _fact(value_text="3", value_number=3.0, unit="t")
    out = normalize_fact(fact)
    assert out.normalized_number is None
    assert out.normalized_unit is None
    assert "norm:unit_ambiguous:t" in out.ambiguity_flags


# ---------------------------------------------------------------------------
# 7. Malformed numbers: kept, not dropped
# ---------------------------------------------------------------------------


def test_malformed_number_kept_not_dropped_with_flag():
    # Validator-owned malformed flow: build_fact marks value_unparseable.
    doc_id = uuid4()
    eid = uuid4()
    draft = FactDraft(
        subject="Synthetic subject",
        predicate="reported total",
        value_text="n/a",
        value_kind=ValueKind.NUMERIC,
        evidence_ids=[eid],
        confidence=0.8,
    )
    chunk = EvidenceChunk(
        document_id=doc_id,
        pdf_page_number=0,
        units=[ChunkUnit(evidence_id=eid, evidence_type="TEXT", text="synthetic")],
    )
    fact, reason = build_fact(draft, doc_id, chunk)
    assert reason is None and fact is not None
    assert fact.value_number is None
    assert "value_unparseable" in fact.ambiguity_flags

    out = normalize_fact(fact)
    assert out is not None  # kept, not dropped
    assert out.normalized_number is None
    assert out.normalized_unit is None
    assert "value_unparseable" in out.ambiguity_flags  # flag survives
    assert out.subject == fact.subject
    assert out.evidence_ids == fact.evidence_ids


def test_bare_none_number_is_clean_skip_without_norm_flag():
    # Implementation contract: value_number None with no prior flags is a
    # clean skip (no invented norm: flag), not ambiguity.
    fact = _fact(value_text="n/a", value_number=None, unit="units")
    out = normalize_fact(fact)
    assert out.normalized_number is None
    assert out.normalized_unit is None
    assert not any(f.startswith("norm:") for f in out.ambiguity_flags)


# ---------------------------------------------------------------------------
# 8. Fiscal periods: byte-identical, no dates invented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_text", "time_kind"),
    [("FY24", TimeKind.FISCAL_YEAR), ("Q4 FY25", TimeKind.QUARTER)],
)
def test_fiscal_periods_pass_through_byte_identical(time_text, time_kind):
    fact = _fact(
        time_text=time_text,
        time_kind=time_kind,
        time_start=None,
        time_end=None,
    )
    out = normalize_fact(fact)
    assert out.time_text == time_text
    assert out.time_kind == time_kind
    assert out.time_start is None and out.time_end is None  # no dates invented


def test_explicit_iso_date_passes_through():
    fact = _fact(
        time_text="2024-05-17",
        time_kind=TimeKind.DATE,
        time_start=date(2024, 5, 17),
        time_end=date(2024, 5, 17),
    )
    out = normalize_fact(fact)
    assert out.time_text == "2024-05-17"
    assert out.time_kind == TimeKind.DATE
    assert out.time_start == date(2024, 5, 17)
    assert out.time_end == date(2024, 5, 17)


# ---------------------------------------------------------------------------
# 9. Estimate / actual context identical before/after
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", list(EstimateStatus))
def test_estimate_status_geography_scope_context_identical(status):
    fact = _fact(
        estimate_status=status,
        geography="Synthetic region",
        scope_text="synthetic scope",
        context={"basis": "synthetic"},
    )
    out = normalize_fact(fact)
    assert out.estimate_status == status
    assert out.geography == "Synthetic region"
    assert out.scope_text == "synthetic scope"
    assert out.context == {"basis": "synthetic"}


# ---------------------------------------------------------------------------
# 10. Missing units: (number, None), no flag
# ---------------------------------------------------------------------------


def test_missing_unit_numeric_yields_number_none_unit_no_flag():
    fact = _fact(value_text="740", value_number=740.0, unit=None)
    out = normalize_fact(fact)
    assert out.normalized_number == pytest.approx(740.0)
    assert out.normalized_unit is None
    assert not any(f.startswith("norm:") for f in out.ambiguity_flags)


# ---------------------------------------------------------------------------
# 11. Preservation of raw fields, evidence, confidence, status
# ---------------------------------------------------------------------------


def test_raw_fields_evidence_confidence_status_preserved():
    eid = uuid4()
    fact = _fact(
        subject="Synthetic subject",
        predicate="reported total",
        value_text="Rs 100",
        value_number=100.0,
        unit="Rs",
        scope_text="synthetic scope",
        estimate_status=EstimateStatus.ACTUAL,
        geography="Synthetic region",
        context={"k": "v"},
        time_text="FY24",
        time_kind=TimeKind.FISCAL_YEAR,
        evidence_ids=[eid],
        extraction_confidence=0.73,
        status=FactStatus.AMBIGUOUS,
        ambiguity_flags=["seed-flag"],
    )
    out = normalize_fact(fact)
    assert out.subject == "Synthetic subject"
    assert out.predicate == "reported total"
    assert out.value_text == "Rs 100"
    assert out.value_number == pytest.approx(100.0)
    assert out.unit == "Rs"
    assert out.time_text == "FY24"
    assert out.scope_text == "synthetic scope"
    assert out.context == {"k": "v"}
    assert out.evidence_ids == [eid]
    assert out.extraction_confidence == pytest.approx(0.73)
    assert out.status == FactStatus.AMBIGUOUS
    # Append-only flags: seed preserved.
    assert "seed-flag" in out.ambiguity_flags


# ---------------------------------------------------------------------------
# 12. Immutability: input model_dump identical before/after
# ---------------------------------------------------------------------------


def test_normalize_fact_does_not_mutate_input():
    fact = _fact(value_text="Rs 100", value_number=100.0, unit="Rs")
    before = fact.model_dump()
    normalize_fact(fact)
    assert fact.model_dump() == before


# ---------------------------------------------------------------------------
# 13. No fabrication: every None case carries a flag OR is a clean TEXT skip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expectation"),
    [
        (
            {"value_text": "100", "value_number": 100.0, "unit": "JPY"},
            "flag",
        ),
        (
            {"value_text": "3", "value_number": 3.0, "unit": "furlongs"},
            "flag",
        ),
        (
            {"value_text": "3", "value_number": 3.0, "unit": "t"},
            "flag",
        ),
        (
            {
                "value_text": "steady",
                "value_number": None,
                "value_kind": ValueKind.TEXT,
                "unit": "units",
            },
            "text-skip",
        ),
    ],
)
def test_none_normalized_cases_flagged_or_clean_text_skip(kwargs, expectation):
    out = normalize_fact(_fact(**kwargs))
    assert out.normalized_number is None
    assert out.normalized_unit is None
    if expectation == "flag":
        assert any(f.startswith("norm:") for f in out.ambiguity_flags)
    else:
        # Clean TEXT-kind skip: None with NO flags (assert which).
        assert out.ambiguity_flags == []


# ---------------------------------------------------------------------------
# 14. Endpoint: POST /documents/{id}/normalize + 404 on unknown id
# ---------------------------------------------------------------------------


def test_normalize_endpoint_persists_reserved_columns(api_client, db_session):
    document, _, evidence = _ingest(
        db_session,
        "synthetic-normalize.pdf",
        _make_text_pdf(["Synthetic probe sentence with neutral wording. " * 25]),
    )
    eid = evidence[0].id
    ok_fact = _fact(
        document_id=document.id,
        value_text="81,415",
        value_number=81415.0,
        unit="units",
        evidence_ids=[eid],
    )
    flagged_fact = _fact(
        document_id=document.id,
        value_text="3",
        value_number=3.0,
        unit="furlongs",
        evidence_ids=[eid],
    )
    save_facts(db_session, [ok_fact, flagged_fact])
    db_session.commit()

    resp = api_client.post(f"/documents/{document.id}/normalize")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == str(document.id)
    assert body["facts_processed"] == 2
    assert body["facts_normalized"] >= 2  # canonical names count as normalized
    assert body["facts_flagged"] >= 1
    assert isinstance(body["errors"], list)

    got = api_client.get(f"/documents/{document.id}/facts")
    assert got.status_code == 200
    items = {row["id"]: row for row in got.json()}
    assert items[str(ok_fact.id)]["normalized_number"] == pytest.approx(81415.0)
    assert items[str(ok_fact.id)]["normalized_unit"] == "units"
    assert items[str(flagged_fact.id)]["normalized_number"] is None
    assert any(
        f.startswith("norm:") for f in items[str(flagged_fact.id)]["ambiguity_flags"]
    )

    stored = list_facts_for_document(db_session, document.id)
    assert len(stored) == 2


def test_normalize_endpoint_404_on_unknown_document(api_client):
    missing = uuid.uuid4()
    assert api_client.post(f"/documents/{missing}/normalize").status_code == 404


def test_normalize_document_facts_report_shape(db_session):
    document, _, evidence = _ingest(
        db_session,
        "synthetic-report.pdf",
        _make_text_pdf(["Synthetic probe sentence with neutral wording. " * 25]),
    )
    fact = _fact(document_id=document.id, evidence_ids=[evidence[0].id])
    save_facts(db_session, [fact])
    db_session.commit()

    report = normalize_document_facts(db_session, str(document.id))
    assert isinstance(report, NormalizationReport)
    assert report.document_id == str(document.id)
    assert report.facts_processed == 1
    assert report.facts_normalized >= 1
    assert report.errors == []
