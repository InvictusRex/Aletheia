"""F-TST: fact-layer tests (schema, validator, service, API, provider config).

Conventions:
- All PDFs are generated in-memory with PyMuPDF; ingestion goes through
  ``run_ingestion`` + ``save_ingestion`` into the shared ``db_session``
  fixture (file-backed SQLite, full schema).
- LLM access uses duck-typed fake providers only (no network, no API
  keys). Fakes satisfy the ``LLMProvider`` Protocol structurally
  (``model_name`` + ``extract_facts(prompt)``).
- Synthetic wording is neutral; no production dataset references.
"""

from __future__ import annotations

import re
import uuid
from datetime import date
from uuid import UUID, uuid4

import pymupdf
import pytest
from pydantic import ValidationError

import app.api.facts as facts_api
from app.core.config import settings
from app.db.repositories import (
    get_chunk_statuses,
    get_document_bundle,
    get_fact,
    list_facts_for_document,
    save_facts,
    save_ingestion,
)
from app.extraction.pipeline import run_ingestion
from app.facts.service import FactExtractionReport, extract_facts_for_document
from app.facts.validator import build_fact, parse_number, parse_time, resolve_kind
from app.llm.groq import DEFAULT_GROQ_MODEL, GroqFactsProvider
from app.llm.provider import FactExtractionBatch, LLMProvider, LLMTransportError
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
from app.models.evidence import EvidenceType

SYNTHETIC_TEXT = "Synthetic probe sentence with neutral wording. " * 25

_UNIT_TAG_RE = re.compile(
    r"\[([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\] \((\w+)\)"
)


# ---------------------------------------------------------------------------
# Helpers: PDFs, ingestion, chunks, drafts, fakes
# ---------------------------------------------------------------------------


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


def _draw_grid(page, headers, rows, origin=(72, 72), col_width=120, row_h=28):
    """Ruled grid with text inside each cell (same pattern as test_tables)."""
    x0, y0 = origin
    ncols = len(headers)
    nrows = len(rows) + 1
    x1 = x0 + col_width * ncols
    y1 = y0 + row_h * nrows
    for r in range(nrows + 1):
        y = y0 + r * row_h
        page.draw_line((x0, y), (x1, y))
    for c in range(ncols + 1):
        x = x0 + c * col_width
        page.draw_line((x, y0), (x, y1))
    for r, row in enumerate([headers] + rows):
        for c, value in enumerate(row):
            page.insert_text(
                (x0 + c * col_width + 4, y0 + r * row_h + 18),
                value,
                fontsize=10,
            )


def _make_table_pdf(headers, rows) -> bytes:
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        _draw_grid(page, headers, rows)
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


def _chunk(document_id: UUID, *evidence_ids: UUID) -> EvidenceChunk:
    return EvidenceChunk(
        document_id=document_id,
        pdf_page_number=0,
        units=[
            ChunkUnit(evidence_id=eid, evidence_type="TEXT", text="synthetic unit")
            for eid in evidence_ids
        ],
    )


def _draft(evidence_id: UUID, **overrides) -> FactDraft:
    kwargs = {
        "subject": "Synthetic subject",
        "predicate": "reported total",
        "value_text": "740",
        "value_kind": ValueKind.NUMERIC,
        "time_text": "2024-05-17",
        "evidence_ids": [evidence_id],
        "confidence": 0.8,
    }
    kwargs.update(overrides)
    return FactDraft(**kwargs)


def _fact_kwargs(document_id: UUID, evidence_id: UUID) -> dict:
    return {
        "document_id": document_id,
        "subject": "Synthetic subject",
        "predicate": "reported total",
        "value_kind": ValueKind.NUMERIC,
        "value_text": "740",
        "value_number": 740.0,
        "evidence_ids": [evidence_id],
        "extraction_confidence": 0.8,
    }


class FakeFactsProvider:
    """Duck-typed ``LLMProvider``: cites evidence IDs parsed from the prompt.

    The service renders each chunk unit as ``[<evidence_id>] (<TYPE>) ...``,
    so the fake cites IDs that always validate against the current chunk —
    across pages, retries, and table cells — without any network access.
    """

    def __init__(
        self,
        *,
        model: str = "fake-test-model",
        value_text: str = "740",
        value_kind: ValueKind | None = ValueKind.NUMERIC,
        time_text: str | None = None,
        prefer_type: str | None = None,
        extra_unknown_draft: bool = False,
        fail_on_calls: frozenset[int] = frozenset(),
    ) -> None:
        self.model_name = model
        self.value_text = value_text
        self.value_kind = value_kind
        self.time_text = time_text
        self.prefer_type = prefer_type
        self.extra_unknown_draft = extra_unknown_draft
        self.fail_on_calls = fail_on_calls
        self.calls = 0
        self.prompts: list[str] = []

    def extract_facts(self, prompt: str) -> FactExtractionBatch:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls in self.fail_on_calls:
            raise LLMTransportError("synthetic transport boom")
        matches = _UNIT_TAG_RE.findall(prompt)
        cited: UUID | None = None
        if self.prefer_type is not None:
            for raw_id, kind in matches:
                if kind == self.prefer_type:
                    cited = UUID(raw_id)
                    break
        if cited is None and matches:
            cited = UUID(matches[0][0])
        drafts: list[FactDraft] = []
        if cited is not None:
            drafts.append(
                FactDraft(
                    subject="Synthetic subject",
                    predicate="reported total",
                    value_text=self.value_text,
                    value_kind=self.value_kind,
                    time_text=self.time_text,
                    evidence_ids=[cited],
                    confidence=0.9,
                )
            )
        if self.extra_unknown_draft:
            drafts.append(
                FactDraft(
                    subject="Synthetic subject",
                    predicate="reported total",
                    value_text="12",
                    value_kind=ValueKind.NUMERIC,
                    evidence_ids=[uuid4()],  # never in any chunk -> rejected
                    confidence=0.5,
                )
            )
        return FactExtractionBatch(drafts=drafts, model=self.model_name)


# ---------------------------------------------------------------------------
# 1. Schema: evidence_ids required, confidence bounds enforced
# ---------------------------------------------------------------------------


def test_fact_requires_nonempty_evidence_ids():
    with pytest.raises(ValidationError):
        Fact(**(_fact_kwargs(uuid4(), uuid4()) | {"evidence_ids": []}))


def test_fact_confidence_bounds_enforced():
    good = _fact_kwargs(uuid4(), uuid4())
    Fact(**good)  # sanity: baseline valid
    with pytest.raises(ValidationError):
        Fact(**(good | {"extraction_confidence": 1.5}))
    with pytest.raises(ValidationError):
        Fact(**(good | {"extraction_confidence": -0.1}))
    Fact(**(good | {"extraction_confidence": 0.0}))
    Fact(**(good | {"extraction_confidence": 1.0}))


def test_fact_draft_confidence_bounds_enforced():
    eid = uuid4()
    with pytest.raises(ValidationError):
        _draft(eid, confidence=1.5)
    with pytest.raises(ValidationError):
        _draft(eid, confidence=-0.2)


# ---------------------------------------------------------------------------
# 2. parse_number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("81,415", 81415.0),
        ("1,429K", 1429000.0),
        ("1.6%", 1.6),
        ("(1,234)", -1234.0),
        ("n/a", None),
        ("", None),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


# ---------------------------------------------------------------------------
# 3. parse_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "start", "end"),
    [
        ("FY24", TimeKind.FISCAL_YEAR, None, None),
        ("Q4 FY24", TimeKind.QUARTER, None, None),
        ("2024-05-17", TimeKind.DATE, date(2024, 5, 17), date(2024, 5, 17)),
        ("May 2024", TimeKind.DATE, date(2024, 5, 1), date(2024, 5, 31)),
        ("sometime-ish", TimeKind.UNKNOWN, None, None),
        ("", TimeKind.UNKNOWN, None, None),
        (None, TimeKind.UNKNOWN, None, None),
    ],
)
def test_parse_time(text, kind, start, end):
    assert parse_time(text) == (kind, start, end)


# ---------------------------------------------------------------------------
# 4. build_fact: provenance enforcement, status, no fabrication
# ---------------------------------------------------------------------------


def test_build_fact_unknown_evidence_id_rejected():
    doc_id = uuid4()
    known = uuid4()
    chunk = _chunk(doc_id, known)
    draft = _draft(uuid4())  # evidence ID the chunk never supplied
    fact, reason = build_fact(draft, doc_id, chunk)
    assert fact is None
    assert reason is not None and "unknown" in reason.lower()


def test_build_fact_clean_draft_confirmed():
    doc_id = uuid4()
    eid = uuid4()
    fact, reason = build_fact(_draft(eid), doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.status == FactStatus.CONFIRMED
    assert fact.ambiguity_flags == []
    assert fact.value_number == 740.0
    assert fact.value_kind == ValueKind.NUMERIC
    assert fact.time_kind == TimeKind.DATE
    assert fact.time_start == date(2024, 5, 17)
    assert fact.extraction_confidence == 0.8
    assert fact.evidence_ids == [eid]


def test_build_fact_ambiguous_draft_flagged():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, ambiguous=True, notes="two readings possible")
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.status == FactStatus.AMBIGUOUS
    assert "llm_flagged_ambiguous" in fact.ambiguity_flags


def test_build_fact_unparseable_value_never_fabricated():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, value_text="n/a", value_kind=ValueKind.NUMERIC)
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.status == FactStatus.AMBIGUOUS
    assert fact.value_number is None  # never guessed
    assert "value_unparseable" in fact.ambiguity_flags


def test_build_fact_garbage_time_flagged_not_invented():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, time_text="sometime-ish")
    fact, reason = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert reason is None
    assert fact is not None
    assert fact.time_kind == TimeKind.UNKNOWN
    assert fact.time_start is None and fact.time_end is None
    assert "time_unparsed" in fact.ambiguity_flags
    assert fact.status == FactStatus.AMBIGUOUS


# ---------------------------------------------------------------------------
# 5. Value-kind resolution: percentages vs counts
# ---------------------------------------------------------------------------


def test_percentage_suffix_overrides_numeric_draft_with_flag():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(
        eid, value_text="1.6%", value_kind=ValueKind.NUMERIC, time_text=None
    )
    kind, flags = resolve_kind(draft)
    assert kind == ValueKind.PERCENTAGE
    assert "kind_corrected:NUMERIC->PERCENTAGE" in flags
    fact, _ = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert fact is not None
    assert fact.value_kind == ValueKind.PERCENTAGE
    assert fact.value_number == 1.6  # percent-kind numeric flow, no scaling


def test_plain_count_resolves_numeric():
    doc_id = uuid4()
    eid = uuid4()
    draft = _draft(eid, value_text="740", value_kind=None, time_text=None)
    kind, flags = resolve_kind(draft)
    assert kind == ValueKind.NUMERIC
    assert flags == []
    fact, _ = build_fact(draft, doc_id, _chunk(doc_id, eid))
    assert fact is not None
    assert fact.value_kind == ValueKind.NUMERIC
    assert fact.value_number == 740.0


# ---------------------------------------------------------------------------
# 6. Service with fake provider over a real ingested PDF
# ---------------------------------------------------------------------------


def test_service_persists_and_retrieves_facts_with_report_counts(db_session):
    document, _, evidence = _ingest(
        db_session, "synthetic.pdf", _make_text_pdf([SYNTHETIC_TEXT])
    )
    provider = FakeFactsProvider(extra_unknown_draft=True)
    report = extract_facts_for_document(
        db_session, document.id, provider=provider
    )
    db_session.commit()

    assert report.chunks_processed == 1
    assert report.chunks_failed == 0
    assert report.drafts_rejected == 1  # the unknown-evidence draft
    assert len(report.facts) == 1
    assert provider.calls == 1
    assert provider.prompts and all(p.strip() for p in provider.prompts)

    stored = list_facts_for_document(db_session, document.id)
    assert len(stored) == 1
    assert set(stored[0].evidence_ids) == set(report.facts[0].evidence_ids)
    assert stored[0].evidence_ids, "evidence_ids must survive the round-trip"
    assert stored[0].subject == report.facts[0].subject
    assert stored[0].value_number == report.facts[0].value_number
    assert get_fact(db_session, report.facts[0].id) is not None


# ---------------------------------------------------------------------------
# 7. Transport failure mid-run: partial progress, errors, bundle untouched
# ---------------------------------------------------------------------------


def test_service_transport_error_mid_run_is_contained(db_session):
    document, _, _ = _ingest(
        db_session,
        "two-page.pdf",
        _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT]),
    )
    before = get_document_bundle(db_session, document.id)
    assert before is not None
    _, before_pages, before_evidence = before

    provider = FakeFactsProvider(fail_on_calls=frozenset({2}))
    report = extract_facts_for_document(  # must not raise
        db_session, document.id, provider=provider
    )
    db_session.commit()

    assert provider.calls == 2
    assert report.chunks_processed == 1
    assert report.chunks_failed == 1
    assert len(report.facts) == 1
    assert any("LLMTransportError" in e for e in report.errors)

    after = get_document_bundle(db_session, document.id)
    assert after is not None
    _, after_pages, after_evidence = after
    assert len(after_pages) == len(before_pages) == 2
    assert {e.id for e in after_evidence} == {e.id for e in before_evidence}
    assert len(list_facts_for_document(db_session, document.id)) == 1


# ---------------------------------------------------------------------------
# 8. Table evidence: fact links the exact TABLE_CELL ID
# ---------------------------------------------------------------------------


def test_service_fact_links_exact_table_cell_id(db_session):
    pdf = _make_table_pdf(
        ["Item", "Period A"], [["Alpha", "740"], ["Beta", "120"]]
    )
    document, _, evidence = _ingest(db_session, "synthetic-table.pdf", pdf)
    cell_ids = {e.id for e in evidence if e.type == EvidenceType.TABLE_CELL}
    assert cell_ids, "precondition: ingestion must yield TABLE_CELL evidence"

    provider = FakeFactsProvider(
        prefer_type=EvidenceType.TABLE_CELL.value, value_text="740"
    )
    report = extract_facts_for_document(
        db_session, document.id, provider=provider
    )
    db_session.commit()

    assert len(report.facts) == 1
    (fact,) = report.facts
    assert len(fact.evidence_ids) == 1
    assert fact.evidence_ids[0] in cell_ids

    stored = list_facts_for_document(db_session, document.id)
    assert len(stored) == 1
    assert stored[0].evidence_ids == fact.evidence_ids
    by_id = {e.id: e for e in evidence}
    assert by_id[stored[0].evidence_ids[0]].type == EvidenceType.TABLE_CELL


# ---------------------------------------------------------------------------
# 9. API: 503 without key, stubbed extraction, 404s
# ---------------------------------------------------------------------------


def test_api_post_facts_503_without_key(api_client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "")
    assert settings.llm_provider == "groq"
    pdf = _make_text_pdf([SYNTHETIC_TEXT])
    document, _, _ = _ingest(db_session, "api-503.pdf", pdf)

    resp = api_client.post(f"/documents/{document.id}/facts")
    assert resp.status_code == 503


def test_api_post_stubbed_extraction_then_get_persisted(
    api_client, db_session, monkeypatch
):
    pdf = _make_text_pdf([SYNTHETIC_TEXT])
    document, _, evidence = _ingest(db_session, "api-stub.pdf", pdf)

    wanted = Fact(
        **_fact_kwargs(document.id, evidence[0].id),
    )

    def _stub(db, document_id, provider=None, **kwargs):
        save_facts(db, [wanted])
        return FactExtractionReport(
            document_id=str(document_id),
            facts=[wanted],
            chunks_processed=1,
            chunks_failed=0,
            drafts_rejected=0,
            errors=[],
        )

    monkeypatch.setattr(facts_api, "extract_facts_for_document", _stub)

    post = api_client.post(f"/documents/{document.id}/facts")
    assert post.status_code == 200
    body = post.json()
    assert body["document_id"] == str(document.id)
    assert body["chunks_processed"] == 1
    assert len(body["facts"]) == 1
    assert body["facts"][0]["subject"] == wanted.subject
    assert body["facts"][0]["evidence_ids"] == [str(evidence[0].id)]

    got = api_client.get(f"/documents/{document.id}/facts")
    assert got.status_code == 200
    items = got.json()
    assert len(items) == 1
    assert items[0]["id"] == str(wanted.id)
    assert items[0]["evidence_ids"] == [str(evidence[0].id)]


def test_api_facts_404_on_unknown_document(api_client):
    missing = uuid.uuid4()
    assert api_client.get(f"/documents/{missing}/facts").status_code == 404
    assert api_client.post(f"/documents/{missing}/facts").status_code == 404


# ---------------------------------------------------------------------------
# 10. Config-driven model (construct only — never call the real provider)
# ---------------------------------------------------------------------------


def test_groq_default_model_name():
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"
    provider = GroqFactsProvider(api_key="construct-only-key")
    assert provider.model_name == "openai/gpt-oss-120b"
    assert isinstance(provider, LLMProvider)


def test_groq_custom_model_reflected_and_carried_by_batches():
    provider = GroqFactsProvider(
        api_key="construct-only-key", model="custom-model-x"
    )
    assert provider.model_name == "custom-model-x"
    batch = FactExtractionBatch(drafts=[], model=provider.model_name)
    assert batch.model == "custom-model-x"
    # No extract_facts call on the real class anywhere in this module:
    # construction only, so no network access is possible.


# ---------------------------------------------------------------------------
# 11. Per-chunk persistence + resume (duck-typed fakes only, no network)
# ---------------------------------------------------------------------------


def test_per_chunk_progress_rows_persisted(db_session):
    document, _, evidence = _ingest(
        db_session,
        "two-page.pdf",
        _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT]),
    )
    provider = FakeFactsProvider()
    report = extract_facts_for_document(
        db_session, document.id, provider=provider
    )
    db_session.commit()

    assert report.chunks_processed == 2
    assert report.chunks_failed == 0
    assert len(report.facts) == 2
    statuses = get_chunk_statuses(db_session, document.id)
    assert len(statuses) == 2
    assert {(p, i) for p, i in statuses} == {(0, 0), (1, 0)}
    for row in statuses.values():
        assert row.status == "COMPLETED"
        assert row.error is None
        assert len(row.fact_ids) == 1
        assert get_fact(db_session, UUID(str(row.fact_ids[0]))) is not None


def test_failed_chunk_preserves_prior_success(db_session):
    document, _, _ = _ingest(
        db_session,
        "two-page.pdf",
        _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT]),
    )
    provider = FakeFactsProvider(fail_on_calls=frozenset({2}))
    report = extract_facts_for_document(
        db_session, document.id, provider=provider
    )
    db_session.commit()

    assert report.chunks_processed == 1
    assert report.chunks_failed == 1
    assert len(report.facts) == 1
    statuses = get_chunk_statuses(db_session, document.id)
    assert statuses[(0, 0)].status == "COMPLETED"
    assert statuses[(1, 0)].status == "FAILED"
    assert statuses[(1, 0)].error
    # The successful chunk's fact survived the later failure.
    assert len(list_facts_for_document(db_session, document.id)) == 1


def test_rerun_skips_completed_without_recalling_provider(db_session):
    document, _, _ = _ingest(
        db_session,
        "two-page.pdf",
        _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT]),
    )
    first = FakeFactsProvider()
    report1 = extract_facts_for_document(
        db_session, document.id, provider=first
    )
    db_session.commit()
    assert report1.chunks_processed == 2

    # Any provider call now would raise: resume must make zero calls.
    second = FakeFactsProvider(fail_on_calls=frozenset({1, 2, 3}))
    report2 = extract_facts_for_document(
        db_session, document.id, provider=second
    )
    db_session.commit()

    assert second.calls == 0
    assert report2.chunks_processed == 0
    assert report2.chunks_failed == 0
    assert report2.chunks_skipped == 2
    assert len(report2.facts) == 2  # rehydrated from progress rows
    assert len(list_facts_for_document(db_session, document.id)) == 2


def test_legacy_facts_dedupe_without_progress_rows(db_session):
    from app.db.models import ExtractionChunkRow

    document, _, _ = _ingest(
        db_session,
        "two-page.pdf",
        _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT]),
    )
    report1 = extract_facts_for_document(
        db_session, document.id, provider=FakeFactsProvider()
    )
    db_session.commit()
    assert len(report1.facts) == 2

    # Simulate pre-progress facts: drop progress rows, keep facts.
    db_session.query(ExtractionChunkRow).delete()
    db_session.commit()
    assert get_chunk_statuses(db_session, document.id) == {}

    report2 = extract_facts_for_document(
        db_session, document.id, provider=FakeFactsProvider()
    )
    db_session.commit()

    # Re-extraction ran (no rows to skip) but inserted no duplicates.
    assert report2.chunks_processed == 2
    assert report2.facts_skipped_duplicate == 2
    assert len(list_facts_for_document(db_session, document.id)) == 2

    # Progress rows were rebuilt with intact fact links: a third run
    # skips everything without calling the provider.
    third = FakeFactsProvider(fail_on_calls=frozenset({1, 2, 3}))
    report3 = extract_facts_for_document(
        db_session, document.id, provider=third
    )
    db_session.commit()
    assert third.calls == 0
    assert report3.chunks_skipped == 2
    assert len(report3.facts) == 2


# ---------------------------------------------------------------------------
# 12. Page-range scoping (service filter + API validation)
# ---------------------------------------------------------------------------


def test_page_range_limits_chunks_to_window(db_session):
    document, _, evidence = _ingest(
        db_session,
        "two-page.pdf",
        _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT]),
    )
    by_page = {}
    for unit in evidence:
        by_page.setdefault(unit.pdf_page_number, set()).add(unit.id)
    assert set(by_page) == {0, 1}

    provider = FakeFactsProvider()
    report = extract_facts_for_document(
        db_session, document.id, provider=provider, start_page=0, end_page=0
    )
    db_session.commit()

    assert provider.calls == 1
    assert report.chunks_processed == 1
    assert len(report.facts) == 1
    assert set(report.facts[0].evidence_ids) <= by_page[0]
    statuses = get_chunk_statuses(db_session, document.id)
    assert set(statuses) == {(0, 0)}  # page 1 still pending

    # A full rerun resumes: page 0 skipped, page 1 processed.
    full = FakeFactsProvider()
    report_full = extract_facts_for_document(
        db_session, document.id, provider=full
    )
    db_session.commit()
    assert full.calls == 1
    assert report_full.chunks_skipped == 1
    assert report_full.chunks_processed == 1
    assert len(list_facts_for_document(db_session, document.id)) == 2


def test_page_range_validation_rejects_nonsense(db_session):
    document, _, _ = _ingest(
        db_session, "synthetic.pdf", _make_text_pdf([SYNTHETIC_TEXT])
    )
    with pytest.raises(ValueError, match="must be <="):
        extract_facts_for_document(
            db_session, document.id, provider=FakeFactsProvider(),
            start_page=2, end_page=1,
        )
    with pytest.raises(ValueError, match=">= 0"):
        extract_facts_for_document(
            db_session, document.id, provider=FakeFactsProvider(),
            start_page=-1,
        )
    with pytest.raises(ValueError, match=">= 0"):
        extract_facts_for_document(
            db_session, document.id, provider=FakeFactsProvider(),
            end_page=-1,
        )


def test_api_page_range_400_on_inverted_window(api_client, db_session):
    document, _, _ = _ingest(
        db_session, "api-range.pdf", _make_text_pdf([SYNTHETIC_TEXT])
    )
    resp = api_client.post(
        f"/documents/{document.id}/facts?start_page=3&end_page=1"
    )
    assert resp.status_code == 400


def test_api_page_range_422_on_negative(api_client, db_session):
    document, _, _ = _ingest(
        db_session, "api-range.pdf", _make_text_pdf([SYNTHETIC_TEXT])
    )
    resp = api_client.post(
        f"/documents/{document.id}/facts?start_page=-1"
    )
    assert resp.status_code == 422


def test_api_page_range_scoped_extraction(api_client, db_session, monkeypatch):
    pdf = _make_text_pdf([SYNTHETIC_TEXT, SYNTHETIC_TEXT])
    document, _, evidence = _ingest(db_session, "api-scope.pdf", pdf)
    provider = FakeFactsProvider()
    # Route the API through the fake: patch the service reference used
    # by extract_facts_for_document when no provider is passed.
    import app.facts.service as facts_service

    monkeypatch.setattr(facts_service, "get_llm_provider", lambda: provider)

    resp = api_client.post(
        f"/documents/{document.id}/facts?start_page=1&end_page=1"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_processed"] == 1
    assert provider.calls == 1
    page1_ids = {e.id for e in evidence if e.pdf_page_number == 1}
    assert set(body["facts"][0]["evidence_ids"]) <= {
        str(i) for i in page1_ids
    }

    # Unscoped rerun resumes the remaining page only.
    resp2 = api_client.post(f"/documents/{document.id}/facts")
    assert resp2.status_code == 200
    assert resp2.json()["chunks_skipped"] == 1
    assert provider.calls == 2
    got = api_client.get(f"/documents/{document.id}/facts")
    assert len(got.json()) == 2
