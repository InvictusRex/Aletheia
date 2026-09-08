"""T7 unit tests: ingestion pipeline + SQLite repository round-trip.

All PDFs are generated in-memory with PyMuPDF; no binary fixtures.
DB tests use the file-backed SQLite ``db_session`` fixture.
"""

import uuid

import pymupdf

from app.db.repositories import get_document_bundle, save_ingestion
from app.extraction.pipeline import run_ingestion
from app.models.document import IngestionStatus


def _make_pdf_bytes(page_texts):
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return bytes(doc.tobytes())
    finally:
        doc.close()


DENSE_TEXT = "The quick brown fox jumps over the lazy dog. " * 30


def test_run_ingestion_invalid_bytes_failed_without_raise():
    document, pages, evidence = run_ingestion("broken.pdf", b"not a pdf at all")
    assert document.ingestion_status == IngestionStatus.FAILED
    assert pages == []
    assert evidence == []
    assert document.error, "FAILED ingestion must set an error message"


def test_run_ingestion_empty_bytes_failed_without_raise():
    document, pages, evidence = run_ingestion("empty.pdf", b"")
    assert document.ingestion_status == IngestionStatus.FAILED
    assert pages == []
    assert evidence == []
    assert document.error


def test_run_ingestion_unlabeled_source_page_number_none_everywhere():
    data = _make_pdf_bytes([DENSE_TEXT, DENSE_TEXT])
    document, pages, evidence = run_ingestion("plain.pdf", data)
    assert document.ingestion_status == IngestionStatus.COMPLETED
    assert len(pages) == 2
    assert all(p.source_page_number is None for p in pages)
    assert evidence, "dense text pages must yield evidence"
    assert all(e.source_page_number is None for e in evidence)


def test_run_ingestion_pdf_page_number_is_zero_based_physical_index():
    data = _make_pdf_bytes([DENSE_TEXT, DENSE_TEXT, DENSE_TEXT])
    _, pages, evidence = run_ingestion("three.pdf", data)
    assert [p.pdf_page_number for p in pages] == [0, 1, 2]
    assert {e.pdf_page_number for e in evidence} <= {0, 1, 2}


def test_run_ingestion_success_completed_with_evidence_provenance():
    data = _make_pdf_bytes([DENSE_TEXT])
    document, pages, evidence = run_ingestion("good.pdf", data)
    assert document.ingestion_status == IngestionStatus.COMPLETED
    assert document.page_count == 1
    assert document.error is None
    assert evidence, "expected text evidence for a dense page"
    for unit in evidence:
        assert unit.document_id == document.id
        assert unit.bbox is not None
        assert len(unit.bbox) == 4
        assert unit.text.strip()


def test_save_get_round_trip_preserves_bundle_and_provenance(db_session):
    data = _make_pdf_bytes([DENSE_TEXT, DENSE_TEXT])
    document, pages, evidence = run_ingestion("roundtrip.pdf", data, source="unit-test")
    assert evidence, "precondition: ingestion must produce evidence"

    save_ingestion(db_session, document, pages, evidence)
    db_session.commit()

    bundle = get_document_bundle(db_session, document.id)
    assert bundle is not None
    got_doc, got_pages, got_evidence = bundle

    assert got_doc.id == document.id
    assert got_doc.filename == document.filename
    assert got_doc.page_count == document.page_count
    assert got_doc.file_sha256 == document.file_sha256
    assert got_doc.ingestion_status == document.ingestion_status
    assert got_doc.source == "unit-test"

    assert len(got_pages) == len(pages)
    for want, got in zip(pages, got_pages):
        assert got.document_id == document.id
        assert got.pdf_page_number == want.pdf_page_number
        assert got.source_page_number == want.source_page_number
        assert got.char_count == want.char_count
        assert got.word_count == want.word_count

    assert len(got_evidence) == len(evidence)
    want_by_id = {e.id: e for e in evidence}
    for got in got_evidence:
        want = want_by_id[got.id]
        assert got.document_id == document.id
        assert got.pdf_page_number == want.pdf_page_number
        assert got.source_page_number == want.source_page_number
        assert got.text == want.text
        assert got.extraction_method == want.extraction_method
        if want.bbox is None:
            assert got.bbox is None
        else:
            assert got.bbox is not None
            assert tuple(got.bbox) == tuple(want.bbox)


def test_get_document_bundle_unknown_id_returns_none(db_session):
    assert get_document_bundle(db_session, uuid.uuid4()) is None
