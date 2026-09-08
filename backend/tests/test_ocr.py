"""T-O5 unit tests: OCR fallback in the ingestion pipeline.

All PDFs are generated in-memory with PyMuPDF; no binary fixtures.
PaddleOCR/paddle are NEVER imported here (not installed in this env):
``get_ocr_provider`` is monkeypatched with duck-type fakes, plus one test
asserting the REAL factory returns None without raising.
"""

import pymupdf
import pytest

import app.extraction.pipeline as pipeline
from app.extraction.ocr_provider import (
    NullProvider,
    OcrBlock,
    OcrError,
    OcrInitError,
    OcrMalformedError,
    OcrModelMissingError,
    OcrResult,
    OcrRuntimeError,
    OcrUnavailableError,
)
from app.extraction.quality import needs_ocr
from app.models.document import IngestionStatus
from app.models.evidence import EvidenceType, ExtractionMethod
from app.models.page import QualityVerdict

SPARSE_TEXT = "Sparse note."


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


def _make_good_pdf_bytes(n_pages=1):
    """Dense multi-line pages that score GOOD (single insert_text clips)."""
    line = "The quick brown fox jumps over the lazy dog. " * 4
    doc = pymupdf.open()
    try:
        for _ in range(n_pages):
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), line * 20)
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _merge_pdf_bytes(docs):
    """Concatenate single/multi-page PDFs (keeps GOOD-page builder usable)."""
    out = pymupdf.open()
    try:
        for data in docs:
            with pymupdf.open(stream=data, filetype="pdf") as src:
                out.insert_pdf(src)
        return bytes(out.tobytes())
    finally:
        out.close()


class _FakeOcrProvider:
    """Duck-type OcrProvider fake (OcrProvider is a runtime-checkable Protocol)."""

    def __init__(self, blocks=None, error=None, provider="paddleocr"):
        self.blocks = list(blocks) if blocks else []
        self.error = error
        self.provider = provider
        self.calls = []

    def extract_page(self, pdf_bytes, pdf_page_number):
        self.calls.append((pdf_bytes, pdf_page_number))
        if self.error is not None:
            raise self.error
        return OcrResult(blocks=list(self.blocks), provider=self.provider)


def _text_units(evidence):
    return [u for u in evidence if u.type == EvidenceType.TEXT]


def _ocr_units(evidence):
    return [u for u in evidence if u.type == EvidenceType.OCR_TEXT]


# 1. Routing: BAD triggers, GOOD does not -----------------------------------

def test_bad_page_triggers_ocr(monkeypatch):
    data = _make_pdf_bytes([""])
    fake = _FakeOcrProvider(
        blocks=[OcrBlock(text="recovered", bbox=(10, 10, 100, 50), confidence=0.9)]
    )
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: fake)
    document, pages, evidence = pipeline.run_ingestion("bad.pdf", data)
    assert pages[0].quality_verdict == QualityVerdict.BAD
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == data
    assert fake.calls[0][1] == 0
    assert _ocr_units(evidence), "BAD page should yield OCR_TEXT"
    assert document.ingestion_status == IngestionStatus.COMPLETED


def test_good_page_never_triggers_ocr(monkeypatch):
    data = _make_good_pdf_bytes(1)
    fake = _FakeOcrProvider(blocks=[OcrBlock(text="should not appear")])
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: fake)
    document, pages, evidence = pipeline.run_ingestion("good.pdf", data)
    assert pages[0].quality_verdict == QualityVerdict.GOOD
    assert fake.calls == []
    assert _ocr_units(evidence) == []
    assert _text_units(evidence), "GOOD page keeps native TEXT"


# 2. OCR unavailable: native evidence identical to pre-OCR -------------------

def test_ocr_unavailable_native_evidence_preserved(monkeypatch):
    sparse = _make_pdf_bytes([SPARSE_TEXT])
    good = _make_good_pdf_bytes(1)
    data = _merge_pdf_bytes([sparse, good])
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: None)
    document, pages, evidence = pipeline.run_ingestion("nounavailable.pdf", data)
    assert _ocr_units(evidence) == []
    assert _text_units(evidence), "native TEXT must be present without OCR"
    assert document.ingestion_status == IngestionStatus.COMPLETED
    assert all(p.error is None for p in pages)


# 3. OCR success: additive OCR_TEXT ------------------------------------------

def test_ocr_success_appends_ocr_text(monkeypatch):
    data = _make_pdf_bytes([SPARSE_TEXT])
    bbox = (10.0, 10.0, 100.0, 50.0)
    fake = _FakeOcrProvider(
        blocks=[
            OcrBlock(text="ocr recovered", bbox=bbox, confidence=0.87),
            OcrBlock(text="no confidence block", bbox=bbox, confidence=None),
        ]
    )
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: fake)
    document, pages, evidence = pipeline.run_ingestion("sparse.pdf", data)
    assert pages[0].quality_verdict == QualityVerdict.BAD
    native = [u for u in _text_units(evidence) if u.pdf_page_number == 0]
    assert native, "native TEXT still present on a sparse-text BAD page"
    ocr = [u for u in _ocr_units(evidence) if u.pdf_page_number == 0]
    assert len(ocr) == 2
    first = ocr[0]
    assert first.extraction_method == ExtractionMethod.PADDLEOCR
    assert first.bbox == pytest.approx(bbox)
    assert first.extraction_quality == pytest.approx(0.87)
    assert first.meta["provider"] == "paddleocr"
    assert first.meta["trigger"] == "BAD verdict"
    assert first.meta["retries"] == 0
    assert set(first.meta) >= {"provider", "trigger", "retries"}
    # confidence None falls back to the page score.
    assert ocr[1].extraction_quality == pytest.approx(pages[0].extraction_quality)
    assert document.ingestion_status == IngestionStatus.COMPLETED


# 4. OCR errors never fail the document --------------------------------------

@pytest.mark.parametrize(
    ("error", "category"),
    [
        (OcrUnavailableError("no engine"), "unavailable"),
        (OcrInitError("bad init"), "init"),
        (OcrModelMissingError("no weights"), "models"),
        (OcrRuntimeError("crashed"), "runtime"),
        (OcrMalformedError("garbage"), "malformed"),
    ],
)
def test_ocr_errors_never_fail_document(monkeypatch, error, category):
    assert isinstance(error, OcrError)
    data = _make_pdf_bytes([SPARSE_TEXT])
    fake = _FakeOcrProvider(error=error)
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: fake)
    document, pages, evidence = pipeline.run_ingestion("ocrerror.pdf", data)
    assert document.ingestion_status == IngestionStatus.COMPLETED
    assert document.ingestion_status != IngestionStatus.FAILED
    assert _text_units(evidence), "native TEXT preserved despite OCR error"
    assert _ocr_units(evidence) == []
    assert pages[0].error is not None
    assert f"ocr:{category}:" in pages[0].error


# 5. needs_ocr predicate ------------------------------------------------------

def test_needs_ocr_routing():
    assert needs_ocr(QualityVerdict.BAD) is True
    assert needs_ocr(QualityVerdict.GOOD) is False


# 6. OCR evidence IDs differ from TEXT IDs ------------------------------------

def test_ocr_evidence_ids_differ_from_text_ids(monkeypatch):
    data = _make_pdf_bytes([SPARSE_TEXT])
    fake = _FakeOcrProvider(blocks=[OcrBlock(text="recovered", confidence=0.9)])
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: fake)
    _, _, evidence = pipeline.run_ingestion("ids.pdf", data)
    text_ids = {u.id for u in _text_units(evidence) if u.pdf_page_number == 0}
    ocr_ids = {u.id for u in _ocr_units(evidence) if u.pdf_page_number == 0}
    assert text_ids and ocr_ids
    assert not (text_ids & ocr_ids)


# 7. Two BAD pages: independent per-page calls ---------------------------------

def test_two_bad_pages_trigger_per_page(monkeypatch):
    data = _make_pdf_bytes(["", ""])
    fake = _FakeOcrProvider(blocks=[OcrBlock(text="recovered", confidence=0.8)])
    monkeypatch.setattr(pipeline, "get_ocr_provider", lambda: fake)
    document, pages, evidence = pipeline.run_ingestion("twobad.pdf", data)
    assert [p.quality_verdict for p in pages] == [
        QualityVerdict.BAD,
        QualityVerdict.BAD,
    ]
    assert [call[1] for call in fake.calls] == [0, 1]
    assert all(call[0] == data for call in fake.calls)
    by_page = {}
    for unit in _ocr_units(evidence):
        by_page.setdefault(unit.pdf_page_number, []).append(unit)
    assert set(by_page) == {0, 1}
    assert all(by_page[n] for n in (0, 1))
    assert document.ingestion_status == IngestionStatus.COMPLETED


# 8. Real factory: None here (no paddle/models), never raises ------------------

def test_real_get_ocr_provider_returns_none_without_paddle():
    assert pipeline.get_ocr_provider() is None


# 9. NullProvider ---------------------------------------------------------------

def test_null_provider_raises_unavailable():
    with pytest.raises(OcrUnavailableError):
        NullProvider().extract_page(b"%PDF-1.4 fake", 0)
