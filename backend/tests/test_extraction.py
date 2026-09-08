"""T7 unit tests: PDF extraction, quality scoring, evidence IDs, page labels.

All PDFs are generated in-memory with PyMuPDF; no binary fixtures.
"""

import uuid

import pymupdf
import pytest

from app.extraction.pymupdf import extract_pdf
from app.extraction.quality import assess_quality
from app.models.evidence import EvidenceType, make_evidence_id
from app.models.page import QualityVerdict


def _make_pdf_bytes(page_texts, title=None, labels=None):
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        if title is not None:
            doc.set_metadata({"title": title})
        if labels is not None:
            doc.set_page_labels(labels)
        return bytes(doc.tobytes())
    finally:
        doc.close()


DENSE_TEXT = "The quick brown fox jumps over the lazy dog. " * 30


def test_extract_pdf_page_count_and_metadata():
    data = _make_pdf_bytes(
        ["First page content. " * 20, "Second page content. " * 20],
        title="Extraction Test Doc",
    )
    metadata, pages = extract_pdf(data)
    assert metadata.page_count == 2
    assert metadata.title == "Extraction Test Doc"
    assert len(pages) == 2


def test_extract_pdf_text_blocks_carry_text_and_bbox():
    data = _make_pdf_bytes(["Hello excavation world. " * 20])
    _, pages = extract_pdf(data)
    assert len(pages) == 1
    assert pages[0].blocks, "expected at least one text block"
    for block in pages[0].blocks:
        assert block.text.strip(), "text block must carry non-empty text"
        assert block.bbox is not None
        assert len(block.bbox) == 4
        x0, y0, x1, y1 = block.bbox
        assert all(isinstance(v, float) for v in block.bbox)
        assert x1 > x0
        assert y1 > y0
    joined = " ".join(b.text for b in pages[0].blocks)
    assert "Hello" in joined


@pytest.mark.parametrize("bad_input", [b"", b"not a pdf at all", b"\x00\x01\x02\x03"])
def test_extract_pdf_invalid_input_raises_valueerror(bad_input):
    with pytest.raises(ValueError):
        extract_pdf(bad_input)


def test_extract_pdf_non_bytes_raises_valueerror():
    with pytest.raises(ValueError):
        extract_pdf("definitely not bytes")


def test_make_evidence_id_deterministic():
    document_id = uuid.uuid4()
    first = make_evidence_id(document_id, 0, 3, EvidenceType.TEXT)
    second = make_evidence_id(document_id, 0, 3, EvidenceType.TEXT)
    assert first == second


def test_make_evidence_id_differs_by_block():
    document_id = uuid.uuid4()
    a = make_evidence_id(document_id, 0, 0, EvidenceType.TEXT)
    b = make_evidence_id(document_id, 0, 1, EvidenceType.TEXT)
    assert a != b


def test_make_evidence_id_differs_by_page_and_type():
    document_id = uuid.uuid4()
    base = make_evidence_id(document_id, 0, 0, EvidenceType.TEXT)
    other_page = make_evidence_id(document_id, 1, 0, EvidenceType.TEXT)
    other_type = make_evidence_id(document_id, 0, 0, EvidenceType.TABLE)
    assert base != other_page
    assert base != other_type


def test_assess_quality_good_on_dense_text():
    score, verdict = assess_quality(
        char_count=800, word_count=150, suspicious_ratio=0.0, has_images=False
    )
    assert verdict == QualityVerdict.GOOD
    assert 0.0 <= score <= 1.0


def test_assess_quality_bad_on_empty():
    score, verdict = assess_quality(
        char_count=0, word_count=0, suspicious_ratio=0.0, has_images=False
    )
    assert score == 0.0
    assert verdict == QualityVerdict.BAD


def test_assess_quality_bad_on_garbled():
    score, verdict = assess_quality(
        char_count=800, word_count=150, suspicious_ratio=0.9, has_images=False
    )
    assert verdict == QualityVerdict.BAD
    assert 0.0 <= score <= 1.0


def test_labeled_pdf_surfaces_source_page_number():
    if not hasattr(pymupdf.Document, "set_page_labels"):
        pytest.skip("pymupdf has no Document.set_page_labels in this version")
    labels = [{"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 7}]
    data = _make_pdf_bytes([DENSE_TEXT, DENSE_TEXT], labels=labels)
    _, pages = extract_pdf(data)
    assert len(pages) == 2
    # Physical index stays 0-based; labels come from the PDF, not the index.
    assert [p.pdf_page_number for p in pages] == [0, 1]
    assert pages[0].source_page_number == "A-7"
    assert pages[1].source_page_number == "A-8"
    for page in pages:
        assert page.source_page_number != str(page.pdf_page_number)
