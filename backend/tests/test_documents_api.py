"""T8 integration tests: POST /documents + GET /documents/{id}.

All PDFs are generated in-memory with PyMuPDF; no binary fixtures.
The API's ``get_db`` dependency is overridden with file-backed SQLite.
"""

import uuid

import pymupdf


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


def _upload_pdf(client, page_texts, filename="report.pdf"):
    data = _make_pdf_bytes(page_texts)
    return client.post(
        "/documents",
        files={"file": (filename, data, "application/pdf")},
    )


def test_post_documents_two_page_pdf_201_completed(api_client):
    response = _upload_pdf(api_client, [DENSE_TEXT, DENSE_TEXT])
    assert response.status_code == 201
    body = response.json()

    document = body["document"]
    assert document["ingestion_status"] == "COMPLETED"
    assert document["page_count"] == 2
    assert body["page_count"] == 2
    assert body["evidence_count"] > 0

    doc_id = document["id"]
    bundle = api_client.get(f"/documents/{doc_id}")
    assert bundle.status_code == 200
    evidence = bundle.json()["evidence"]
    assert len(evidence) == body["evidence_count"]
    assert evidence, "expected non-empty evidence"
    for unit in evidence:
        assert unit["document_id"] == doc_id
        assert unit["bbox"] is not None
        assert len(unit["bbox"]) == 4


def test_get_document_returns_bundle(api_client):
    post = _upload_pdf(api_client, [DENSE_TEXT, DENSE_TEXT])
    assert post.status_code == 201
    doc_id = post.json()["document"]["id"]

    response = api_client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["document"]["id"] == doc_id
    assert len(body["pages"]) == 2
    assert len(body["evidence"]) > 0
    page_numbers = [p["pdf_page_number"] for p in body["pages"]]
    assert page_numbers == [0, 1]


def test_post_non_pdf_bytes_400(api_client):
    response = api_client.post(
        "/documents",
        files={"file": ("evil.txt", b"hello world, not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_get_unknown_uuid_404(api_client):
    response = api_client.get(f"/documents/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_malformed_uuid_422(api_client):
    response = api_client.get("/documents/not-a-uuid")
    assert response.status_code == 422
