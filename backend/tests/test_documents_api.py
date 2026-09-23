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


def test_get_documents_lists_ingested_pdfs(api_client):
    first = _upload_pdf(api_client, [DENSE_TEXT], filename="a.pdf")
    second = _upload_pdf(api_client, [DENSE_TEXT], filename="b.pdf")
    assert first.status_code == 201
    assert second.status_code == 201

    response = api_client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    filenames = [d["filename"] for d in body]
    assert filenames == ["a.pdf", "b.pdf"]
    for doc in body:
        assert doc["id"]
        assert doc["ingestion_status"] in (
            "UPLOADED",
            "PROCESSING",
            "COMPLETED",
            "FAILED",
            "PARTIAL",
        )


def test_ingest_route_is_sync_def():
    import asyncio

    from app.api.documents import ingest_document

    assert not asyncio.iscoroutinefunction(ingest_document)


def _tiny_pdf(pages: int = 2) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    try:
        for index in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"Synthetic page {index} with neutral wording.")
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _upload(api_client, data: bytes):
    response = api_client.post(
        "/documents", files={"file": ("probe.pdf", data, "application/pdf")}
    )
    assert response.status_code == 201
    return response.json()["document"]["id"]


def test_page_image_is_served_after_upload(api_client):
    doc_id = _upload(api_client, _tiny_pdf())
    response = api_client.get(f"/documents/{doc_id}/pages/0/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_page_image_highlights_a_real_evidence_unit(api_client):
    doc_id = _upload(api_client, _tiny_pdf())
    bundle = api_client.get(f"/documents/{doc_id}").json()
    evidence = next(e for e in bundle["evidence"] if e.get("bbox"))
    response = api_client.get(
        f"/documents/{doc_id}/pages/{evidence['pdf_page_number']}/image",
        params={"evidence_id": evidence["id"]},
    )
    assert response.status_code == 200
    plain = api_client.get(
        f"/documents/{doc_id}/pages/{evidence['pdf_page_number']}/image"
    )
    assert response.content != plain.content, "the box must actually be drawn"


def test_evidence_on_another_page_is_rejected(api_client):
    doc_id = _upload(api_client, _tiny_pdf())
    bundle = api_client.get(f"/documents/{doc_id}").json()
    evidence = next(e for e in bundle["evidence"] if e["pdf_page_number"] == 0)
    response = api_client.get(
        f"/documents/{doc_id}/pages/1/image", params={"evidence_id": evidence["id"]}
    )
    assert response.status_code == 400
    assert "not on the requested page" in response.json()["detail"]


def test_unknown_evidence_is_rejected(api_client):
    import uuid as _uuid

    doc_id = _upload(api_client, _tiny_pdf())
    response = api_client.get(
        f"/documents/{doc_id}/pages/0/image",
        params={"evidence_id": str(_uuid.uuid4())},
    )
    assert response.status_code == 404


def test_page_image_404s_for_an_unknown_document(api_client):
    import uuid as _uuid

    assert api_client.get(
        f"/documents/{_uuid.uuid4()}/pages/0/image"
    ).status_code == 404
