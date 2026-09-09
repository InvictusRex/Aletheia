"""Deterministic tests for the Streamlit HTTP client (frontend/api.py).

Scope: transport plumbing only — endpoint paths, methods, payloads, and
error propagation. The HTTP layer (``requests``) is stubbed; no network,
no backend, and crucially no Groq involvement. Real provider validation
happens against the live backend, never here.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "frontend_api",
    Path(__file__).resolve().parents[2] / "frontend" / "api.py",
)
frontend_api = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(frontend_api)


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


# ---------------------------------------------------------------------------
# Base URL resolution (pure, no network)
# ---------------------------------------------------------------------------


def test_base_url_default(monkeypatch):
    monkeypatch.delenv("ALETHEIA_API_URL", raising=False)
    assert frontend_api.get_base_url() == "http://localhost:8000"


def test_base_url_env_and_trailing_slash(monkeypatch):
    monkeypatch.setenv("ALETHEIA_API_URL", "http://backend:8000///")
    assert frontend_api.get_base_url() == "http://backend:8000"


# ---------------------------------------------------------------------------
# Upload → document ID
# ---------------------------------------------------------------------------


def test_upload_returns_document_id(monkeypatch):
    seen = {}

    def fake_post(url, files=None, timeout=None):
        seen["url"] = url
        seen["filename"] = files["file"][0]
        return _Resp(201, {"document": {"id": "doc-1"}, "page_count": 1})

    monkeypatch.setattr(frontend_api.requests, "post", fake_post)
    data, err = frontend_api.upload_pdf("report.pdf", b"%PDF-1.4")
    assert err is None
    assert data["document"]["id"] == "doc-1"
    assert seen["url"].endswith("/documents")
    assert seen["filename"] == "report.pdf"


def test_upload_propagates_backend_error(monkeypatch):
    monkeypatch.setattr(
        frontend_api.requests,
        "post",
        lambda *a, **k: _Resp(400, {"detail": "not a PDF"}, text="not a PDF"),
    )
    data, err = frontend_api.upload_pdf("x.pdf", b"nope")
    assert data is None
    assert "400" in err and "not a PDF" in err


# ---------------------------------------------------------------------------
# Pipeline triggers: endpoint invocation + error propagation
# ---------------------------------------------------------------------------


def test_trigger_facts_posts_report_path(monkeypatch):
    seen = {}

    def fake_post(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Resp(200, {"chunks_processed": 2, "facts": []})

    monkeypatch.setattr(frontend_api.requests, "post", fake_post)
    data, err = frontend_api.trigger_facts("doc-9")
    assert err is None
    assert data["chunks_processed"] == 2
    assert seen["url"].endswith("/documents/doc-9/facts")
    assert seen["timeout"] >= 60


def test_trigger_facts_404_maps_to_not_found(monkeypatch):
    monkeypatch.setattr(
        frontend_api.requests, "post", lambda *a, **k: _Resp(404, text="{}")
    )
    data, err = frontend_api.trigger_facts("missing")
    assert data is None
    assert err == "document not found"


def test_trigger_facts_503_surfaces_unavailable(monkeypatch):
    monkeypatch.setattr(
        frontend_api.requests,
        "post",
        lambda *a, **k: _Resp(
            503, {"detail": "LLM extraction unavailable"}, text="unavailable"
        ),
    )
    data, err = frontend_api.trigger_facts("doc-9")
    assert data is None
    assert "503" in err and "unavailable" in err


def test_trigger_normalize_and_relationships_paths(monkeypatch):
    seen = []

    def fake_post(url, timeout=None):
        seen.append(url)
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(frontend_api.requests, "post", fake_post)
    data, err = frontend_api.trigger_normalize("doc-9")
    assert err is None and data == {"ok": True}
    data, err = frontend_api.trigger_relationships("doc-9")
    assert err is None and data == {"ok": True}
    assert seen[0].endswith("/documents/doc-9/normalize")
    assert seen[1].endswith("/documents/doc-9/relationships")


def test_pipeline_unreachable_maps_cleanly(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(frontend_api.requests, "post", boom)
    for call in (
        frontend_api.trigger_facts,
        frontend_api.trigger_normalize,
        frontend_api.trigger_relationships,
    ):
        data, err = call("doc-9")
        assert data is None
        assert "unreachable" in err.lower()


# ---------------------------------------------------------------------------
# Read helpers surface backend data verbatim (no fabrication)
# ---------------------------------------------------------------------------


def test_list_facts_returns_backend_list(monkeypatch):
    facts = [{"id": "f1", "subject": "Revenue"}]
    monkeypatch.setattr(
        frontend_api.requests, "get", lambda *a, **k: _Resp(200, facts)
    )
    data, err = frontend_api.list_facts("doc-9")
    assert err is None
    assert data == facts


def test_list_facts_error_is_surfaced_not_silenced(monkeypatch):
    monkeypatch.setattr(
        frontend_api.requests, "get", lambda *a, **k: _Resp(500, text="boom")
    )
    data, err = frontend_api.list_facts("doc-9")
    assert data is None
    assert "500" in err
