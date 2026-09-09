"""Centralized HTTP client for the Aletheia FastAPI backend.

All Streamlit code talks to the backend through this module. Every helper
returns a ``(data, error)`` tuple — ``error`` is a human-readable string or
``None``. The UI never synthesizes data when the backend fails.
"""

from __future__ import annotations

import os

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
ENV_VAR = "ALETHEIA_API_URL"

_READ_TIMEOUT = 30
_UPLOAD_TIMEOUT = 300
# Fact extraction is bounded (representative chunks, serial Groq calls
# with pacing/backoff), but a full run can still take several minutes.
_PIPELINE_TIMEOUT = 900
_RELATIONSHIPS_TIMEOUT = 600


def get_base_url() -> str:
    """Resolve the API base URL (env wins, stripped of trailing slash)."""
    try:
        import streamlit as st

        secret_url = ""
        try:
            secret_url = str(st.secrets.get(ENV_VAR, "") or "")
        except Exception:
            secret_url = ""
        if secret_url.strip():
            return secret_url.strip().rstrip("/")
    except Exception:
        pass
    return os.environ.get(ENV_VAR, DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL


def _url(path: str) -> str:
    return f"{get_base_url()}{path}"


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def health() -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(_url("/health"), timeout=10)
        if resp.status_code != 200:
            return None, f"backend returned HTTP {resp.status_code}"
        return resp.json(), None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def list_documents() -> tuple[list[dict] | None, str | None]:
    """List all documents. Returns (None, 'NOT_SUPPORTED') when the
    backend predates the minimal GET /documents endpoint so the caller
    can fall back to the session registry."""
    try:
        resp = requests.get(_url("/documents"), timeout=_READ_TIMEOUT)
        if resp.status_code == 404:
            return None, "NOT_SUPPORTED"
        if resp.status_code != 200:
            return None, f"GET /documents returned HTTP {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        return data if isinstance(data, list) else [], None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def get_bundle(document_id: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(_url(f"/documents/{document_id}"), timeout=_READ_TIMEOUT)
        if resp.status_code == 404:
            return None, "document not found"
        if resp.status_code != 200:
            return None, f"GET /documents/{document_id} HTTP {resp.status_code}: {resp.text[:300]}"
        return resp.json(), None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def upload_pdf(filename: str, data: bytes) -> tuple[dict | None, str | None]:
    try:
        resp = requests.post(
            _url("/documents"),
            files={"file": (filename, data, "application/pdf")},
            timeout=_UPLOAD_TIMEOUT,
        )
        if resp.status_code == 201:
            return resp.json(), None
        detail = ""
        try:
            detail = str(resp.json().get("detail", resp.text[:300]))
        except Exception:
            detail = resp.text[:300]
        return None, f"HTTP {resp.status_code}: {detail}"
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def list_facts(document_id: str) -> tuple[list[dict] | None, str | None]:
    try:
        resp = requests.get(_url(f"/documents/{document_id}/facts"), timeout=_READ_TIMEOUT)
        if resp.status_code == 404:
            return None, "document not found"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        return data if isinstance(data, list) else [], None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def list_relationships(
    document_id: str,
    relationship_type: str | None = None,
    min_confidence: float = 0.0,
) -> tuple[list[dict] | None, str | None]:
    try:
        params: dict = {"min_confidence": min_confidence}
        if relationship_type:
            params["relationship_type"] = relationship_type
        resp = requests.get(
            _url(f"/documents/{document_id}/relationships"),
            params=params,
            timeout=_READ_TIMEOUT,
        )
        if resp.status_code == 404:
            return None, "document not found"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        return data if isinstance(data, list) else [], None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def get_relationship_detail(relationship_id: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(_url(f"/relationships/{relationship_id}"), timeout=_READ_TIMEOUT)
        if resp.status_code == 404:
            return None, "relationship not found"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return resp.json(), None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def search(
    query: str,
    document_id: str | None = None,
    limit: int = 20,
    min_score: float = 0.0,
) -> tuple[dict | None, str | None]:
    try:
        payload: dict = {"query": query, "limit": limit, "min_score": min_score}
        if document_id:
            payload["document_id"] = document_id
        resp = requests.post(_url("/search"), json=payload, timeout=_READ_TIMEOUT)
        if resp.status_code == 400:
            return None, "query must not be blank"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return resp.json(), None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def _post_pipeline_action(
    path: str, timeout: int
) -> tuple[dict | None, str | None]:
    """POST a per-document pipeline action (facts/normalize/relationships).

    Returns the backend report verbatim on success; surfaces backend
    failures (including 503 extraction-unavailable and per-chunk errors
    inside the report body) without synthesizing results.
    """
    try:
        resp = requests.post(_url(path), timeout=timeout)
        if resp.status_code == 404:
            return None, "document not found"
        if resp.status_code != 200:
            detail = ""
            try:
                detail = str(resp.json().get("detail", resp.text[:300]))
            except Exception:
                detail = resp.text[:300]
            return None, f"HTTP {resp.status_code}: {detail}"
        return resp.json(), None
    except Exception as exc:
        return None, f"API unreachable at {get_base_url()} — {_err(exc)}"


def trigger_facts(document_id: str) -> tuple[dict | None, str | None]:
    """Run bounded fact extraction; report carries chunks/facts/errors."""
    return _post_pipeline_action(
        f"/documents/{document_id}/facts", _PIPELINE_TIMEOUT
    )


def trigger_normalize(document_id: str) -> tuple[dict | None, str | None]:
    """Run deterministic normalization; report carries counts/errors."""
    return _post_pipeline_action(
        f"/documents/{document_id}/normalize", _READ_TIMEOUT
    )


def trigger_relationships(document_id: str) -> tuple[dict | None, str | None]:
    """Run relationship reasoning; report carries relationship counts."""
    return _post_pipeline_action(
        f"/documents/{document_id}/relationships", _RELATIONSHIPS_TIMEOUT
    )
