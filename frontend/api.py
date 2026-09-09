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
