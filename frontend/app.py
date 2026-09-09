from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

import api as api_client
import scope as scope_util

ACCENT = "#E2A52C"
ERROR = "#D1604D"
GREEN = "#7FB069"

REL_TYPES = ["CORROBORATES", "CONTRADICTS", "CONTEXTUAL_DIFFERENCE", "RELATED"]

CSS = """
<style>
.stApp { background: #0B0B0B; color: #E8E8E8; }
#MainMenu, footer { display: none !important; }
.block-container { max-width: 1480px; padding-top: 0.7rem;
  padding-left: 1.5rem; padding-right: 1.5rem; padding-bottom: 3rem; }
.navbar { display: flex; justify-content: space-between; align-items: flex-start;
  border-bottom: 1px solid #202020; padding-bottom: 10px; }
.brand { font-size: 22px; font-weight: 700; letter-spacing: 0.22em; }
.brand-sub { font-size: 12px; letter-spacing: 0.1em; color: #8A8A8A; margin-top: 2px; }
.section { font-size: 16px; font-weight: 700; letter-spacing: 0.16em;
  color: #E8E8E8; margin: 16px 0 6px; }
.hr { border-top: 1px solid #202020; margin: 2px 0 10px; }
.tbl { width: 100%; border-collapse: collapse; font-size: 14px; }
.tbl th { text-align: left; font-size: 11.5px; font-weight: 600;
  letter-spacing: 0.12em; color: #8A8A8A; padding: 6px 10px;
  border-bottom: 1px solid #202020; }
.tbl td { padding: 8px 10px; border-bottom: 1px solid #161616; color: #E8E8E8;
  vertical-align: top; }
.tbl tr:last-child td { border-bottom: none; }
.doc-name { font-size: 15.5px; font-weight: 600; overflow-wrap: anywhere; }
.doc-row { background: #111111; border: 1px solid #202020; border-radius: 3px;
  padding: 6px 12px; margin-bottom: 6px; }
.doc-row-sel { border-left: 3px solid #E2A52C; }
.doc-sel-name { color: #E2A52C; }
.doc-tbl-head { font-size: 11.5px; font-weight: 700; letter-spacing: 0.12em;
  color: #8A8A8A; padding: 6px 10px; border-bottom: 1px solid #2A2A2A; }
.doc-tbl-cell { font-size: 13.5px; color: #E8E8E8; padding: 7px 10px;
  border-bottom: 1px solid #1A1A1A; overflow-wrap: anywhere; }
.st-ok { color: #7FB069; } .st-warn { color: #E2A52C; } .st-err { color: #D1604D; } .st-mut { color: #8A8A8A; }
.fact { background: #111111; border: 1px solid #202020; border-left: 2px solid #3A3A3A;
  border-radius: 3px; padding: 12px 16px; margin-bottom: 10px; }
.fact-subj { font-size: 12.5px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: #8A8A8A; }
.fact-pred { font-size: 17px; font-weight: 600; margin-top: 2px; }
.fact-value { font-size: 25px; font-weight: 700; margin: 6px 0 2px; }
.fact-norm-label { font-size: 11.5px; letter-spacing: 0.12em; color: #5F5F5F;
  margin-top: 6px; }
.fact-norm { font-size: 13.5px; color: #B5B5B5; }
.fact-norm b { color: #E2A52C; font-weight: 600; }
.fact-meta { font-size: 12.5px; color: #8A8A8A; margin-top: 8px; }
.fact-meta b { color: #B5B5B5; font-weight: 600; }
.rel { background: #111111; border: 1px solid #202020; border-radius: 3px;
  padding: 12px 16px; margin-bottom: 10px; }
.rel-contra { border-left: 2px solid #D1604D; }
.rel-corrob { border-left: 2px solid #2E5C43; }
.rel-ctx { border-left: 2px dashed #8A8A8A; }
.badge { display: inline-block; font-size: 11px; font-weight: 700;
  letter-spacing: 0.12em; padding: 2px 8px; border: 1px solid #3A3128;
  border-radius: 2px; }
.badge-contra { color: #D1604D; border-color: #D1604D; }
.badge-review { color: #E2A52C; border-color: #E2A52C; }
.badge-grn { color: #7FB069; border-color: #7FB069; }
.badge-dim { color: #A89C8C; }
.badge-pend { color: #E2A52C; border-color: #E2A52C; }
.conf { font-size: 12.5px; color: #A89C8C; }
.expl { font-size: 13.5px; color: #B5B5B5; line-height: 1.55; margin-top: 8px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.pair-cell { background: #161616; border: 1px solid #202020; border-radius: 2px;
  padding: 10px 12px; }
.pair-val { font-size: 17px; font-weight: 700; }
.pair-doc { font-size: 12.5px; color: #A89C8C; margin-top: 3px; }
.ev { font-size: 12.5px; color: #B5B5B5; background: #0B0B0B;
  border: 1px solid #202020; border-radius: 2px; padding: 8px 10px;
  margin-top: 6px; white-space: pre-wrap; font-family: monospace; }
.prov { font-size: 12.5px; color: #A89C8C; margin-top: 8px;
  letter-spacing: 0.06em; }
div.stButton > button { background: #161616; color: #E8E8E8;
  border: 1px solid #3A3128; border-radius: 4px; font-size: 13px;
  font-weight: 600; letter-spacing: 0.08em; padding: 7px 16px; }
div.stButton > button:hover { border-color: #E2A52C; color: #E2A52C; }
div.stButton > button[kind="primary"] { background: #E2A52C; color: #1A140D;
  border: 1px solid #E2A52C; }
.upload-box { max-width: 600px; margin: 24px auto 0; background: #111111;
  border: 1px solid #202020; border-radius: 4px; padding: 24px 26px; }
.upload-title { font-size: 15px; font-weight: 600; letter-spacing: 0.14em; }
.hero { text-align: center; padding: 6vh 0 0; }
.hero-title { font-size: 32px; font-weight: 700; letter-spacing: 0.3em;
  text-indent: 0.3em; }
.hero-sub { font-size: 13px; color: #8A8A8A; letter-spacing: 0.08em; margin-top: 6px; }
.step { display: inline-block; font-size: 12px; font-weight: 700;
  letter-spacing: 0.1em; padding: 3px 10px; border: 1px solid #3A3128;
  border-radius: 2px; margin-right: 6px; color: #A89C8C; }
.step-done { color: #7FB069; border-color: #7FB069; }
.step-fail { color: #D1604D; border-color: #D1604D; }
.step-run { color: #E2A52C; border-color: #E2A52C; }
</style>
"""


def _init_state() -> None:
    defaults = {
        "selected_doc_ids": [],
        "rel_scope_ids": [],
        "show_rel_scope": False,
        "selected_fact_id": None,
        "selected_rel_id": None,
        "query": "",
        "rel_type_filter": "All",
        "rel_min_conf": 0.0,
        "doc_ids": [],
        "doc_sizes": {},
        "doc_stats": {},
        "pipeline_reports": {},
        "rel_cache": {"key": None, "rels": []},
        "rel_details": {},
        "fact_cache": {},
        "bundle_cache": {},
        "documents_cache": None,
        "search_cache": {},
        "show_uploader": False,
        "uploader_nonce": 0,
        "entered": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_time(value) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%b %d")
    except Exception:
        return str(value)[:16]


def _fmt_conf(value) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _norm_text(fact: dict) -> str | None:
    num, unit = fact.get("normalized_number"), fact.get("normalized_unit")
    if num is None and not unit:
        return None
    parts = ""
    if num is not None:
        parts = f"{num:g}" if isinstance(num, (int, float)) else str(num)
    if unit:
        parts = f"{parts} {unit}".strip()
    return parts or None


def _page_label(evidence: dict) -> str:
    try:
        pdf_page = int(evidence.get("pdf_page_number", 0)) + 1
    except (TypeError, ValueError):
        return "page unknown"
    label = f"PDF p. {pdf_page}"
    if evidence.get("source_page_number"):
        label += f" (printed p. {evidence['source_page_number']})"
    return label


def _fact_page_label(fact: dict, evidence_by_id: dict) -> str:
    for eid in fact.get("evidence_ids", []) or []:
        ev = evidence_by_id.get(str(eid))
        if ev is not None:
            return _page_label(ev)
    return "page unknown"


def _cached_documents() -> tuple[list[dict], str | None]:
    cached = st.session_state.documents_cache
    if isinstance(cached, list):
        return cached, None
    docs, err = api_client.list_documents()
    if docs is not None:
        for doc in docs:
            doc_id = str(doc.get("id"))
            if doc_id and doc_id not in st.session_state.doc_ids:
                st.session_state.doc_ids.append(doc_id)
        st.session_state.documents_cache = docs
        return docs, None
    if err == "NOT_SUPPORTED":
        fallback: list[dict] = []
        for doc_id in list(st.session_state.doc_ids):
            bundle, _ = _cached_bundle(doc_id)
            if bundle is not None:
                fallback.append(bundle["document"])
        return fallback, None
    return [], err


def _invalidate_workspace_caches() -> None:
    st.session_state.doc_stats = {}
    st.session_state.rel_cache = {"key": None, "rels": []}
    st.session_state.rel_details = {}
    st.session_state.fact_cache = {}
    st.session_state.bundle_cache = {}
    st.session_state.documents_cache = None
    st.session_state.search_cache = {}


def _prune_state_to_documents(documents: list[dict]) -> list[dict]:
    known = [str(d.get("id")) for d in documents]
    st.session_state.selected_doc_ids = scope_util.prune_selection(
        [str(i) for i in st.session_state.selected_doc_ids], known
    )
    st.session_state.rel_scope_ids = scope_util.prune_selection(
        [str(i) for i in st.session_state.rel_scope_ids], known
    )
    return [d for d in documents if str(d.get("id")) in st.session_state.selected_doc_ids]


def _handle_uploads(files) -> None:
    uploaded_ids: list[str] = []
    for uploaded in files:
        data = uploaded.getvalue()
        with st.spinner(f"Ingesting {uploaded.name} ..."):
            result, err = api_client.upload_pdf(uploaded.name, data)
        if err is not None:
            st.error(f"{uploaded.name}: {err}")
            continue
        doc = (result or {}).get("document", {})
        doc_id = str(doc.get("id", ""))
        if doc_id:
            uploaded_ids.append(doc_id)
            if doc_id not in st.session_state.doc_ids:
                st.session_state.doc_ids.append(doc_id)
            st.session_state.doc_sizes[doc_id] = len(data)
        st.success(f"{uploaded.name}: {doc.get('ingestion_status', '—')} "
                   f"({(result or {}).get('page_count', '?')} pages, "
                   f"{(result or {}).get('evidence_count', '?')} evidence units)")
    st.session_state.selected_doc_ids = scope_util.prune_selection(
        uploaded_ids,
        list(st.session_state.doc_ids),
    )
    st.session_state.documents_cache = None
    st.session_state.bundle_cache = {}
    st.session_state.search_cache = {}
    st.session_state.uploader_nonce += 1
    st.session_state.show_uploader = False
    st.session_state.entered = True
    st.rerun()


def _render_uploader(nonce_key: str, cta: str = "INGEST") -> None:
    files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=nonce_key,
    )
    if files and st.button(cta, type="primary"):
        _handle_uploads(files)


def _render_navbar() -> None:
    st.markdown(
        '<div class="navbar"><div><div class="brand">ALETHEIA</div>'
        '<div class="brand-sub">FACT KNOWLEDGE LAYER</div></div></div>',
        unsafe_allow_html=True,
    )
    left, spacer, up_col, ref_col = st.columns([6, 2, 1.4, 1.2])
    with up_col:
        if st.button("Upload PDFs", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()
    with ref_col:
        if st.button("Refresh", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("sel_") or key.startswith("rs_"):
                    del st.session_state[key]
            st.session_state.selected_doc_ids = []
            st.session_state.rel_scope_ids = []
            st.session_state.show_rel_scope = False
            st.session_state.selected_fact_id = None
            st.session_state.selected_rel_id = None
            st.session_state.query = ""
            st.session_state.rel_type_filter = "All"
            st.session_state.rel_min_conf = 0.0
            _invalidate_workspace_caches()
            st.session_state.show_uploader = False
            st.session_state.entered = False
            st.rerun()
    if st.session_state.show_uploader:
        with st.expander("Upload PDFs", expanded=True):
            _render_uploader(f"uploader_{st.session_state.uploader_nonce}")


def _render_empty_state(doc_count: int = 0) -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">ALETHEIA</div>'
        '<div class="hero-sub">FACT KNOWLEDGE LAYER</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="upload-box"><div class="upload-title">UPLOAD PDF FILES</div>'
        '<div class="fact-meta">One or more PDFs. Each file is ingested into '
        "grounded evidence before any facts are extracted.</div></div>",
        unsafe_allow_html=True,
    )
    _render_uploader(f"hero_uploader_{st.session_state.uploader_nonce}")
    if doc_count > 0 and not st.session_state.entered:
        st.markdown(
            f"<div class='fact-meta' style='text-align:center'>"
            f"{doc_count} document(s) already in the backend.</div>",
            unsafe_allow_html=True,
        )
        if st.button("LOAD WORKSPACE", type="primary"):
            st.session_state.entered = True
            st.rerun()
    health, err = api_client.health()
    if err is not None:
        st.error(err)
    elif health is not None and not (health.get("database", {}) or {}).get("reachable", True):
        detail = (health.get("database", {}) or {}).get("detail", "unknown")
        st.warning(f"Backend is up but the database is unreachable: {detail}")


def _doc_stats(doc_id: str) -> dict:
    cached = st.session_state.doc_stats.get(doc_id)
    if isinstance(cached, dict):
        return cached
    stats: dict = {"facts": None, "normalized": None, "relationships": None}
    facts, ferr = _load_doc_facts(doc_id)
    if ferr is None:
        stats["facts"] = len(facts)
        stats["normalized"] = scope_util.normalization_summary(facts)["normalized"]
    rels, rerr = api_client.list_relationships(doc_id)
    if rerr is None:
        stats["relationships"] = len(rels or [])
    if ferr is None and rerr is None:
        st.session_state.doc_stats[doc_id] = stats
    return stats


def _upload_label(status: str | None) -> str:
    return {"COMPLETED": "Uploaded", "PARTIAL": "Partial", "FAILED": "Failed"}.get(
        status or "", status or "—"
    )


def _stat_text(value) -> str:
    return "—" if value is None else str(value)


def _render_documents_section(documents: list[dict]) -> None:
    st.markdown('<div class="section">DOCUMENTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    with st.spinner("Loading workspace documents ..."):
        for doc in documents:
            _doc_stats(str(doc.get("id")))
    header = st.columns([0.4, 2.8, 0.7, 0.9, 1.0, 0.8, 1.0, 0.9])
    for col, name in zip(header, ["", "DOCUMENT", "PAGES", "UPLOADED", "STATUS", "FACTS",
                                  "NORMALIZED", "RELATIONS"]):
        col.markdown(f"<div class='doc-tbl-head'>{name}</div>", unsafe_allow_html=True)
    for doc in documents:
        doc_id = str(doc.get("id"))
        stats = st.session_state.doc_stats.get(doc_id, {})
        row = st.columns([0.4, 2.8, 0.7, 0.9, 1.0, 0.8, 1.0, 0.9])
        with row[0]:
            checked = st.checkbox(
                "select",
                value=doc_id in st.session_state.selected_doc_ids,
                key=f"sel_{doc_id}",
                label_visibility="collapsed",
            )
        if checked and doc_id not in st.session_state.selected_doc_ids:
            st.session_state.selected_doc_ids.append(doc_id)
            st.session_state.selected_fact_id = None
            st.session_state.selected_rel_id = None
        if not checked and doc_id in st.session_state.selected_doc_ids:
            st.session_state.selected_doc_ids.remove(doc_id)
            st.session_state.selected_fact_id = None
            st.session_state.selected_rel_id = None
        row[1].markdown(f"<div class='doc-tbl-cell doc-name'>{_esc(doc.get('filename', '—'))}</div>",
                        unsafe_allow_html=True)
        row[2].markdown(f"<div class='doc-tbl-cell'>{_esc(doc.get('page_count', '—'))}</div>",
                        unsafe_allow_html=True)
        row[3].markdown(f"<div class='doc-tbl-cell'>{_esc(_fmt_time(doc.get('created_at')))}</div>",
                        unsafe_allow_html=True)
        row[4].markdown(f"<div class='doc-tbl-cell'>{_esc(_upload_label(doc.get('ingestion_status')))}</div>",
                        unsafe_allow_html=True)
        row[5].markdown(f"<div class='doc-tbl-cell'>{_stat_text(stats.get('facts'))}</div>",
                        unsafe_allow_html=True)
        row[6].markdown(f"<div class='doc-tbl-cell'>{_stat_text(stats.get('normalized'))}</div>",
                        unsafe_allow_html=True)
        row[7].markdown(f"<div class='doc-tbl-cell'>{_stat_text(stats.get('relationships'))}</div>",
                        unsafe_allow_html=True)
    st.markdown(
        f"<div class='fact-meta'>{scope_util.selection_label(len(st.session_state.selected_doc_ids))}</div>",
        unsafe_allow_html=True,
    )


def _doc_stage_badges(doc_id: str, stored: dict) -> str:
    stats = st.session_state.doc_stats.get(doc_id, {})
    facts_n = stats.get("facts")
    norm_n = stats.get("normalized")
    rels_n = stats.get("relationships")
    facts_entry = stored.get("facts") or {}
    norm_entry = stored.get("normalize") or {}
    rel_entry = stored.get("relationships") or {}
    if facts_entry.get("error") is not None:
        facts_badge = '<span class="badge badge-contra">EXTRACTION FAILED</span>'
    elif isinstance(facts_entry.get("report"), dict) or (facts_n or 0) > 0:
        facts_badge = '<span class="badge badge-grn">FACTS EXTRACTED</span>'
    else:
        facts_badge = '<span class="badge badge-pend">FACTS PENDING</span>'
    if norm_entry.get("error") is not None:
        norm_badge = '<span class="badge badge-contra">NORMALIZATION FAILED</span>'
    elif isinstance(norm_entry.get("report"), dict) or (norm_n or 0) > 0:
        norm_badge = '<span class="badge badge-grn">NORMALIZED</span>'
    else:
        norm_badge = '<span class="badge badge-pend">NORMALIZATION PENDING</span>'
    if rel_entry.get("error") is not None:
        rel_badge = '<span class="badge badge-contra">RELATIONS FAILED</span>'
    elif isinstance(rel_entry.get("report"), dict) or (rels_n or 0) > 0:
        rel_badge = '<span class="badge badge-grn">RELATIONS FORMED</span>'
    else:
        rel_badge = '<span class="badge badge-pend">RELATIONS PENDING</span>'
    return f"{facts_badge} {norm_badge} {rel_badge}"


def _render_overview(documents: list[dict]) -> None:
    st.markdown('<div class="section">SELECTED DOCUMENTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if not documents:
        st.info("Select one or more documents above to inspect the workspace.")
        return
    for doc in documents:
        doc_id = str(doc.get("id"))
        stored = _pipeline_reports(doc_id)
        st.markdown(
            f"<div class='doc-row doc-row-sel'><span class='doc-name doc-sel-name'>"
            f"{_esc(doc.get('filename', '—'))}</span> "
            f"{_doc_stage_badges(doc_id, stored)}</div>",
            unsafe_allow_html=True,
        )


def _pipeline_reports(doc_id: str) -> dict:
    store = st.session_state.pipeline_reports
    report = store.get(doc_id)
    if not isinstance(report, dict):
        report = {}
        store[doc_id] = report
    return report


def _load_doc_facts(doc_id: str) -> tuple[list[dict], str | None]:
    cached = st.session_state.fact_cache.get(doc_id)
    if isinstance(cached, list):
        return cached, None
    facts, err = api_client.list_facts(doc_id)
    if err is not None:
        return [], err
    items = facts or []
    st.session_state.fact_cache[doc_id] = items
    return items, None


def _cached_doc_facts(doc_id: str) -> list[dict]:
    items, _ = _load_doc_facts(doc_id)
    return items


def _cached_bundle(doc_id: str) -> tuple[dict | None, str | None]:
    cache = st.session_state.bundle_cache
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.bundle_cache = cache
    cached = cache.get(doc_id)
    if isinstance(cached, dict):
        return cached, None
    bundle, err = api_client.get_bundle(doc_id)
    if err is not None or not isinstance(bundle, dict):
        return None, err
    cache[doc_id] = bundle
    return bundle, None


def _search_key(query: str, selected_ids: list[str], limit: int) -> tuple:
    normalized = " ".join(query.strip().split()).casefold()
    return (normalized, tuple(sorted(set(str(i) for i in selected_ids))), int(limit))


def _cached_search(
    query: str, documents: list[dict], limit: int = 30
) -> tuple[list[dict], str | None]:
    cache = st.session_state.search_cache
    if not isinstance(cache, dict):
        cache = {}
        st.session_state.search_cache = cache
    selected_ids = [str(d.get("id")) for d in documents]
    key = _search_key(query, selected_ids, limit)
    cached = cache.get(key)
    if isinstance(cached, dict):
        return cached.get("hits", []), cached.get("error")
    hits: list[dict] = []
    first_err = None
    for doc in documents:
        result, err = api_client.search(query.strip(), str(doc.get("id")), limit=limit)
        if err is not None:
            first_err = first_err or err
            continue
        for hit in (result or {}).get("hits", []) or []:
            hits.append(hit)
    cache[key] = {"hits": hits, "error": first_err}
    return hits, first_err


def _refresh_doc_stats(doc_ids: list[str]) -> None:
    for doc_id in doc_ids:
        st.session_state.doc_stats.pop(doc_id, None)


def _invalidate_fact_cache(doc_ids: list[str]) -> None:
    for doc_id in doc_ids:
        st.session_state.fact_cache.pop(doc_id, None)


def _invalidate_fact_caches(doc_ids: list[str]) -> None:
    _invalidate_fact_cache(doc_ids)
    st.session_state.search_cache = {}


def _run_extract(documents: list[dict]) -> None:
    for doc in documents:
        doc_id = str(doc.get("id"))
        with st.spinner(f"Extracting facts for {doc.get('filename', doc_id[:8])} ..."):
            report, err = api_client.trigger_facts(doc_id)
        _pipeline_reports(doc_id)["facts"] = {"report": report, "error": err}
    doc_ids = [str(d.get("id")) for d in documents]
    _refresh_doc_stats(doc_ids)
    _invalidate_fact_caches(doc_ids)


def _run_normalize(documents: list[dict]) -> None:
    for doc in documents:
        doc_id = str(doc.get("id"))
        with st.spinner(f"Normalizing facts for {doc.get('filename', doc_id[:8])} ..."):
            report, err = api_client.trigger_normalize(doc_id)
        _pipeline_reports(doc_id)["normalize"] = {"report": report, "error": err}
    doc_ids = [str(d.get("id")) for d in documents]
    _refresh_doc_stats(doc_ids)
    _invalidate_fact_caches(doc_ids)


def _run_relationships(documents: list[dict]) -> None:
    for doc in documents:
        doc_id = str(doc.get("id"))
        with st.spinner(f"Finding relationships for {doc.get('filename', doc_id[:8])} ..."):
            report, err = api_client.trigger_relationships(doc_id)
        _pipeline_reports(doc_id)["relationships"] = {"report": report, "error": err}
    _refresh_doc_stats([str(d.get("id")) for d in documents])
    st.session_state.search_cache = {}


def _scope_fact_total(documents: list[dict]) -> int | None:
    total = 0
    for doc in documents:
        stats = _doc_stats(str(doc.get("id")))
        count = stats.get("facts")
        if count is None:
            return None
        total += count
    return total


def _render_processing(documents: list[dict], all_documents: list[dict]) -> None:
    st.markdown('<div class="section">PROCESSING</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if not documents:
        st.info("Select at least one document to enable pipeline actions.")
    fact_total = _scope_fact_total(documents) if documents else None
    col_e, col_n, col_r = st.columns(3)
    with col_e:
        if st.button("EXTRACT FACTS", key="pipe_extract", type="primary",
                     use_container_width=True, disabled=not documents):
            _run_extract(documents)
            st.rerun()
    with col_n:
        extract_disabled = not documents or fact_total == 0
        if st.button("NORMALIZE", key="pipe_normalize",
                     use_container_width=True, disabled=extract_disabled):
            _run_normalize(documents)
            st.rerun()
        if documents and extract_disabled:
            st.caption("No extracted facts in scope yet.")
    with col_r:
        if st.button("FORM RELATIONS", key="pipe_relate_open",
                     use_container_width=True, disabled=extract_disabled):
            st.session_state.show_rel_scope = True
            st.session_state.rel_scope_ids = [str(d.get("id")) for d in documents]
            st.rerun()
        if extract_disabled and documents:
            st.caption("No extracted facts in scope yet.")
    if st.session_state.show_rel_scope:
        with st.expander("FORM RELATIONS — select documents to compare", expanded=True):
            for doc in all_documents:
                doc_id = str(doc.get("id"))
                checked = st.checkbox(
                    str(doc.get("filename", "—")),
                    value=doc_id in st.session_state.rel_scope_ids,
                    key=f"rs_{doc_id}",
                )
                if checked and doc_id not in st.session_state.rel_scope_ids:
                    st.session_state.rel_scope_ids.append(doc_id)
                if not checked and doc_id in st.session_state.rel_scope_ids:
                    st.session_state.rel_scope_ids.remove(doc_id)
            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("CANCEL", key="pipe_relate_cancel", use_container_width=True):
                    st.session_state.show_rel_scope = False
                    st.rerun()
            with btn_cols[1]:
                if st.button("FORM RELATIONS", key="pipe_relate_go", type="primary",
                             use_container_width=True):
                    scope_docs = [d for d in all_documents
                                  if str(d.get("id")) in st.session_state.rel_scope_ids]
                    if not scope_docs:
                        st.info("Select at least one document to compare.")
                    else:
                        _run_relationships(scope_docs)
                        st.session_state.show_rel_scope = False
                        st.session_state.rel_cache = {"key": None, "rels": []}
                        st.session_state.rel_details = {}
                        st.session_state.search_cache = {}
                        st.rerun()
    for doc in documents:
        stored = _pipeline_reports(str(doc.get("id")))
        for kind, label in (("facts", "Fact extraction"),
                            ("normalize", "Normalization"),
                            ("relationships", "Relationship reasoning")):
            entry = stored.get(kind) or {}
            if entry.get("error") is not None:
                st.error(f"{doc.get('filename', '—')}: {label} failed: {entry['error']}")
            elif isinstance(entry.get("report"), dict):
                rep = dict(entry["report"])
                for list_key in ("facts", "relationships"):
                    if isinstance(rep.get(list_key), list):
                        rep[list_key] = f"{len(rep[list_key])} items (see table below)"
                with st.expander(
                    f"{doc.get('filename', '—')}: {label} report", expanded=False
                ):
                    st.json(rep)


def _rel_badge(rel: dict) -> str:
    rtype = str(rel.get("relationship_type", "?"))
    if rtype == "CONTRADICTS":
        cls = "badge-contra"
    elif rtype == "CORROBORATES":
        cls = "badge-grn"
    else:
        cls = "badge-dim"
    review = ""
    if rel.get("status") == "NEEDS_REVIEW":
        review = ' <span class="badge badge-review">NEEDS REVIEW</span>'
    return (f'<span class="badge {cls}">{_esc(rtype)}</span>{review} '
            f'<span class="conf">{_fmt_conf(rel.get("confidence"))}</span>')


def _fact_summary(fact: dict, doc_names: dict) -> str:
    return (f"<div class='fact-subj'>{_esc(fact.get('subject', '—'))}</div>"
            f"<div class='pair-val'>{_esc(fact.get('value_text', '—'))}</div>"
            f"<div class='pair-doc'>{_esc(fact.get('predicate', '—'))} · "
            f"{_esc(doc_names.get(str(fact.get('document_id')), '—'))}</div>")


def _match_signals(detail: dict) -> str:
    rel = (detail or {}).get("relationship", {}) or {}
    meta = rel.get("reasoning_metadata", {}) or {}
    if not isinstance(meta, dict) or not meta:
        return ""
    fields = [
        ("numeric", meta.get("numeric_verdict")),
        ("incompatible", meta.get("incompatible_dimensions")),
        ("unknown", meta.get("unknown_dimensions")),
        ("strong", meta.get("strong_match")),
        ("overlap", meta.get("semantic_overlap")),
        ("model", meta.get("llm_model")),
    ]
    parts = []
    for name, value in fields:
        if value is None or value == []:
            continue
        if isinstance(value, list):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        parts.append(f"{name}: {text}")
    return "; ".join(parts)


def _render_relationships_section(documents: list[dict], doc_names: dict) -> None:
    st.markdown('<div class="section">RELATIONSHIPS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if not documents:
        st.info("Select documents to inspect their relationships.")
        return
    type_cols = st.columns(5)
    for col, opt in zip(type_cols, ["All"] + REL_TYPES):
        btn_type = "primary" if st.session_state.rel_type_filter == opt else "secondary"
        if col.button(opt, key=f"reltype_{opt}", type=btn_type, use_container_width=True):
            st.session_state.rel_type_filter = opt
            st.rerun()
    min_conf = st.slider("Min confidence", 0.0, 1.0,
                         float(st.session_state.rel_min_conf), 0.05)
    st.session_state.rel_min_conf = min_conf
    sel = st.session_state.rel_type_filter
    rel_type = None if sel == "All" else sel
    scope_ids = [str(d.get("id")) for d in documents]
    scope_key = scope_util.rel_scope_key(scope_ids)
    cache = st.session_state.rel_cache
    if not isinstance(cache, dict):
        cache = {"key": None, "rels": []}
        st.session_state.rel_cache = cache
    if cache.get("key") != scope_key:
        with st.spinner("Loading relationships ..."):
            rels: dict[str, dict] = {}
            first_err = None
            for doc in documents:
                items, err = api_client.list_relationships(str(doc.get("id")))
                if err is not None:
                    first_err = first_err or err
                    continue
                for rel in items or []:
                    rels[str(rel.get("id"))] = rel
            scope_fact_ids: set[str] = set()
            for doc_id in scope_ids:
                items, fact_err = _load_doc_facts(doc_id)
                if fact_err is not None:
                    first_err = first_err or fact_err
                    continue
                for fact in items:
                    scope_fact_ids.add(str(fact.get("id")))
            st.session_state.rel_cache = {
                "key": scope_key,
                "rels": scope_util.rels_in_scope(
                    list(rels.values()), scope_fact_ids
                ),
                "error": first_err,
            }
            cache = st.session_state.rel_cache
    ordered = scope_util.apply_relationship_view(
        cache.get("rels", []), rel_type, min_conf
    )
    if cache.get("error") is not None and not ordered:
        st.error(f"Could not load relationships: {cache['error']}")
        return
    if not ordered:
        st.info("No persisted relationships in scope for this filter.")
        return
    fact_map: dict[str, dict] = {}
    for doc_id in scope_ids:
        for fact in _cached_doc_facts(doc_id):
            fact_map[str(fact.get("id"))] = fact
    st.markdown(f"<div class='fact-meta'>{len(ordered)} relationship(s) in scope</div>",
                unsafe_allow_html=True)
    for rel in ordered[:100]:
        rel_id = str(rel.get("id"))
        rtype = str(rel.get("relationship_type", ""))
        css_class = ("rel rel-contra" if rtype == "CONTRADICTS"
                     else "rel rel-ctx" if rtype == "CONTEXTUAL_DIFFERENCE"
                     else "rel rel-corrob")
        fact_a = fact_map.get(str(rel.get("fact_a_id")), {})
        fact_b = fact_map.get(str(rel.get("fact_b_id")), {})
        st.markdown(f"<div class='{css_class}'>{_rel_badge(rel)}", unsafe_allow_html=True)
        st.markdown(
            f"<div class='pair'><div class='pair-cell'>"
            f"{_fact_summary(fact_a, doc_names)}</div>"
            f"<div class='pair-cell'>{_fact_summary(fact_b, doc_names)}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='fact-meta'>FACT A<br><b>{_esc(fact_a.get('subject', '—'))}</b> · "
            f"{_esc(fact_a.get('predicate', '—'))} · "
            f"<b>{_esc(fact_a.get('value_text', '—'))}</b></div>"
            f"<div class='fact-meta'>RELATIONSHIP<br><b>{_esc(rtype)}</b> · "
            f"{_esc(_fmt_conf(rel.get('confidence')))}</div>"
            f"<div class='fact-meta'>FACT B<br><b>{_esc(fact_b.get('subject', '—'))}</b> · "
            f"{_esc(fact_b.get('predicate', '—'))} · "
            f"<b>{_esc(fact_b.get('value_text', '—'))}</b></div>",
            unsafe_allow_html=True,
        )
        if rtype == "CONTRADICTS":
            st.markdown(
                "<div class='expl'>Material disagreement. "
                "Neither source is judged correct.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='expl'>{_esc(rel.get('explanation', ''))}</div>",
                    unsafe_allow_html=True)
        if st.button("VIEW EVIDENCE", key=f"rel_{rel.get('id')}"):
            st.session_state.selected_rel_id = rel_id
            st.rerun()
        if st.session_state.selected_rel_id == rel_id:
            detail = st.session_state.rel_details.get(rel_id)
            if detail is None:
                with st.spinner("Loading relationship evidence ..."):
                    fetched, derr = api_client.get_relationship_detail(rel_id)
                if derr is not None or fetched is None:
                    st.error(f"Could not load relationship evidence: {derr}")
                else:
                    st.session_state.rel_details[rel_id] = fetched
                    detail = fetched
            if detail is not None:
                with st.expander("RELATIONSHIP EVIDENCE", expanded=True):
                    signals = _match_signals(detail)
                    if signals:
                        st.markdown(
                            f"<div class='fact-meta'>MATCH SIGNALS<br>{_esc(signals)}</div>",
                            unsafe_allow_html=True)
                    ev_fact_a = detail.get("fact_a", {}) or fact_a
                    ev_fact_b = detail.get("fact_b", {}) or fact_b
                    for side, fact, evs in (("A", ev_fact_a, detail.get("evidence_a", []) or []),
                                            ("B", ev_fact_b, detail.get("evidence_b", []) or [])):
                        ev_map = {str(e.get("id")): e for e in evs}
                        page = _fact_page_label(fact, ev_map) if evs else "page unknown"
                        st.markdown(
                            f"<div class='prov'>Side {_esc(side)}: "
                            f"{_esc(fact.get('subject', '—'))} — "
                            f"{_esc(fact.get('value_text', '—'))} ({_esc(page)})</div>",
                            unsafe_allow_html=True)
                        if not evs:
                            st.markdown("<div class='ev'>no evidence returned</div>",
                                        unsafe_allow_html=True)
                        for ev in evs:
                            st.markdown(
                                f"<div class='ev'>{_esc(_page_label(ev))} · "
                                f"{_esc(ev.get('type', '?'))} · "
                                f"{_esc(ev.get('extraction_method', '?'))}\n"
                                f"{_esc((ev.get('text') or '')[:2000])}"
                                f"\n-- evidence_id {_esc(ev.get('id'))}</div>",
                                unsafe_allow_html=True,
                            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_fact_block(fact: dict, doc_name: str, evidence_by_id: dict, key: str) -> None:
    norm = _norm_text(fact)
    norm_html = (f'<div class="fact-norm-label">NORMALIZED</div>'
                 f'<div class="fact-norm"><b>{_esc(norm)}</b></div>') if norm else ""
    scope = fact.get("scope_text") or ""
    scope_html = f" · {_esc(scope)}" if scope else ""
    geo = fact.get("geography") or ""
    geo_html = f" · {_esc(geo)}" if geo else ""
    selected = st.session_state.selected_fact_id == str(fact.get("id"))
    border = ' style="border-left-color: #E2A52C;"' if selected else ""
    st.markdown(
        f'<div class="fact"{border}>'
        f'<div class="fact-subj">{_esc(fact.get("subject", "—"))}</div>'
        f'<div class="fact-pred">{_esc(fact.get("predicate", "—"))}</div>'
        f'<div class="fact-value">{_esc(fact.get("value_text", "—"))}</div>'
        f"{norm_html}"
        f'<div class="fact-meta"><b>{_esc(fact.get("time_text") or "time unknown")}</b>'
        f" · {_fmt_conf(fact.get('extraction_confidence'))} confidence"
        f"{scope_html}{geo_html}<br>{_esc(doc_name)} · "
        f"{_esc(_fact_page_label(fact, evidence_by_id))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Evidence / provenance", key=f"ev_{key}_{fact.get('id')}"):
        st.session_state.selected_fact_id = str(fact.get("id"))
        st.rerun()
    if selected:
        _render_fact_evidence(fact, doc_name, evidence_by_id)


def _render_fact_evidence(fact: dict, doc_name: str, evidence_by_id: dict) -> None:
    with st.expander("FACT DETAIL / PROVENANCE", expanded=True):
        st.markdown('<div class="prov">FACT → EVIDENCE → DOCUMENT → PDF PAGE</div>',
                    unsafe_allow_html=True)
        detail = {
            "fact_id": str(fact.get("id")),
            "subject": fact.get("subject"),
            "predicate": fact.get("predicate"),
            "original_value": fact.get("value_text"),
            "normalized": _norm_text(fact),
            "unit": fact.get("unit"),
            "time": fact.get("time_text"),
            "scope": fact.get("scope_text"),
            "geography": fact.get("geography"),
            "confidence": fact.get("extraction_confidence"),
            "status": fact.get("status"),
            "ambiguity_flags": fact.get("ambiguity_flags") or [],
            "document": doc_name,
        }
        st.json(detail)
        eids = fact.get("evidence_ids", []) or []
        if not eids:
            st.warning("No evidence IDs linked to this fact.")
            return
        full = dict(evidence_by_id)
        missing = [eid for eid in eids if str(eid) not in full]
        if missing:
            bundle, _ = _cached_bundle(str(fact.get("document_id")))
            if bundle is not None:
                for ev in bundle.get("evidence", []) or []:
                    full[str(ev.get("id"))] = ev
        for eid in eids:
            ev = full.get(str(eid))
            if ev is None:
                st.markdown(f"<div class='ev'>evidence {_esc(eid)}: unavailable</div>",
                            unsafe_allow_html=True)
                continue
            bbox = ev.get("bbox")
            st.markdown(
                f"<div class='ev'>{_esc(_page_label(ev))} · {_esc(ev.get('type', '?'))} · "
                f"{_esc(ev.get('extraction_method', '?'))} · "
                f"quality {ev.get('extraction_quality', '—')} · "
                f"bbox {_esc(bbox if bbox else '—')}\n{_esc((ev.get('text') or '')[:2000])}"
                f"\n-- evidence_id {_esc(ev.get('id'))}</div>",
                unsafe_allow_html=True,
            )


def _render_knowledge_section(documents: list[dict], doc_names: dict) -> None:
    st.markdown('<div class="section">KNOWLEDGE / FACTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if not documents:
        st.info("Select documents to inspect their facts.")
        return
    query = st.text_input(
        "Search facts, entities, metrics",
        value=st.session_state.query,
        placeholder="Search facts, entities, metrics...",
    )
    st.session_state.query = query

    if query.strip():
        with st.spinner("Searching knowledge layer ..."):
            hits, first_err = _cached_search(query, documents, limit=30)
        if first_err is not None and not hits:
            st.error(f"Search failed: {first_err}")
            return
        if not hits:
            st.info("No facts match this query.")
            return
        st.markdown(f"<div class='fact-meta'>{len(hits)} result(s)</div>",
                    unsafe_allow_html=True)
        for hit in hits:
            fact = hit.get("fact", {})
            doc = hit.get("document") or {}
            ev_by_id = {str(e.get("id")): e for e in hit.get("evidence", []) or []}
            st.markdown(
                f"<div class='fact-meta'>score {hit.get('score', 0):.2f} · "
                f"{_esc(hit.get('match_kind', '?'))}</div>",
                unsafe_allow_html=True,
            )
            _render_fact_block(fact, str(doc.get("filename", "—")), ev_by_id,
                               key=f"s{hit.get('score', 0):.3f}")
        return

    with st.spinner("Loading facts ..."):
        facts: list[dict] = []
        first_err = None
        for doc in documents:
            items, err = _load_doc_facts(str(doc.get("id")))
            if err is not None:
                first_err = first_err or err
                continue
            facts.extend(items[:50])
        facts = scope_util.filter_facts_by_scope(
            facts, [str(d.get("id")) for d in documents]
        )
        facts.sort(key=lambda f: str(f.get("id")))
    if first_err is not None and not facts:
        st.error(f"Could not load facts: {first_err}")
        return
    if not facts:
        st.info("No persisted facts yet for the selected documents. "
                "Run the pipeline below to extract some.")
        return
    ev_by_id: dict = {}
    for doc in documents:
        bundle, _ = _cached_bundle(str(doc.get("id")))
        if bundle is not None:
            for ev in bundle.get("evidence", []) or []:
                ev_by_id[str(ev.get("id"))] = ev
    st.markdown(f"<div class='fact-meta'>{len(facts)} fact(s)</div>",
                unsafe_allow_html=True)
    for fact in facts[:100]:
        _render_fact_block(fact, doc_names.get(str(fact.get("document_id")), "—"),
                           ev_by_id, key="all")


def main() -> None:
    st.set_page_config(page_title="Aletheia — Fact Knowledge Layer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()

    documents, err = _cached_documents()
    if err is not None:
        st.markdown('<div class="brand">ALETHEIA</div>', unsafe_allow_html=True)
        st.error(err)
        st.info("Start the backend (docker compose up --build), then press Refresh.")
        if st.button("Refresh"):
            st.rerun()
        return

    if not st.session_state.entered or not documents:
        _render_empty_state(len(documents))
        return

    _render_navbar()
    _render_documents_section(documents)
    selected = _prune_state_to_documents(documents)
    doc_names = {str(d.get("id")): str(d.get("filename", "—")) for d in documents}
    _render_overview(selected)
    _render_processing(selected, documents)
    _render_relationships_section(selected, doc_names)
    _render_knowledge_section(selected, doc_names)


if __name__ == "__main__":
    main()
