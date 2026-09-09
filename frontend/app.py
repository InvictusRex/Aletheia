"""Aletheia — Fact Knowledge Layer (Streamlit frontend, single page).

Dark technical interface over the real FastAPI backend. Fully data-driven:
no hardcoded documents, facts, values, or explanations.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import api as api_client

# ---------------------------------------------------------------- constants

ACCENT = "#D0D500"
ERROR = "#FF4B3F"

REL_TYPES = ["CORROBORATES", "CONTRADICTS", "CONTEXTUAL_DIFFERENCE", "RELATED"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
#MainMenu, footer, header [data-testid="stToolbar"], .stDeployButton,
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: #0B0B0B; }
.stApp { background: #0B0B0B; color: #E8E8E8;
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.block-container { max-width: 1200px; padding: 1.2rem 2rem 3rem; }
.brand { font-size: 20px; font-weight: 700; letter-spacing: 0.22em; color: #E8E8E8; }
.brand-sub { font-size: 12px; letter-spacing: 0.08em; color: #8A8A8A; margin-top: 2px; }
.section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.18em;
  color: #8A8A8A; margin: 22px 0 8px; }
.hr { border-top: 1px solid #202020; margin: 4px 0 12px; }
div.stButton > button { background: #161616; color: #E8E8E8; border: 1px solid #2A2A2A;
  border-radius: 3px; font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
  padding: 6px 14px; }
div.stButton > button:hover { border-color: #D0D500; color: #D0D500; }
div.stButton > button[kind="primary"] { background: #D0D500; color: #111111;
  border: 1px solid #D0D500; }
div.stButton > button[kind="primary"]:hover { color: #111111; }
.fact { background: #111111; border: 1px solid #202020; border-left: 2px solid #3A3A3A;
  border-radius: 3px; padding: 12px 14px; margin-bottom: 10px; }
.fact-subj { font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: #8A8A8A; }
.fact-pred { font-size: 14px; font-weight: 600; color: #E8E8E8; margin-top: 2px; }
.fact-value { font-size: 22px; font-weight: 700; color: #E8E8E8; margin: 6px 0 2px; }
.fact-norm { font-size: 12.5px; color: #8A8A8A; }
.fact-norm b { color: #D0D500; font-weight: 600; }
.fact-meta { font-size: 12px; color: #5F5F5F; margin-top: 6px; }
.fact-meta b { color: #8A8A8A; font-weight: 600; }
.rel { background: #111111; border: 1px solid #202020; border-radius: 3px;
  padding: 12px 14px; margin-bottom: 10px; }
.rel-contra { border-left: 2px solid #FF4B3F; }
.rel-corrob { border-left: 2px solid #3A3A3A; }
.rel-ctx { border-left: 2px dashed #8A8A8A; }
.badge { display: inline-block; font-size: 10.5px; font-weight: 700;
  letter-spacing: 0.12em; padding: 2px 8px; border: 1px solid #2A2A2A;
  border-radius: 2px; color: #E8E8E8; }
.badge-contra { color: #FF4B3F; border-color: #FF4B3F; }
.badge-review { color: #D0D500; border-color: #D0D500; }
.badge-dim { color: #8A8A8A; }
.conf { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8A8A8A; }
.expl { font-size: 13px; color: #B5B5B5; line-height: 1.55; margin-top: 8px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.pair-cell { background: #161616; border: 1px solid #202020; border-radius: 2px;
  padding: 8px 10px; }
.pair-val { font-size: 16px; font-weight: 700; color: #E8E8E8; }
.pair-doc { font-size: 11.5px; color: #8A8A8A; margin-top: 2px; }
.ev { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: #8A8A8A;
  background: #0B0B0B; border: 1px solid #202020; border-radius: 2px;
  padding: 8px 10px; margin-top: 6px; white-space: pre-wrap; }
.prov { font-size: 12px; color: #8A8A8A; margin-top: 6px; }
.status-ok { color: #9AA29A; } .status-warn { color: #D0D500; }
.status-err { color: #FF4B3F; } .status-mut { color: #5F5F5F; }
div[data-testid="stExpander"] { background: #111111; border: 1px solid #202020;
  border-radius: 3px; }
div[data-testid="stExpander"] summary { color: #8A8A8A; font-size: 12px;
  font-weight: 600; letter-spacing: 0.08em; }
input, textarea { color: #E8E8E8 !important; }
.hero { text-align: center; padding: 8vh 0 4vh; }
.hero-title { font-size: 30px; font-weight: 700; letter-spacing: 0.3em;
  text-indent: 0.3em; }
.hero-sub { font-size: 13px; color: #8A8A8A; letter-spacing: 0.06em; margin-top: 6px; }
.hero-box { max-width: 560px; margin: 26px auto 0; background: #111111;
  border: 1px solid #202020; border-radius: 4px; padding: 22px 24px; text-align: left; }
.hero-box-title { font-size: 12px; font-weight: 600; letter-spacing: 0.14em;
  color: #E8E8E8; }
</style>
"""

# ---------------------------------------------------------------- helpers


def _init_state() -> None:
    defaults = {
        "active_tab": "KNOWLEDGE",
        "selected_doc_id": None,
        "selected_fact_id": None,
        "selected_rel_id": None,
        "query": "",
        "rel_type_filter": "All",
        "rel_min_conf": 0.0,
        "doc_ids": [],
        "doc_sizes": {},
        "show_uploader": False,
        "uploader_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _fmt_size(num_bytes) -> str:
    if num_bytes is None:
        return "—"
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "—"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size:.0f} B"


def _fmt_time(value) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%b %d %H:%M")
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
    label = f"PDF page {pdf_page}"
    if evidence.get("source_page_number"):
        label += f" (printed p. {evidence['source_page_number']})"
    return label


def _fact_page_label(fact: dict, evidence_by_id: dict) -> str:
    for eid in fact.get("evidence_ids", []) or []:
        ev = evidence_by_id.get(str(eid))
        if ev is not None:
            return _page_label(ev)
    return "page unknown"


def _status_class(status: str) -> str:
    if status == "COMPLETED":
        return "status-ok"
    if status == "PARTIAL":
        return "status-warn"
    if status == "FAILED":
        return "status-err"
    return "status-mut"


def _load_documents() -> tuple[list[dict], str | None, bool]:
    """Return (documents, error, list_supported). Falls back to the session
    registry of uploaded IDs when the backend predates GET /documents."""
    docs, err = api_client.list_documents()
    if docs is not None:
        for doc in docs:
            doc_id = str(doc.get("id"))
            if doc_id and doc_id not in st.session_state.doc_ids:
                st.session_state.doc_ids.append(doc_id)
        return docs, None, True
    if err == "NOT_SUPPORTED":
        fallback: list[dict] = []
        for doc_id in list(st.session_state.doc_ids):
            bundle, _ = api_client.get_bundle(doc_id)
            if bundle is not None:
                fallback.append(bundle["document"])
        return fallback, None, False
    return [], err, True


def _aggregate_facts(documents: list[dict], limit_per_doc: int = 50) -> tuple[list[dict], dict, str | None]:
    """Aggregate persisted facts across documents. Returns (facts, doc_names, error)."""
    facts: list[dict] = []
    doc_names = {str(d.get("id")): d.get("filename", "—") for d in documents}
    first_err = None
    for doc in documents:
        doc_id = str(doc.get("id"))
        items, err = api_client.list_facts(doc_id)
        if err is not None:
            first_err = first_err or err
            continue
        facts.extend(items[:limit_per_doc])
    facts.sort(key=lambda f: str(f.get("id")))
    return facts, doc_names, first_err


def _aggregate_relationships(
    documents: list[dict], rel_type: str | None, min_conf: float
) -> tuple[list[dict], dict, str | None]:
    rels: dict[str, dict] = {}
    doc_names = {str(d.get("id")): d.get("filename", "—") for d in documents}
    first_err = None
    for doc in documents:
        items, err = api_client.list_relationships(str(doc.get("id")), rel_type, min_conf)
        if err is not None:
            first_err = first_err or err
            continue
        for rel in items:
            rels[str(rel.get("id"))] = rel
    ordered = sorted(rels.values(), key=lambda r: (-float(r.get("confidence", 0) or 0), str(r.get("id"))))
    return ordered, doc_names, first_err


# ---------------------------------------------------------------- render


def _render_header(has_docs: bool) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="brand">ALETHEIA</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">Fact Knowledge Layer</div>', unsafe_allow_html=True)
    with right:
        cols = st.columns([1, 1, 1])
        with cols[1]:
            if st.button("UPLOAD PDFS", use_container_width=True):
                st.session_state.show_uploader = not st.session_state.show_uploader
                st.rerun()
        with cols[2]:
            if st.button("REFRESH", use_container_width=True):
                for key in ("selected_doc_id", "selected_fact_id", "selected_rel_id",
                            "query", "rel_type_filter", "show_uploader"):
                    defaults = {"selected_doc_id": None, "selected_fact_id": None,
                                "selected_rel_id": None, "query": "",
                                "rel_type_filter": "All", "show_uploader": False}
                    st.session_state[key] = defaults[key]
                st.session_state.rel_min_conf = 0.0
                st.cache_data.clear()
                st.rerun()
    if has_docs and st.session_state.show_uploader:
        _render_compact_uploader()


def _handle_uploads(files) -> None:
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
            if doc_id not in st.session_state.doc_ids:
                st.session_state.doc_ids.append(doc_id)
            st.session_state.doc_sizes[doc_id] = len(data)
        status = doc.get("ingestion_status", "—")
        st.success(f"{uploaded.name}: {status} "
                   f"({(result or {}).get('page_count', '?')} pages, "
                   f"{(result or {}).get('evidence_count', '?')} evidence units)")
    st.session_state.uploader_nonce += 1
    st.session_state.show_uploader = False
    st.rerun()


def _render_compact_uploader() -> None:
    with st.expander("Upload PDFs", expanded=True):
        files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_nonce}",
            label_visibility="collapsed",
        )
        if files:
            if st.button("INGEST", type="primary"):
                _handle_uploads(files)


def _render_empty_state() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">ALETHEIA</div>'
        '<div class="hero-sub">Fact Knowledge Layer</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-box"><div class="hero-box-title">UPLOAD PDF FILES</div>'
        '<div class="fact-meta">One or more PDFs. Each file is ingested into '
        "grounded evidence before any facts are extracted.</div></div>",
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"hero_uploader_{st.session_state.uploader_nonce}",
        label_visibility="collapsed",
    )
    if files and st.button("INGEST", type="primary"):
        _handle_uploads(files)
    health, err = api_client.health()
    if err is not None:
        st.error(err)
    elif health is not None and not (health.get("database", {}) or {}).get("reachable", True):
        st.warning(f"Backend is up but the database is unreachable: "
                   f"{(health.get('database', {}) or {}).get('detail', 'unknown')}")


def _render_documents_section(documents: list[dict]) -> list[dict]:
    st.markdown('<div class="section-label">DOCUMENTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if not documents:
        st.info("No documents ingested yet.")
        return documents
    rows = []
    for doc in documents:
        doc_id = str(doc.get("id"))
        rows.append({
            "Filename": doc.get("filename", "—"),
            "Size": _fmt_size(st.session_state.doc_sizes.get(doc_id)),
            "Pages": doc.get("page_count", "—"),
            "Uploaded": _fmt_time(doc.get("created_at")),
            "Status": doc.get("ingestion_status", "—"),
            "_id": doc_id,
        })
    st.dataframe(
        [{k: r[k] for k in ("Filename", "Size", "Pages", "Uploaded", "Status")} for r in rows],
        use_container_width=True,
        hide_index=True,
    )
    options = ["All documents"] + [r["Filename"] for r in rows]
    current = st.session_state.selected_doc_id
    try:
        default_idx = 0 if current is None else [r["_id"] for r in rows].index(current) + 1
    except ValueError:
        default_idx = 0
    choice = st.selectbox("Filter to document", options, index=default_idx,
                          label_visibility="collapsed")
    if choice == "All documents":
        st.session_state.selected_doc_id = None
        return documents
    picked = next(r for r in rows if r["Filename"] == choice)
    st.session_state.selected_doc_id = picked["_id"]
    return [d for d in documents if str(d.get("id")) == picked["_id"]]


def _render_tabs() -> None:
    labels = ["KNOWLEDGE", "RELATIONSHIPS", "DOCUMENTS"]
    active = st.session_state.active_tab
    cols = st.columns(3)
    for col, label in zip(cols, labels):
        with col:
            btn_type = "primary" if active == label else "secondary"
            if st.button(label, key=f"tab_{label}", type=btn_type, use_container_width=True):
                if st.session_state.active_tab != label:
                    st.session_state.active_tab = label
                    st.session_state.selected_fact_id = None
                    st.session_state.selected_rel_id = None
                    st.rerun()
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


def _render_fact_block(fact: dict, doc_name: str, evidence_by_id: dict, key: str) -> None:
    norm = _norm_text(fact)
    norm_html = f'<div class="fact-norm">Normalized: <b>{norm}</b></div>' if norm else ""
    scope = fact.get("scope_text") or ""
    scope_html = f" · {scope}" if scope else ""
    geo = fact.get("geography") or ""
    geo_html = f" · {geo}" if geo else ""
    selected = st.session_state.selected_fact_id == str(fact.get("id"))
    border = f' style="border-left-color: {ACCENT};"' if selected else ""
    st.markdown(
        f'<div class="fact"{border}>'
        f'<div class="fact-subj">{fact.get("subject", "—")}</div>'
        f'<div class="fact-pred">{fact.get("predicate", "—")}</div>'
        f'<div class="fact-value">{fact.get("value_text", "—")}</div>'
        f"{norm_html}"
        f'<div class="fact-meta"><b>{fact.get("time_text") or "time unknown"}</b>'
        f"{scope_html}{geo_html} · Confidence: <b>{_fmt_conf(fact.get('extraction_confidence'))}</b>"
        f" · {doc_name} · {_fact_page_label(fact, evidence_by_id)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 4])
    with cols[0]:
        if st.button("EVIDENCE", key=f"ev_{key}_{fact.get('id')}", use_container_width=True):
            st.session_state.selected_fact_id = str(fact.get("id"))
            st.rerun()
    if selected:
        _render_fact_evidence(fact, doc_name, evidence_by_id)


def _render_fact_evidence(fact: dict, doc_name: str, evidence_by_id: dict) -> None:
    with st.expander("FACT DETAIL / PROVENANCE", expanded=True):
        st.markdown(
            f'<div class="prov">FACT → SOURCE EVIDENCE → SOURCE DOCUMENT → PDF PAGE</div>',
            unsafe_allow_html=True,
        )
        detail = {
            "fact_id": str(fact.get("id")),
            "document": doc_name,
            "status": fact.get("status"),
            "value_kind": fact.get("value_kind"),
            "unit": fact.get("unit"),
            "estimate_status": fact.get("estimate_status"),
            "ambiguity_flags": fact.get("ambiguity_flags") or [],
            "extraction_confidence": fact.get("extraction_confidence"),
        }
        st.json(detail)
        eids = fact.get("evidence_ids", []) or []
        if not eids:
            st.warning("No evidence IDs linked to this fact.")
            return
        bundle, err = api_client.get_bundle(str(fact.get("document_id")))
        full_evidence = {}
        if bundle is not None:
            full_evidence = {str(e.get("id")): e for e in bundle.get("evidence", [])}
        full_evidence.update(evidence_by_id)
        for eid in eids:
            ev = full_evidence.get(str(eid))
            if ev is None:
                st.markdown(f'<div class="ev">evidence {eid}: unavailable</div>',
                            unsafe_allow_html=True)
                continue
            bbox = ev.get("bbox")
            st.markdown(
                f'<div class="ev">{_page_label(ev)} · {ev.get("type", "?")} · '
                f'{ev.get("extraction_method", "?")} · '
                f'quality {ev.get("extraction_quality", "—")} · '
                f'bbox {bbox if bbox else "—"}\n{(ev.get("text") or "")[:2000]}'
                f"\n-- evidence_id {ev.get('id')}</div>",
                unsafe_allow_html=True,
            )


def _render_knowledge(documents: list[dict]) -> None:
    query = st.text_input(
        "Search facts, entities, metrics...",
        value=st.session_state.query,
        label_visibility="collapsed",
        placeholder="Search facts, entities, metrics...",
    )
    st.session_state.query = query
    doc_filter = st.session_state.selected_doc_id

    if query.strip():
        with st.spinner("Searching knowledge layer ..."):
            result, err = api_client.search(query.strip(), doc_filter, limit=30)
        if err is not None:
            st.error(f"Search failed: {err}")
            return
        hits = (result or {}).get("hits", []) or []
        if not hits:
            st.info("No facts match this query.")
            return
        st.markdown(f'<div class="fact-meta">{len(hits)} result(s)</div>',
                    unsafe_allow_html=True)
        for hit in hits:
            fact = hit.get("fact", {})
            doc = hit.get("document") or {}
            ev_by_id = {str(e.get("id")): e for e in hit.get("evidence", []) or []}
            st.markdown(
                f'<div class="fact-meta">score {hit.get("score", 0):.2f} · '
                f'{hit.get("match_kind", "?")}</div>',
                unsafe_allow_html=True,
            )
            _render_fact_block(fact, doc.get("filename", "—"), ev_by_id,
                               key=f"search_{hit.get('score', 0):.3f}")
        return

    with st.spinner("Loading facts ..."):
        facts, doc_names, err = _aggregate_facts(documents)
    if doc_filter:
        facts = [f for f in facts if str(f.get("document_id")) == doc_filter]
    if err is not None and not facts:
        st.error(f"Could not load facts: {err}")
        return
    if not facts:
        st.info("No persisted facts yet for these documents. "
                "Fact extraction runs explicitly per document via the API.")
        return
    # Evidence lookup for page labels: resolve from search-free path via
    # one bundle fetch per document (bounded, demo scale).
    ev_by_id: dict = {}
    for doc in documents:
        if doc_filter and str(doc.get("id")) != doc_filter:
            continue
        bundle, _ = api_client.get_bundle(str(doc.get("id")))
        if bundle is not None:
            for ev in bundle.get("evidence", []) or []:
                ev_by_id[str(ev.get("id"))] = ev
    st.markdown(f'<div class="fact-meta">{len(facts)} fact(s)</div>', unsafe_allow_html=True)
    for fact in facts[:100]:
        _render_fact_block(fact, doc_names.get(str(fact.get("document_id")), "—"),
                           ev_by_id, key="all")


def _rel_badge(rel: dict) -> str:
    rtype = rel.get("relationship_type", "?")
    cls = "badge-contra" if rtype == "CONTRADICTS" else ("badge-dim")
    review = ""
    if rel.get("status") == "NEEDS_REVIEW":
        review = ' <span class="badge badge-review">NEEDS REVIEW</span>'
    return (f'<span class="badge {cls}">{rtype}</span>{review} '
            f'<span class="conf">{_fmt_conf(rel.get("confidence"))}</span>')


def _fact_summary(fact: dict, doc_names: dict) -> str:
    return (f'<div class="fact-subj">{fact.get("subject", "—")}</div>'
            f'<div class="pair-val">{fact.get("value_text", "—")}</div>'
            f'<div class="pair-doc">{fact.get("predicate", "—")} · '
            f'{doc_names.get(str(fact.get("document_id")), "—")}</div>')


def _render_relationships(documents: list[dict]) -> None:
    cols = st.columns([2, 1])
    with cols[0]:
        options = ["All"] + REL_TYPES
        selfilter = st.selectbox("Relationship type", options,
                                 index=options.index(st.session_state.rel_type_filter)
                                 if st.session_state.rel_type_filter in options else 0)
        st.session_state.rel_type_filter = selfilter
    with cols[1]:
        min_conf = st.slider("Min confidence", 0.0, 1.0,
                             float(st.session_state.rel_min_conf), 0.05)
        st.session_state.rel_min_conf = min_conf
    rel_type = None if selfilter == "All" else selfilter
    doc_filter = st.session_state.selected_doc_id
    visible = [d for d in documents
               if doc_filter is None or str(d.get("id")) == doc_filter]
    with st.spinner("Loading relationships ..."):
        rels, doc_names, err = _aggregate_relationships(visible, rel_type, min_conf)
    if err is not None and not rels:
        st.error(f"Could not load relationships: {err}")
        return
    if not rels:
        st.info("No persisted relationships for this filter. "
                "Relationship reasoning runs explicitly per document via the API.")
        return
    st.markdown(f'<div class="fact-meta">{len(rels)} relationship(s)</div>',
                unsafe_allow_html=True)
    for rel in rels[:100]:
        rtype = rel.get("relationship_type", "")
        css_class = "rel rel-contra" if rtype == "CONTRADICTS" else (
            "rel rel-ctx" if rtype == "CONTEXTUAL_DIFFERENCE" else "rel rel-corrob")
        st.markdown(f'<div class="{css_class}">{_rel_badge(rel)}', unsafe_allow_html=True)
        detail, derr = api_client.get_relationship_detail(str(rel.get("id")))
        if derr is not None or detail is None:
            st.warning(f"Could not expand relationship {rel.get('id')}: {derr}")
            st.markdown("</div>", unsafe_allow_html=True)
            continue
        fact_a, fact_b = detail.get("fact_a", {}), detail.get("fact_b", {})
        st.markdown(
            f'<div class="pair"><div class="pair-cell">{_fact_summary(fact_a, doc_names)}</div>'
            f'<div class="pair-cell">{_fact_summary(fact_b, doc_names)}</div></div>',
            unsafe_allow_html=True,
        )
        if rtype == "CONTRADICTS":
            st.markdown(
                '<div class="expl">Same entity, same metric, materially disagreeing '
                "values. Neither source is judged correct.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="expl">{rel.get("explanation", "")}</div>',
                    unsafe_allow_html=True)
        if st.button("VIEW EVIDENCE", key=f"rel_{rel.get('id')}",
                     use_container_width=False):
            st.session_state.selected_rel_id = str(rel.get("id"))
            st.rerun()
        if st.session_state.selected_rel_id == str(rel.get("id")):
            with st.expander("RELATIONSHIP EVIDENCE", expanded=True):
                for side, fact, evs in (("A", fact_a, detail.get("evidence_a", []) or []),
                                        ("B", fact_b, detail.get("evidence_b", []) or [])):
                    ev_map = {str(e.get("id")): e for e in evs}
                    page = _fact_page_label(fact, ev_map) if evs else "page unknown"
                    subj = fact.get("subject", "—")
                    val = fact.get("value_text", "—")
                    st.markdown(
                        f"<div class='prov'>Side {side}: {subj} — {val} ({page})</div>",
                        unsafe_allow_html=True)
                    if not evs:
                        st.markdown('<div class="ev">no evidence returned</div>',
                                    unsafe_allow_html=True)
                    for ev in evs:
                        st.markdown(
                            f'<div class="ev">{_page_label(ev)} · {ev.get("type", "?")} · '
                            f'{ev.get("extraction_method", "?")}\n{(ev.get("text") or "")[:2000]}'
                            f"\n-- evidence_id {ev.get('id')}</div>",
                            unsafe_allow_html=True,
                        )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_document_detail(documents: list[dict]) -> None:
    if not documents:
        st.info("No documents ingested yet.")
        return
    doc = documents[0]
    if st.session_state.selected_doc_id:
        match = [d for d in documents if str(d.get("id")) == st.session_state.selected_doc_id]
        if match:
            doc = match[0]
    doc_id = str(doc.get("id"))
    bundle, err = api_client.get_bundle(doc_id)
    if err is not None or bundle is None:
        st.error(f"Could not load document: {err}")
        return
    document = bundle.get("document", {})
    pages = bundle.get("pages", []) or []
    evidence = bundle.get("evidence", []) or []
    facts, ferr = api_client.list_facts(doc_id)
    rels, rerr = api_client.list_relationships(doc_id)
    st.markdown(f'<div class="fact-pred">{document.get("filename", "—")}</div>',
                unsafe_allow_html=True)
    meta_cols = st.columns(4)
    meta_cols[0].markdown(f'<div class="fact-meta">Size<br><b>{_fmt_size(st.session_state.doc_sizes.get(doc_id))}</b></div>',
                         unsafe_allow_html=True)
    meta_cols[1].markdown(f'<div class="fact-meta">Pages<br><b>{document.get("page_count", len(pages))}</b></div>',
                         unsafe_allow_html=True)
    meta_cols[2].markdown(f'<div class="fact-meta">Status<br><b>{document.get("ingestion_status", "—")}</b></div>',
                         unsafe_allow_html=True)
    meta_cols[3].markdown(f'<div class="fact-meta">Uploaded<br><b>{_fmt_time(document.get("created_at"))}</b></div>',
                         unsafe_allow_html=True)
    counts = st.columns(3)
    counts[0].markdown(f'<div class="fact-meta">Evidence units<br><b>{len(evidence)}</b></div>',
                       unsafe_allow_html=True)
    counts[1].markdown(f'<div class="fact-meta">Facts<br><b>{len(facts) if facts is not None else f"unavailable ({ferr})"}</b></div>',
                       unsafe_allow_html=True)
    counts[2].markdown(f'<div class="fact-meta">Relationships<br><b>{len(rels) if rels is not None else f"unavailable ({rerr})"}</b></div>',
                       unsafe_allow_html=True)
    if document.get("error"):
        st.error(f"Processing error: {document['error']}")
    bad = [p for p in pages if p.get("error")]
    if bad:
        st.warning(f"{len(bad)} page(s) with extraction errors.")
    if pages:
        st.markdown('<div class="section-label">PAGES</div>', unsafe_allow_html=True)
        st.dataframe(
            [{"PDF page": int(p.get("pdf_page_number", 0)) + 1,
              "Printed": p.get("source_page_number") or "—",
              "Verdict": p.get("quality_verdict", "—"),
              "Quality": round(float(p.get("extraction_quality", 0) or 0), 2),
              "Chars": p.get("char_count", 0)} for p in pages],
            use_container_width=True,
            hide_index=True,
        )
    if evidence:
        st.markdown('<div class="section-label">EVIDENCE SAMPLE (first 10)</div>',
                    unsafe_allow_html=True)
        for ev in evidence[:10]:
            st.markdown(
                f'<div class="ev">{_page_label(ev)} · {ev.get("type", "?")} · '
                f'{ev.get("extraction_method", "?")}\n{(ev.get("text") or "")[:1200]}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="Aletheia — Fact Knowledge Layer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()

    documents, err, _supported = _load_documents()
    if err is not None:
        st.markdown('<div class="brand">ALETHEIA</div>', unsafe_allow_html=True)
        st.error(err)
        st.info("Start the backend (docker compose up --build), then press Refresh.")
        if st.button("REFRESH"):
            st.rerun()
        return

    if not documents:
        _render_empty_state()
        return

    _render_header(has_docs=True)
    visible_docs = _render_documents_section(documents)
    _render_tabs()
    if st.session_state.active_tab == "KNOWLEDGE":
        _render_knowledge(visible_docs if st.session_state.selected_doc_id else documents)
    elif st.session_state.active_tab == "RELATIONSHIPS":
        _render_relationships(documents)
    else:
        _render_document_detail(visible_docs)


if __name__ == "__main__":
    main()
