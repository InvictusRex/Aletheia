"""Aletheia - Fact Knowledge Layer (Streamlit frontend, single page).

Client of the real FastAPI backend. Fully data-driven: every document,
fact, relationship, and evidence block rendered here comes from an API
response. Empty backend state renders as an empty state, never as
placeholder content.
"""

from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

import api as api_client

# ---------------------------------------------------------------- constants

ACCENT = "#D0D500"
ERROR = "#FF4B3F"
GREEN = "#6FCF97"

REL_TYPES = ["CORROBORATES", "CONTRADICTS", "CONTEXTUAL_DIFFERENCE", "RELATED"]

# Minimal, non-invasive CSS. Only our own classes plus safe container and
# color adjustments. Nothing hides or restructures Streamlit's internal
# component DOM (FileUploader, Selectbox, TextInput must keep working).
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
  color: #E8E8E8; margin: 18px 0 6px; }
.hr { border-top: 1px solid #202020; margin: 2px 0 10px; }
.tbl { width: 100%; border-collapse: collapse; font-size: 14px; }
.tbl th { text-align: left; font-size: 11.5px; font-weight: 600;
  letter-spacing: 0.12em; color: #8A8A8A; padding: 6px 10px;
  border-bottom: 1px solid #202020; }
.tbl td { padding: 8px 10px; border-bottom: 1px solid #161616; color: #E8E8E8;
  vertical-align: top; }
.tbl tr:last-child td { border-bottom: none; }
.doc-name { font-size: 14.5px; font-weight: 600; }
.st-ok { color: #6FCF97; } .st-warn { color: #D0D500; } .st-err { color: #FF4B3F; } .st-mut { color: #5F5F5F; }
.fact { background: #111111; border: 1px solid #202020; border-left: 2px solid #3A3A3A;
  border-radius: 3px; padding: 12px 16px; margin-bottom: 10px; }
.fact-subj { font-size: 12.5px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: #8A8A8A; }
.fact-pred { font-size: 17px; font-weight: 600; margin-top: 2px; }
.fact-value { font-size: 25px; font-weight: 700; margin: 6px 0 2px; }
.fact-norm-label { font-size: 11.5px; letter-spacing: 0.12em; color: #5F5F5F;
  margin-top: 6px; }
.fact-norm { font-size: 13.5px; color: #B5B5B5; }
.fact-norm b { color: #D0D500; font-weight: 600; }
.fact-meta { font-size: 12.5px; color: #5F5F5F; margin-top: 8px; }
.fact-meta b { color: #8A8A8A; font-weight: 600; }
.rel { background: #111111; border: 1px solid #202020; border-radius: 3px;
  padding: 12px 16px; margin-bottom: 10px; }
.rel-contra { border-left: 2px solid #FF4B3F; }
.rel-corrob { border-left: 2px solid #2E5C43; }
.rel-ctx { border-left: 2px dashed #8A8A8A; }
.badge { display: inline-block; font-size: 11px; font-weight: 700;
  letter-spacing: 0.12em; padding: 2px 8px; border: 1px solid #2A2A2A;
  border-radius: 2px; }
.badge-contra { color: #FF4B3F; border-color: #FF4B3F; }
.badge-review { color: #D0D500; border-color: #D0D500; }
.badge-grn { color: #6FCF97; border-color: #6FCF97; }
.badge-dim { color: #8A8A8A; }
.conf { font-size: 12.5px; color: #8A8A8A; }
.expl { font-size: 13.5px; color: #B5B5B5; line-height: 1.55; margin-top: 8px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.pair-cell { background: #161616; border: 1px solid #202020; border-radius: 2px;
  padding: 10px 12px; }
.pair-val { font-size: 17px; font-weight: 700; }
.pair-doc { font-size: 12.5px; color: #8A8A8A; margin-top: 3px; }
.ev { font-size: 12.5px; color: #B5B5B5; background: #0B0B0B;
  border: 1px solid #202020; border-radius: 2px; padding: 8px 10px;
  margin-top: 6px; white-space: pre-wrap; font-family: monospace; }
.prov { font-size: 12.5px; color: #8A8A8A; margin-top: 8px;
  letter-spacing: 0.06em; }
div.stButton > button { background: #161616; color: #E8E8E8;
  border: 1px solid #2A2A2A; border-radius: 3px; font-size: 13px;
  font-weight: 600; letter-spacing: 0.08em; padding: 7px 16px; }
div.stButton > button:hover { border-color: #D0D500; color: #D0D500; }
div.stButton > button[kind="primary"] { background: #D0D500; color: #111111;
  border: 1px solid #D0D500; }
.upload-box { max-width: 600px; margin: 24px auto 0; background: #111111;
  border: 1px solid #202020; border-radius: 4px; padding: 24px 26px; }
.upload-title { font-size: 15px; font-weight: 600; letter-spacing: 0.14em; }
.hero { text-align: center; padding: 6vh 0 0; }
.hero-title { font-size: 32px; font-weight: 700; letter-spacing: 0.3em;
  text-indent: 0.3em; }
.hero-sub { font-size: 13px; color: #8A8A8A; letter-spacing: 0.08em; margin-top: 6px; }
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
        "entered": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


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


def _status_span(status: str) -> str:
    cls = {"COMPLETED": "st-ok", "PARTIAL": "st-warn", "FAILED": "st-err"}.get(status, "")
    return f'<span class="{cls}">{_esc(status)}</span>' if cls else _esc(status)


def _load_documents() -> tuple[list[dict], str | None]:
    """GET /documents. Falls back to per-id bundle fetches for the session
    registry when the backend predates the list endpoint."""
    docs, err = api_client.list_documents()
    if docs is not None:
        for doc in docs:
            doc_id = str(doc.get("id"))
            if doc_id and doc_id not in st.session_state.doc_ids:
                st.session_state.doc_ids.append(doc_id)
        return docs, None
    if err == "NOT_SUPPORTED":
        fallback: list[dict] = []
        for doc_id in list(st.session_state.doc_ids):
            bundle, _ = api_client.get_bundle(doc_id)
            if bundle is not None:
                fallback.append(bundle["document"])
        return fallback, None
    return [], err


# ---------------------------------------------------------------- upload


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
        st.success(f"{uploaded.name}: {doc.get('ingestion_status', '—')} "
                   f"({(result or {}).get('page_count', '?')} pages, "
                   f"{(result or {}).get('evidence_count', '?')} evidence units)")
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


# ---------------------------------------------------------------- chrome


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
            # UI reset only: clears the page back to the upload screen.
            # Backend documents are never touched.
            st.session_state.selected_doc_id = None
            st.session_state.selected_fact_id = None
            st.session_state.selected_rel_id = None
            st.session_state.query = ""
            st.session_state.rel_type_filter = "All"
            st.session_state.rel_min_conf = 0.0
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


def _render_doc_filter(documents: list[dict]) -> list[dict]:
    """Document scope picker. Always rendered at the top of tab content so
    the dropdown menu opens downward into free space."""
    names = [str(d.get("filename", "—")) for d in documents]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    options = ["All documents"]
    opt_ids: list[str | None] = [None]
    for doc, name in zip(documents, names):
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            options.append(f"{name} · {str(doc.get('id'))[:8]}")
        else:
            options.append(name)
        opt_ids.append(str(doc.get("id")))
    current = st.session_state.selected_doc_id
    try:
        default_idx = 0 if current is None else opt_ids.index(current)
    except ValueError:
        default_idx = 0
    choice = st.selectbox("Document", options, index=default_idx, key="doc_filter")
    picked_id = opt_ids[options.index(choice)]
    if picked_id is None:
        st.session_state.selected_doc_id = None
        return documents
    st.session_state.selected_doc_id = picked_id
    return [d for d in documents if str(d.get("id")) == picked_id]


def _render_documents_table(documents: list[dict]) -> None:
    cells = []
    for doc in documents:
        doc_id = str(doc.get("id"))
        cells.append(
            f"<tr><td><span class='doc-name'>{_esc(doc.get('filename', '—'))}</span></td>"
            f"<td>{_esc(_fmt_size(st.session_state.doc_sizes.get(doc_id)))}</td>"
            f"<td>{_esc(doc.get('page_count', '—'))}</td>"
            f"<td>{_esc(_fmt_time(doc.get('created_at')))}</td>"
            f"<td>{_status_span(doc.get('ingestion_status', '—'))}</td></tr>"
        )
    st.markdown(
        "<table class='tbl'><thead><tr><th>FILENAME</th><th>SIZE</th><th>PAGES</th>"
        "<th>UPLOADED</th><th>STATUS</th></tr></thead><tbody>"
        + "".join(cells) + "</tbody></table>",
        unsafe_allow_html=True,
    )


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


# ---------------------------------------------------------------- knowledge


def _render_fact_block(fact: dict, doc_name: str, evidence_by_id: dict, key: str) -> None:
    norm = _norm_text(fact)
    norm_html = (f'<div class="fact-norm-label">NORMALIZED</div>'
                 f'<div class="fact-norm"><b>{_esc(norm)}</b></div>') if norm else ""
    scope = fact.get("scope_text") or ""
    scope_html = f" · {_esc(scope)}" if scope else ""
    geo = fact.get("geography") or ""
    geo_html = f" · {_esc(geo)}" if geo else ""
    selected = st.session_state.selected_fact_id == str(fact.get("id"))
    border = f' style="border-left-color: {ACCENT};"' if selected else ""
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
        bundle, _ = api_client.get_bundle(str(fact.get("document_id")))
        full = dict(evidence_by_id)
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


def _render_knowledge(documents: list[dict]) -> None:
    documents = _render_doc_filter(documents)
    query = st.text_input(
        "Search facts, entities, metrics",
        value=st.session_state.query,
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
        doc_names = {str(d.get("id")): str(d.get("filename", "—")) for d in documents}
        first_err = None
        for doc in documents:
            items, err = api_client.list_facts(str(doc.get("id")))
            if err is not None:
                first_err = first_err or err
                continue
            facts.extend((items or [])[:50])
        facts.sort(key=lambda f: str(f.get("id")))
    if doc_filter:
        facts = [f for f in facts if str(f.get("document_id")) == doc_filter]
    if first_err is not None and not facts:
        st.error(f"Could not load facts: {first_err}")
        return
    if not facts:
        st.info("No persisted facts yet for these documents. "
                "Fact extraction runs explicitly per document via the API.")
        return
    ev_by_id: dict = {}
    for doc in documents:
        if doc_filter and str(doc.get("id")) != doc_filter:
            continue
        bundle, _ = api_client.get_bundle(str(doc.get("id")))
        if bundle is not None:
            for ev in bundle.get("evidence", []) or []:
                ev_by_id[str(ev.get("id"))] = ev
    st.markdown(f"<div class='fact-meta'>{len(facts)} fact(s)</div>",
                unsafe_allow_html=True)
    for fact in facts[:100]:
        _render_fact_block(fact, doc_names.get(str(fact.get("document_id")), "—"),
                           ev_by_id, key="all")


# ---------------------------------------------------------------- relationships


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


def _render_relationships(documents: list[dict]) -> None:
    all_names = {str(d.get("id")): str(d.get("filename", "—")) for d in documents}
    documents = _render_doc_filter(documents)
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
    visible = documents
    with st.spinner("Loading relationships ..."):
        rels: dict[str, dict] = {}
        doc_names = dict(all_names)
        first_err = None
        for doc in visible:
            items, err = api_client.list_relationships(
                str(doc.get("id")), rel_type, min_conf)
            if err is not None:
                first_err = first_err or err
                continue
            for rel in items or []:
                rels[str(rel.get("id"))] = rel
        ordered = sorted(rels.values(),
                         key=lambda r: (-float(r.get("confidence", 0) or 0),
                                        str(r.get("id"))))
    if first_err is not None and not ordered:
        st.error(f"Could not load relationships: {first_err}")
        return
    if not ordered:
        st.info("No persisted relationships for this filter. "
                "Relationship reasoning runs explicitly per document via the API.")
        return
    st.markdown(f"<div class='fact-meta'>{len(ordered)} relationship(s)</div>",
                unsafe_allow_html=True)
    for rel in ordered[:100]:
        rtype = str(rel.get("relationship_type", ""))
        css_class = ("rel rel-contra" if rtype == "CONTRADICTS"
                     else "rel rel-ctx" if rtype == "CONTEXTUAL_DIFFERENCE"
                     else "rel rel-corrob")
        detail, derr = api_client.get_relationship_detail(str(rel.get("id")))
        if derr is not None or detail is None:
            st.warning(f"Could not expand relationship {rel.get('id')}: {derr}")
            continue
        fact_a, fact_b = detail.get("fact_a", {}), detail.get("fact_b", {})
        st.markdown(f"<div class='{css_class}'>{_rel_badge(rel)}", unsafe_allow_html=True)
        st.markdown(
            f"<div class='pair'><div class='pair-cell'>"
            f"{_fact_summary(fact_a, doc_names)}</div>"
            f"<div class='pair-cell'>{_fact_summary(fact_b, doc_names)}</div></div>",
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
            st.session_state.selected_rel_id = str(rel.get("id"))
            st.rerun()
        if st.session_state.selected_rel_id == str(rel.get("id")):
            with st.expander("RELATIONSHIP EVIDENCE", expanded=True):
                for side, fact, evs in (("A", fact_a, detail.get("evidence_a", []) or []),
                                        ("B", fact_b, detail.get("evidence_b", []) or [])):
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


# ---------------------------------------------------------------- documents tab


def _render_documents_tab(documents: list[dict]) -> None:
    st.markdown('<div class="section">DOCUMENTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    _render_documents_table(documents)
    st.markdown('<div class="section">INSPECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    _render_doc_filter(documents)
    _render_document_detail(documents)


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
    st.markdown(f"<div class='fact-pred'>{_esc(document.get('filename', '—'))}</div>",
                unsafe_allow_html=True)
    meta = st.columns(4)
    meta[0].markdown(f"<div class='fact-meta'>Size<br><b>"
                     f"{_esc(_fmt_size(st.session_state.doc_sizes.get(doc_id)))}</b></div>",
                     unsafe_allow_html=True)
    meta[1].markdown(f"<div class='fact-meta'>Pages<br><b>"
                     f"{_esc(document.get('page_count', len(pages)))}</b></div>",
                     unsafe_allow_html=True)
    meta[2].markdown(f"<div class='fact-meta'>Status<br><b>"
                     f"{_esc(document.get('ingestion_status', '—'))}</b></div>",
                     unsafe_allow_html=True)
    meta[3].markdown(f"<div class='fact-meta'>Uploaded<br><b>"
                     f"{_esc(_fmt_time(document.get('created_at')))}</b></div>",
                     unsafe_allow_html=True)
    counts = st.columns(3)
    counts[0].markdown(f"<div class='fact-meta'>Evidence units<br><b>{len(evidence)}</b></div>",
                       unsafe_allow_html=True)
    fact_count = len(facts) if facts is not None else f"unavailable ({_esc(ferr)})"
    rel_count = len(rels) if rels is not None else f"unavailable ({_esc(rerr)})"
    counts[1].markdown(f"<div class='fact-meta'>Facts<br><b>{fact_count}</b></div>",
                       unsafe_allow_html=True)
    counts[2].markdown(f"<div class='fact-meta'>Relationships<br><b>{rel_count}</b></div>",
                       unsafe_allow_html=True)
    if document.get("error"):
        st.error(f"Processing error: {document['error']}")
    bad = [p for p in pages if p.get("error")]
    if bad:
        st.warning(f"{len(bad)} page(s) with extraction errors.")
    if pages:
        st.markdown('<div class="section">PAGES</div>', unsafe_allow_html=True)
        rows = []
        for p in pages:
            try:
                pdf_page = int(p.get("pdf_page_number", 0)) + 1
            except (TypeError, ValueError):
                pdf_page = "—"
            try:
                quality = round(float(p.get("extraction_quality", 0) or 0), 2)
            except (TypeError, ValueError):
                quality = "—"
            rows.append(
                f"<tr><td>{pdf_page}</td><td>{_esc(p.get('source_page_number') or '—')}</td>"
                f"<td>{_esc(p.get('quality_verdict', '—'))}</td><td>{quality}</td>"
                f"<td>{_esc(p.get('char_count', 0))}</td></tr>"
            )
        st.markdown(
            "<table class='tbl'><thead><tr><th>PDF PAGE</th><th>PRINTED</th>"
            "<th>VERDICT</th><th>QUALITY</th><th>CHARACTERS</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>",
            unsafe_allow_html=True,
        )
    if evidence:
        st.markdown('<div class="section">EVIDENCE (first 10)</div>', unsafe_allow_html=True)
        for ev in evidence[:10]:
            st.markdown(
                f"<div class='ev'>{_esc(_page_label(ev))} · {_esc(ev.get('type', '?'))} · "
                f"{_esc(ev.get('extraction_method', '?'))}\n"
                f"{_esc((ev.get('text') or '')[:1200])}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="Aletheia — Fact Knowledge Layer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()

    documents, err = _load_documents()
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
    _render_tabs()
    if st.session_state.active_tab == "KNOWLEDGE":
        _render_knowledge(documents)
    elif st.session_state.active_tab == "RELATIONSHIPS":
        _render_relationships(documents)
    else:
        _render_documents_tab(documents)


if __name__ == "__main__":
    main()
