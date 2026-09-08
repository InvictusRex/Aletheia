"""S-TST: POST /search retrieval tests (lexical / vector / hybrid).

Conventions:
- No network, no API keys, no model downloads. The vector path is
  exercised with a duck-typed fake embedder passed directly to
  ``search_facts``; API-level tests run the lexical path (forced via
  ``FactEmbedder.available`` so they stay deterministic even if the
  optional sentence-transformers dependency is ever installed).
- Provenance tests persist real Document rows via ``run_ingestion`` +
  ``save_ingestion`` over in-memory PyMuPDF PDFs (same pattern as
  test_facts.py), then ``save_facts`` with evidence_ids pointing at real
  evidence rows. All synthetic wording is neutral.
"""

from __future__ import annotations

import re
import uuid
from uuid import uuid4

import pymupdf
import pytest
from sqlalchemy import func, select

import app.search.service as search_service
from app.core.config import settings
from app.db.models import (
    EvidenceUnitRow,
    FactEmbeddingRow,
    FactRow,
    RelationshipRow,
)
from app.db.repositories import (
    save_embedding,
    save_facts,
    save_ingestion,
    save_relationship,
)
from app.extraction.pipeline import run_ingestion
from app.matching.embeddings import cosine_similarity
from app.models import Fact, Relationship, RelationshipType, ValueKind
from app.search.service import hydrate_hit, search_facts, tokenize_query

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_pdf(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _ingest(db_session, filename: str, pdf_bytes: bytes):
    """Run ingestion and persist the bundle; return (document, pages, evidence)."""
    from app.models import IngestionStatus

    document, pages, evidence = run_ingestion(filename, pdf_bytes)
    assert document.ingestion_status == IngestionStatus.COMPLETED
    assert evidence, "precondition: ingestion must produce evidence"
    save_ingestion(db_session, document, pages, evidence)
    db_session.commit()
    return document, pages, evidence


def _ingest_two_docs(db_session):
    """Two persisted documents; returns (doc_a, ev_a, doc_b, ev_b)."""
    text_a = "Synthetic harbor ledger sentence with neutral wording. " * 25
    text_b = "Synthetic orchard ledger sentence with neutral wording. " * 25
    doc_a, _, ev_a = _ingest(db_session, "synthetic-a.pdf", _make_text_pdf([text_a]))
    doc_b, _, ev_b = _ingest(db_session, "synthetic-b.pdf", _make_text_pdf([text_b]))
    return doc_a, ev_a, doc_b, ev_b


def _fact(document_id, evidence_id, subject, **overrides):
    params = {
        "document_id": document_id,
        "subject": subject,
        "predicate": "reported total",
        "value_kind": ValueKind.NUMERIC,
        "value_text": "740",
        "value_number": 740.0,
        "evidence_ids": [evidence_id],
        "extraction_confidence": 0.8,
    }
    params.update(overrides)
    return Fact(**params)


def _save(db_session, facts):
    save_facts(db_session, facts)
    db_session.commit()
    return facts


def _expected_jaccard(query: str, fact: Fact) -> float:
    query_tokens = set(tokenize_query(query))
    fact_tokens = set(
        _TOKEN_RE.findall(
            " ".join(
                [fact.subject, fact.predicate, fact.value_text, fact.unit or ""]
            ).lower()
        )
    )
    return len(query_tokens & fact_tokens) / len(query_tokens | fact_tokens)


def _expected_vector_score(query_vec: list[float], stored_vec: list[float]) -> float:
    distance = 1.0 - cosine_similarity(query_vec, stored_vec)
    return max(0.0, 1.0 - distance)


class _FixedEmbedder:
    """Duck-typed stand-in for FactEmbedder: embed() returns a fixed vector."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = list(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def _table_counts(db_session) -> dict:
    out = {}
    for model, key in (
        (FactRow, "facts"),
        (EvidenceUnitRow, "evidence"),
        (RelationshipRow, "relationships"),
        (FactEmbeddingRow, "embeddings"),
    ):
        out[key] = db_session.scalar(select(func.count()).select_from(model))
    return out


@pytest.fixture()
def force_lexical(monkeypatch):
    """Force the lexical-only path inside search_facts (incl. via the API)."""
    from app.search.service import FactEmbedder

    monkeypatch.setattr(
        FactEmbedder,
        "available",
        staticmethod(lambda: (False, "forced lexical for tests")),
    )
    return True


# ---------------------------------------------------------------------------
# 1. Valid lexical search
# ---------------------------------------------------------------------------


def test_lexical_search_returns_match_with_score(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    target = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    _save(db_session, [target])

    # Service level, lexical path forced with embedder=None.
    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert hits, "expected at least one lexical hit"
    by_id = {fact.id: (score, kind) for fact, score, kind in hits}
    assert target.id in by_id
    score, kind = by_id[target.id]
    assert kind == "lexical"
    assert 0.0 < score <= 1.0
    assert score == pytest.approx(_expected_jaccard("harbor", target))

    # API level: same hit over HTTP.
    resp = api_client.post("/search", json={"query": "harbor"})
    assert resp.status_code == 200
    body = resp.json()
    api_by_id = {h["fact"]["id"]: h for h in body["hits"]}
    assert str(target.id) in api_by_id
    api_hit = api_by_id[str(target.id)]
    assert api_hit["match_kind"] == "lexical"
    assert 0.0 < api_hit["score"] <= 1.0


# ---------------------------------------------------------------------------
# 2. Vector path via injected fake embedder
# ---------------------------------------------------------------------------


def test_vector_path_with_fake_embedder(db_session):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    near = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    far = _fact(doc_a.id, ev_a[0].id, "Orchard depot tally")
    _save(db_session, [near, far])
    query_vec = [1.0, 0.0, 0.0]
    save_embedding(db_session, near.id, [1.0, 0.0, 0.0])
    save_embedding(db_session, far.id, [0.0, 1.0, 0.0])
    db_session.commit()

    # Query tokens share nothing with any fact field -> lexical pool empty.
    hits = search_facts(
        db_session, "quasar nebula drift", None, 20, 0.0,
        embedder=_FixedEmbedder(query_vec),
    )
    by_id = {fact.id: (score, kind) for fact, score, kind in hits}
    assert near.id in by_id
    score, kind = by_id[near.id]
    assert kind == "vector"
    expected = _expected_vector_score(query_vec, [1.0, 0.0, 0.0])
    assert score == pytest.approx(expected)
    # Orthogonal embedding -> vector score 0 -> filtered (score > 0 required).
    assert far.id not in by_id


# ---------------------------------------------------------------------------
# 3. Hybrid: found by both paths -> mean score
# ---------------------------------------------------------------------------


def test_hybrid_score_is_mean(db_session):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    target = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    _save(db_session, [target])
    query_vec = [1.0, 0.0, 0.0]
    save_embedding(db_session, target.id, [1.0, 0.0, 0.0])
    db_session.commit()

    hits = search_facts(
        db_session, "harbor", None, 20, 0.0, embedder=_FixedEmbedder(query_vec)
    )
    assert len(hits) == 1
    fact, score, kind = hits[0]
    assert fact.id == target.id
    assert kind == "hybrid"
    lexical = _expected_jaccard("harbor", target)
    vector = _expected_vector_score(query_vec, [1.0, 0.0, 0.0])
    assert score == pytest.approx((lexical + vector) / 2.0)


# ---------------------------------------------------------------------------
# 4. Empty / blank query validation
# ---------------------------------------------------------------------------


def test_empty_and_blank_query_rejected(api_client):
    resp_empty = api_client.post("/search", json={"query": ""})
    assert resp_empty.status_code == 422  # pydantic min_length=1
    resp_blank = api_client.post("/search", json={"query": "   "})
    assert resp_blank.status_code == 400  # endpoint blank check


# ---------------------------------------------------------------------------
# 5. min_score filters
# ---------------------------------------------------------------------------


def test_min_score_filters_results(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    target = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    _save(db_session, [target])

    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert any(f.id == target.id for f, _, _ in hits)
    (score,) = [s for f, s, _ in hits if f.id == target.id]
    assert 0.0 < score < 0.99

    high = search_facts(db_session, "harbor", None, 20, 0.99, embedder=None)
    assert all(f.id != target.id for f, _, _ in high)

    resp = api_client.post("/search", json={"query": "harbor", "min_score": 0.99})
    assert resp.status_code == 200
    assert resp.json()["hits"] == []
    assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# 6. limit truncates deterministically
# ---------------------------------------------------------------------------


def test_limit_truncates_deterministically(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    facts = [
        _fact(doc_a.id, ev_a[0].id, "Harbor bridge span", id=uuid4())
        for _ in range(3)
    ]
    _save(db_session, facts)

    full = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert len(full) == 3
    first = search_facts(db_session, "harbor", None, 2, 0.0, embedder=None)
    second = search_facts(db_session, "harbor", None, 2, 0.0, embedder=None)
    assert len(first) == 2
    assert [(f.id, s, k) for f, s, k in first] == [
        (f.id, s, k) for f, s, k in full[:2]
    ]
    assert [(f.id, s, k) for f, s, k in first] == [
        (f.id, s, k) for f, s, k in second
    ]

    resp = api_client.post("/search", json={"query": "harbor", "limit": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hits"]) == 2
    assert [h["fact"]["id"] for h in body["hits"]] == [
        str(f.id) for f, _, _ in full[:2]
    ]


# ---------------------------------------------------------------------------
# 7. document_id filter
# ---------------------------------------------------------------------------


def test_document_id_filter_restricts_results(db_session, api_client, force_lexical):
    doc_a, ev_a, doc_b, ev_b = _ingest_two_docs(db_session)
    fact_a = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    fact_b = _fact(doc_b.id, ev_b[0].id, "Harbor depot tally")
    _save(db_session, [fact_a, fact_b])

    unfiltered = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert {f.id for f, _, _ in unfiltered} == {fact_a.id, fact_b.id}

    only_a = search_facts(db_session, "harbor", doc_a.id, 20, 0.0, embedder=None)
    assert [f.id for f, _, _ in only_a] == [fact_a.id]
    only_b = search_facts(db_session, "harbor", doc_b.id, 20, 0.0, embedder=None)
    assert [f.id for f, _, _ in only_b] == [fact_b.id]

    resp = api_client.post(
        "/search", json={"query": "harbor", "document_id": str(doc_a.id)}
    )
    assert resp.status_code == 200
    assert [h["fact"]["id"] for h in resp.json()["hits"]] == [str(fact_a.id)]


# ---------------------------------------------------------------------------
# 7b. No results -> 200 with empty hits
# ---------------------------------------------------------------------------


def test_no_results_returns_empty_hits(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    _save(db_session, [_fact(doc_a.id, ev_a[0].id, "Harbor bridge span")])

    resp = api_client.post("/search", json={"query": "zxqv wjkb"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"] == []
    assert body["total"] == 0
    assert body["query"] == "zxqv wjkb"


# ---------------------------------------------------------------------------
# 8. Deterministic ordering: ties broken by fact id string
# ---------------------------------------------------------------------------


def test_tied_scores_ordered_by_fact_id_string(db_session):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    id_low, id_high = sorted([uuid4(), uuid4()], key=str)
    # Identical scored fields -> identical Jaccard scores by construction.
    first = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span", id=id_high)
    second = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span", id=id_low)
    _save(db_session, [first, second])  # persist in non-sorted order

    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert len(hits) == 2
    assert hits[0][1] == pytest.approx(hits[1][1])  # genuine tie
    assert [str(f.id) for f, _, _ in hits] == [str(id_low), str(id_high)]


# ---------------------------------------------------------------------------
# 9. Duplicate elimination across paths
# ---------------------------------------------------------------------------


def test_same_fact_appears_once_when_found_by_both_paths(db_session):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    target = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    other = _fact(doc_a.id, ev_a[0].id, "Harbor depot tally")
    _save(db_session, [target, other])
    save_embedding(db_session, target.id, [1.0, 0.0, 0.0])
    save_embedding(db_session, other.id, [1.0, 0.0, 0.0])
    db_session.commit()

    hits = search_facts(
        db_session, "harbor", None, 20, 0.0, embedder=_FixedEmbedder([1.0, 0.0, 0.0])
    )
    ids = [fact.id for fact, _, _ in hits]
    assert len(ids) == len(set(ids))  # no duplicates
    assert ids.count(target.id) == 1
    assert {fact.id for fact, _, _ in hits} == {target.id, other.id}


# ---------------------------------------------------------------------------
# 10. Provenance: document + evidence covering evidence_ids
# ---------------------------------------------------------------------------


def test_hit_carries_document_and_covering_evidence(
    db_session, api_client, force_lexical
):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    target = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    _save(db_session, [target])

    resp = api_client.post("/search", json={"query": "harbor"})
    assert resp.status_code == 200
    (hit,) = [h for h in resp.json()["hits"] if h["fact"]["id"] == str(target.id)]
    assert hit["document"] is not None
    assert hit["document"]["id"] == str(doc_a.id)
    assert hit["evidence"], "expected non-empty evidence for a grounded fact"
    returned_evidence_ids = {e["id"] for e in hit["evidence"]}
    assert set(map(str, target.evidence_ids)) <= returned_evidence_ids

    # Service-level hydration agrees.
    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    (fact, _, _) = next(item for item in hits if item[0].id == target.id)
    document, evidence, _ = hydrate_hit(db_session, fact)
    assert document is not None and document.id == doc_a.id
    assert {e.id for e in evidence} >= set(fact.evidence_ids)


# ---------------------------------------------------------------------------
# 11. Relationships listed on the hit
# ---------------------------------------------------------------------------


def test_hit_lists_persisted_relationship(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    fact_a = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    fact_b = _fact(doc_a.id, ev_a[0].id, "Harbor depot tally")
    _save(db_session, [fact_a, fact_b])
    rel = Relationship(
        fact_a_id=fact_a.id,
        fact_b_id=fact_b.id,
        relationship_type=RelationshipType.CORROBORATES,
        confidence=0.9,
        explanation="synthetic corroboration link",
    )
    save_relationship(db_session, rel)
    db_session.commit()

    _, _, rels_a = hydrate_hit(db_session, fact_a)
    assert [r.id for r in rels_a] == [rel.id]
    _, _, rels_b = hydrate_hit(db_session, fact_b)
    assert [r.id for r in rels_b] == [rel.id]

    resp = api_client.post("/search", json={"query": "harbor depot tally"})
    assert resp.status_code == 200
    hit_b = next(
        h for h in resp.json()["hits"] if h["fact"]["id"] == str(fact_b.id)
    )
    assert str(rel.id) in [r["id"] for r in hit_b["relationships"]]


# ---------------------------------------------------------------------------
# 12. Response metadata: query echoed, total == len(hits)
# ---------------------------------------------------------------------------


def test_response_metadata_query_and_total(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    _save(db_session, [_fact(doc_a.id, ev_a[0].id, "Harbor bridge span")])

    resp = api_client.post("/search", json={"query": "harbor"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "harbor"
    assert body["total"] == len(body["hits"])
    assert body["total"] >= 1


# ---------------------------------------------------------------------------
# 13. Malformed persisted data: dangling evidence links
# ---------------------------------------------------------------------------


def test_dangling_evidence_links_yield_empty_evidence_but_hit_survives(
    db_session, api_client, force_lexical
):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    dangling = _fact(
        doc_a.id, ev_a[0].id, "Harbor bridge span", evidence_ids=[uuid4()]
    )
    _save(db_session, [dangling])

    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    (fact, _, _) = next(item for item in hits if item[0].id == dangling.id)
    document, evidence, _ = hydrate_hit(db_session, fact)
    assert document is not None  # document row itself is fine
    assert evidence == []  # nonexistent evidence rows simply resolve to nothing

    resp = api_client.post("/search", json={"query": "harbor"})
    assert resp.status_code == 200
    hit = next(
        h for h in resp.json()["hits"] if h["fact"]["id"] == str(dangling.id)
    )
    assert hit["evidence"] == []


# ---------------------------------------------------------------------------
# 14. No mutation: counts identical, search never commits
# ---------------------------------------------------------------------------


def test_search_does_not_mutate_persisted_rows(db_session, api_client, force_lexical):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    fact_a = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    fact_b = _fact(doc_a.id, ev_a[0].id, "Harbor depot tally")
    _save(db_session, [fact_a, fact_b])
    save_embedding(db_session, fact_a.id, [1.0, 0.0, 0.0])
    save_relationship(
        db_session,
        Relationship(
            fact_a_id=fact_a.id,
            fact_b_id=fact_b.id,
            relationship_type=RelationshipType.RELATED,
            confidence=0.5,
            explanation="synthetic relation link",
        ),
    )
    db_session.commit()

    before = _table_counts(db_session)
    hits = search_facts(
        db_session, "harbor", None, 20, 0.0, embedder=_FixedEmbedder([1.0, 0.0, 0.0])
    )
    for fact, _, _ in hits:
        hydrate_hit(db_session, fact)
    assert _table_counts(db_session) == before

    resp = api_client.post("/search", json={"query": "harbor"})
    assert resp.status_code == 200
    assert _table_counts(db_session) == before


def test_search_never_calls_commit(db_session, monkeypatch):
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    _save(db_session, [_fact(doc_a.id, ev_a[0].id, "Harbor bridge span")])

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("search must never commit")

    monkeypatch.setattr(db_session, "commit", _boom)
    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert hits
    for fact, _, _ in hits:
        hydrate_hit(db_session, fact)
    db_session.rollback()  # leave the session clean without committing


# ---------------------------------------------------------------------------
# 15. No LLM involvement
# ---------------------------------------------------------------------------


def test_search_never_touches_llm_layer(db_session, monkeypatch):
    svc = search_service
    assert not any("llm" in name for name in dir(svc))
    for name, obj in vars(svc).items():
        assert not getattr(obj, "__name__", "").startswith("app.llm"), name
    assert not hasattr(svc, "get_llm_provider")

    # Functional: retrieval works with no API key configured.
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert settings.llm_provider == "gemini"
    doc_a, ev_a, _, _ = _ingest_two_docs(db_session)
    target = _fact(doc_a.id, ev_a[0].id, "Harbor bridge span")
    _save(db_session, [target])
    hits = search_facts(db_session, "harbor", None, 20, 0.0, embedder=None)
    assert any(f.id == target.id for f, _, _ in hits)


# ---------------------------------------------------------------------------
# 16. API regression: existing routes unaffected by the search router
# ---------------------------------------------------------------------------


def test_api_regression_existing_routes_unaffected(api_client):
    assert api_client.get("/health").status_code == 200
    missing = uuid.uuid4()
    assert (
        api_client.post(f"/documents/{missing}/relationships").status_code == 404
    )
    assert api_client.get(f"/documents/{missing}/facts").status_code == 404
