"""POST /search: retrieval over persisted knowledge, never generation.

Returns stored facts with provenance. No LLM is involved; the database
is never mutated by this endpoint.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document, EvidenceUnit, Fact, Relationship
from app.search.service import hydrate_hit, search_facts

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: UUID | None = None
    limit: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchHit(BaseModel):
    fact: Fact
    score: float
    match_kind: str
    document: Document | None = None
    evidence: list[EvidenceUnit] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    total: int = 0


@router.post("/search", response_model=SearchResponse)
def search_knowledge_layer(
    request: SearchRequest, db: Session = Depends(get_db)
) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")
    ranked = search_facts(
        db,
        request.query,
        request.document_id,
        request.limit,
        request.min_score,
    )
    hits: list[SearchHit] = []
    for fact, score, kind in ranked:
        document, evidence, relationships = hydrate_hit(db, fact)
        hits.append(
            SearchHit(
                fact=fact,
                score=score,
                match_kind=kind,
                document=document,
                evidence=evidence,
                relationships=relationships,
            )
        )
    return SearchResponse(query=request.query, hits=hits, total=len(hits))
