"""Matching package: embedding-ranked candidate discovery (R-EMB).

Ranking only — re-exported helpers order candidates; relationship
typing lives elsewhere.
"""

from app.matching.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FactEmbedder,
    build_embedding_text,
    cosine_similarity,
    lexical_score,
    rank_candidates,
)

__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "FactEmbedder",
    "build_embedding_text",
    "cosine_similarity",
    "lexical_score",
    "rank_candidates",
]
