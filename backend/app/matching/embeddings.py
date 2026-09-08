"""Embedding-based candidate discovery (R-EMB).

RANKING ONLY: every score produced here orders candidate pairs so that
downstream comparison can focus on the most promising ones first.
Nothing in this module returns or implies a relationship type and no
score is a classification — typing decisions belong to the compare /
judgment siblings, never to embeddings.
"""

from __future__ import annotations

import math
import re
from typing import Any
from uuid import UUID

import numpy as np

from app.models.fact import Fact
from app.models.relationship import CandidatePair  # noqa: F401 -- imported so callers reuse the shared contract instead of redefining it; scores below feed CandidatePair.score (ordering only).

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_ST_MODELS: dict[str, Any] = {}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _get_st_model(model_name: str = EMBEDDING_MODEL) -> Any:
    """Return the lazily-created singleton engine for ``model_name``.

    The sentence-transformers import lives inside this function so that
    importing this module never fails when the optional dependency is
    absent. Callers needing vectors without the dependency receive a
    RuntimeError carrying install guidance (the service maps that to
    embedding-unavailable).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers (and torch) are required for embedding "
            "vectors; install them with: pip install sentence-transformers torch. "
            "Without them, embedding-based candidate ranking is unavailable."
        ) from exc
    if model_name not in _ST_MODELS:
        _ST_MODELS[model_name] = SentenceTransformer(model_name)
    return _ST_MODELS[model_name]


def build_embedding_text(fact: Fact) -> str:
    """Build the deterministic text embedded for a fact.

    Each field falls back gracefully (canonical form first, then the raw
    form, then the empty string) so that missing optional fields never
    break embedding input construction.
    """
    subject = fact.canonical_subject or fact.subject
    predicate = fact.canonical_predicate or fact.predicate
    unit = fact.normalized_unit or fact.unit or ""
    time_text = fact.time_text or ""
    scope_text = fact.scope_text or ""
    value_text = fact.value_text or ""
    return " | ".join([subject, predicate, unit, time_text, scope_text, value_text])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1] for ordering candidates.

    Zero-vectors, empty inputs, and incompatible inputs yield 0.0. This
    function never raises.
    """
    try:
        va = np.asarray(a, dtype=np.float64).ravel()
        vb = np.asarray(b, dtype=np.float64).ravel()
        if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
            return 0.0
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        score = float(np.dot(va, vb) / (norm_a * norm_b))
        if not math.isfinite(score):
            return 0.0
        return max(-1.0, min(1.0, score))
    except Exception:
        return 0.0


def _subject_predicate_tokens(fact: Fact) -> set[str]:
    text = " ".join(
        [fact.canonical_subject or fact.subject, fact.canonical_predicate or fact.predicate]
    ).lower()
    return set(_TOKEN_RE.findall(text))


def lexical_score(fact_a: Fact, fact_b: Fact) -> float:
    """Token-Jaccard overlap over subject + predicate text.

    Lowercase alphanumeric tokens only; an empty token union yields 0.0.
    Pure stdlib. Ordering signal only — never a relationship verdict.
    """
    tokens_a = _subject_predicate_tokens(fact_a)
    tokens_b = _subject_predicate_tokens(fact_b)
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


class FactEmbedder:
    """Embeds fact texts with a lazily-loaded sentence-transformer engine.

    Construction stores configuration only and never touches the model.
    Use :meth:`available` for an import-only readiness probe and
    :meth:`embed` to produce unit-length vectors for ranking.
    """

    dim: int = EMBEDDING_DIM

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.dim = EMBEDDING_DIM

    @staticmethod
    def available() -> tuple[bool, str]:
        """Import-only readiness probe (never downloads a model).

        Returns ``(True, reason)`` when sentence-transformers is
        importable, else ``(False, reason)``. Fail-closed: any problem
        yields ``False`` and this method never raises.
        """
        try:
            import sentence_transformers  # noqa: F401
        except Exception as exc:
            return (False, f"sentence-transformers unavailable: {exc}")
        return (True, f"sentence-transformers importable; model '{EMBEDDING_MODEL}' loads lazily on first embed")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into unit-length vectors.

        Empty input yields ``[]`` without touching the model engine.
        Each output row is normalized to unit length when its norm is
        positive; zero rows are returned unchanged.
        """
        if not texts:
            return []
        model = _get_st_model(self.model_name)
        vectors = model.encode(texts)
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        nonzero = norms > 0
        out = np.divide(arr, np.where(nonzero, norms, 1.0))
        return [[float(x) for x in row] for row in out]


def rank_candidates(
    query_id: UUID,
    query_vec: list[float],
    pool: list[tuple[UUID, list[float]]],
    top_k: int,
    floor: float,
) -> list[tuple[UUID, float]]:
    """Order pool entries by cosine similarity to the query vector.

    Excludes ``query_id`` itself, drops scores below ``floor``, keeps at
    most ``top_k`` entries, and breaks score ties by id string so output
    order is deterministic. Pure function (no DB access). Scores rank
    only and carry no relationship meaning.
    """
    if top_k <= 0:
        return []
    scored: list[tuple[UUID, float]] = []
    for fact_id, vec in pool:
        if fact_id == query_id:
            continue
        score = cosine_similarity(query_vec, vec)
        if score < floor:
            continue
        scored.append((fact_id, score))
    scored.sort(key=lambda item: (-item[1], str(item[0])))
    return scored[:top_k]
