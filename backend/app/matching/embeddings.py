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
from uuid import UUID

import numpy as np  # via pgvector (hard backend dependency)

from app.core.config import settings
from app.ml.client import MLServiceClient, MLServiceError
from app.models.fact import Fact

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


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
    dim: int = EMBEDDING_DIM

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        _client: MLServiceClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self._client = _client

    def _service(self) -> MLServiceClient:
        return self._client or MLServiceClient(base_url=settings.ml_service_url)

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            status = MLServiceClient(base_url=settings.ml_service_url).health(
                timeout_s=2.0
            )
            embeddings = status.get("embeddings") if isinstance(status, dict) else None
            if isinstance(embeddings, dict) and embeddings.get("available"):
                return (True, f"ML embeddings ready ({embeddings.get('model', '?')})")
            reason = embeddings.get("reason") if isinstance(embeddings, dict) else status
            return (False, f"ML embeddings unavailable: {reason}")
        except MLServiceError as exc:
            return (False, f"ML service unreachable: {exc}")
        except Exception as exc:
            return (False, f"ML embeddings probe failed ({type(exc).__name__}): {exc}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            payload = self._service().embed_texts(texts)
            vectors = payload.get("vectors")
        except MLServiceError as exc:
            raise RuntimeError(f"ML embedding failed: {exc}") from exc
        rows = _validated_rows(vectors)
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        nonzero = norms > 0
        out = np.divide(arr, np.where(nonzero, norms, 1.0))
        return [[float(x) for x in row] for row in out]


def _validated_rows(vectors: object) -> list[list[float]]:
    if not isinstance(vectors, list) or not vectors:
        raise RuntimeError("ML service returned no vectors")
    rows: list[list[float]] = []
    for row in vectors:
        if not isinstance(row, list) or not row:
            raise RuntimeError("ML service returned a malformed vector")
        values: list[float] = []
        for value in row:
            if isinstance(value, bool):
                raise RuntimeError("ML service returned a malformed vector")
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"ML service returned a non-numeric vector: {exc}"
                ) from exc
            if not math.isfinite(number):
                raise RuntimeError("ML service returned a non-finite vector")
            values.append(number)
        rows.append(values)
    return rows


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
