from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MODEL_DIM = 384

_model: object | None = None
_model_lock = threading.Lock()
_model_error: str | None = None


def init_embeddings(model_name: str = MODEL_NAME) -> tuple[bool, str]:
    global _model, _model_error
    with _model_lock:
        if _model is not None:
            return True, f"embedding model ready ({model_name})"
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            _model_error = f"sentence-transformers not installed: {exc}"
            return False, _model_error
        try:
            _model = SentenceTransformer(model_name)
        except Exception as exc:
            _model_error = f"embedding model load failed ({model_name}): {exc}"
            return False, _model_error
        return True, f"embedding model ready ({model_name})"


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of strings")
    for text in texts:
        if not isinstance(text, str):
            raise ValueError("every text must be a string")
    with _model_lock:
        model = _model
    if model is None:
        raise RuntimeError("embedding model is not initialised")
    try:
        vectors = model.encode(texts)  # type: ignore[union-attr]
    except Exception as exc:
        raise RuntimeError(
            f"embedding inference failed ({type(exc).__name__}): {exc}"
        ) from exc
    try:
        rows = [list(row) for row in vectors]
        return [[float(x) for x in row] for row in rows]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"embedding output is not numeric: {exc}") from exc
