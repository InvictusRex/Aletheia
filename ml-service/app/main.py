from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
import base64
import binascii

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings_engine import (
    MODEL_DIM,
    MODEL_NAME,
    embed_texts,
    init_embeddings,
)
from .ocr_engine import DEFAULT_LANG, PROVIDER_TAG, init_ocr, ocr_png_base64

_state: dict[str, Any] = {
    "ocr": {"available": False, "reason": "not initialised yet"},
    "embeddings": {"available": False, "reason": "not initialised yet"},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ok, reason = init_ocr(DEFAULT_LANG)
    _state["ocr"] = {"available": ok, "lang": DEFAULT_LANG, "reason": reason}
    ok, reason = init_embeddings(MODEL_NAME)
    _state["embeddings"] = {
        "available": ok,
        "model": MODEL_NAME,
        "dim": MODEL_DIM,
        "reason": reason,
    }
    yield


app = FastAPI(title="aletheia-ml-service", lifespan=lifespan)


class OcrRequest(BaseModel):
    png_base64: str = Field(min_length=1)
    lang: str = Field(default=DEFAULT_LANG, min_length=1)


class OcrResponse(BaseModel):
    provider: str = PROVIDER_TAG
    texts: list[str] = Field(default_factory=list)
    scores: list[float | None] = Field(default_factory=list)
    boxes: list[Any] = Field(default_factory=list)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=2000)


class EmbedResponse(BaseModel):
    model: str = MODEL_NAME
    dim: int = MODEL_DIM
    vectors: list[list[float]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    ready = _state["ocr"]["available"] and _state["embeddings"]["available"]
    return {
        "status": "ok" if ready else "degraded",
        "service": "aletheia-ml-service",
        "ocr": _state["ocr"],
        "embeddings": _state["embeddings"],
    }


@app.post("/ocr", response_model=OcrResponse)
def run_ocr(request: OcrRequest) -> OcrResponse:
    if not _state["ocr"]["available"]:
        raise HTTPException(
            status_code=503,
            detail=f"OCR engine unavailable: {_state['ocr'].get('reason')}",
        )
    try:
        raw = base64.b64decode(request.png_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"png_base64 is not valid base64: {exc}"
        ) from exc
    if not raw:
        raise HTTPException(status_code=422, detail="png_base64 decodes to empty bytes")
    try:
        result = ocr_png_base64(request.png_base64, request.lang)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return OcrResponse(
        texts=result["texts"], scores=result["scores"], boxes=result["boxes"]
    )


@app.post("/embed", response_model=EmbedResponse)
def run_embed(request: EmbedRequest) -> EmbedResponse:
    if not _state["embeddings"]["available"]:
        raise HTTPException(
            status_code=503,
            detail=f"embedding model unavailable: {_state['embeddings'].get('reason')}",
        )
    try:
        vectors = embed_texts(request.texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return EmbedResponse(vectors=vectors)
