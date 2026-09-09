"""Aletheia backend application factory (Phase 1)."""

import logging

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.facts import router as facts_router
from app.api.health import router as health_router
from app.api.relationships import router as relationships_router
from app.api.search import router as search_router
from app.core.config import settings

# Ensure application INFO logs (e.g. per-request Groq token usage) are
# visible under uvicorn, whose default config leaves the root logger at
# WARNING. No-op when a handler is already configured (e.g. tests).
logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(facts_router)
    app.include_router(relationships_router)
    app.include_router(search_router)
    return app


app = create_app()
