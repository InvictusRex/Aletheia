"""Shared fixtures for T7/T8 backend tests.

- ``db_session``: file-backed SQLite session with the Phase 1 schema
  created (``Base.metadata.create_all``). File-backed (not ``:memory:``)
  so the same database is visible across sessions/connections.
- ``api_client``: FastAPI ``TestClient`` with the ``get_db`` dependency
  overridden to yield ``db_session`` (``get_db`` is a generator
  dependency, so the override must also be a generator).
"""

from __future__ import annotations

import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base


@pytest.fixture()
def db_session():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = tmp.name
    tmp.close()
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        os.unlink(path)


@pytest.fixture()
def api_client(db_session):
    from fastapi.testclient import TestClient

    from app.db.session import get_db
    from app.main import create_app

    app = create_app()

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
