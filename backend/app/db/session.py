"""Database foundation (Phase 0).

Provides the SQLAlchemy 2.0 engine/session factory and a lightweight
connectivity check. No domain tables are defined in Phase 0; the schema
(documents/pages/evidence/facts/relationships/embeddings) arrives in
Phase 1/Phase 4.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection(timeout_seconds: int = 2) -> tuple[bool, str]:
    """Return ``(reachable, detail)`` without raising.

    Uses a short statement timeout so health checks fail fast when the
    database is unavailable instead of hanging startup/readiness probes.
    """
    try:
        with engine.connect() as connection:
            connection.execute(
                text(f"SET LOCAL statement_timeout = {int(timeout_seconds * 1000)}")
            )
            connection.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - health check must not raise
        return False, f"{type(exc).__name__}: {exc}"
