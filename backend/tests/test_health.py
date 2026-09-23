"""Backend tests (Phase 0): import/startup and /health behavior."""

from fastapi.testclient import TestClient

from app.main import app, create_app


def test_app_imports_and_factory():
    assert app is not None
    assert create_app() is not None


def test_health_endpoint_shape():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"]
    assert body["environment"]
    # Application health is distinguished from dependency health.
    assert "database" in body
    assert "configured" in body["database"]
    assert "reachable" in body["database"]
    assert "detail" in body["database"]


def test_unreachable_database_fails_fast_instead_of_hanging():
    """An unreachable host must be bounded by the libpq connect timeout.

    A statement timeout cannot bound a TCP connect, so without
    ``connect_timeout`` this probe hangs indefinitely and takes the
    readiness endpoint (and the test suite) with it.
    """
    import time

    from sqlalchemy import create_engine

    from app.db import session as session_module

    dead = create_engine(
        "postgresql+psycopg://aletheia:aletheia@localhost:5499/aletheia",
        pool_pre_ping=True,
        connect_args={"connect_timeout": session_module.CONNECT_TIMEOUT_S},
    )
    original = session_module.engine
    session_module.engine = dead
    try:
        started = time.monotonic()
        reachable, detail = session_module.check_database_connection()
        elapsed = time.monotonic() - started
    finally:
        session_module.engine = original
        dead.dispose()

    assert reachable is False
    assert detail
    assert elapsed < 15, f"connect probe took {elapsed:.1f}s; connect_timeout not applied"


def test_browser_origin_gets_cors_headers():
    """The web client fetches from the browser, so every call is
    cross-origin. Without this the UI shows "Failed to fetch" and no
    server-side log records anything wrong."""
    from app.core.config import settings

    client = TestClient(create_app())
    origin = settings.cors_origins[0]
    response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_preflight_is_answered_for_a_post_route():
    from app.core.config import settings

    client = TestClient(create_app())
    response = client.options(
        "/search",
        headers={
            "Origin": settings.cors_origins[0],
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
