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
