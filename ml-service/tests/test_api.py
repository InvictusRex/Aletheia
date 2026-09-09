import base64
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _load_package():
    if "mlsvc" in sys.modules:
        return sys.modules["mlsvc"]
    pkg_spec = importlib.util.spec_from_file_location(
        "mlsvc", _APP_DIR / "__init__.py",
        submodule_search_locations=[str(_APP_DIR)],
    )
    package = importlib.util.module_from_spec(pkg_spec)
    sys.modules["mlsvc"] = package
    pkg_spec.loader.exec_module(package)
    return package


def _load_main():
    _load_package()
    spec = importlib.util.spec_from_file_location("mlsvc.main", _APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlsvc.main"] = module
    spec.loader.exec_module(module)
    return module


main_module = _load_main()

from mlsvc import ocr_engine  # noqa: E402  (import-only check below)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        main_module, "init_ocr", lambda lang="en": (True, "stub")
    )
    monkeypatch.setattr(
        main_module, "init_embeddings", lambda model_name="m": (True, "stub")
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_reports_components(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "aletheia-ml-service"
    assert body["ocr"]["available"] is True
    assert body["embeddings"]["available"] is True
    assert body["embeddings"]["dim"] == 384


def test_ocr_validation_rejects_bad_input(client, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("engine must not run on invalid input")

    monkeypatch.setattr(main_module, "ocr_png_base64", must_not_run)
    assert client.post("/ocr", json={}).status_code == 422
    assert client.post(
        "/ocr", json={"png_base64": "!!!not-base64!!!", "lang": "en"}
    ).status_code in (422, 500)


def test_ocr_success_shape(client, monkeypatch):
    def fake_ocr(png_b64, lang="en"):
        assert lang == "en"
        assert base64.b64decode(png_b64, validate=True)
        return {
            "texts": ["hello"],
            "scores": [0.9],
            "boxes": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]],
        }

    monkeypatch.setattr(main_module, "ocr_png_base64", fake_ocr)
    resp = client.post(
        "/ocr",
        json={"png_base64": base64.b64encode(b"fakepng").decode(), "lang": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "paddleocr"
    assert body["texts"] == ["hello"]
    assert body["scores"] == [0.9]
    assert len(body["boxes"]) == 1


def test_ocr_unavailable_maps_to_503(client):
    main_module._state["ocr"] = {"available": False, "reason": "no weights"}
    resp = client.post(
        "/ocr", json={"png_base64": base64.b64encode(b"x").decode()}
    )
    assert resp.status_code == 503


def test_embed_validation(client):
    assert client.post("/embed", json={"texts": []}).status_code == 422
    assert client.post("/embed", json={"texts": "nope"}).status_code == 422
    assert client.post("/embed", json={"texts": [1, 2]}).status_code == 422


def test_embed_success_shape(client, monkeypatch):
    def fake_embed(texts):
        assert texts == ["alpha", "beta"]
        return [[0.1] * 384, [0.2] * 384]

    monkeypatch.setattr(main_module, "embed_texts", fake_embed)
    resp = client.post("/embed", json={"texts": ["alpha", "beta"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 384
    assert len(body["vectors"]) == 2
    assert len(body["vectors"][0]) == 384


def test_embed_unavailable_maps_to_503(client):
    main_module._state["embeddings"] = {"available": False, "reason": "no model"}
    resp = client.post("/embed", json={"texts": ["a"]})
    assert resp.status_code == 503


def test_engine_modules_import_without_heavy_deps():
    assert hasattr(ocr_engine, "ocr_png_base64")
    assert hasattr(ocr_engine, "init_ocr")


def test_as_float_list_converts_numpy_arrays():
    np = pytest.importorskip("numpy")

    boxes = np.array([[[0.0, 0.0], [10.0, 0.0]], [[1.0, 1.0], [5.0, 5.0]]])
    out = ocr_engine._as_float_list(boxes)
    assert out == [[[0.0, 0.0], [10.0, 0.0]], [[1.0, 1.0], [5.0, 5.0]]]
    assert ocr_engine._as_float_list(np.array([0.9, 0.8])) == [0.9, 0.8]
    assert ocr_engine._as_float_list(None) == []
    assert ocr_engine._as_float_list("x") == []
