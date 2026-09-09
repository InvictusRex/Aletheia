import math

import pytest

from app.extraction.ml_service_ocr import MLServiceOCRProvider
from app.extraction.ocr_provider import (
    OcrMalformedError,
    OcrRuntimeError,
    OcrUnavailableError,
)
from app.matching import embeddings as embeddings_module
from app.matching.embeddings import EMBEDDING_DIM, FactEmbedder
from app.ml.client import (
    MLServiceClient,
    MLServiceError,
    MLServiceHTTPError,
    MLServiceUnavailableError,
)


def _ocr_payload(texts=("hello world",), scores=(0.9,), boxes=(((0, 0), (10, 0), (10, 10), (0, 10)),)):
    return {"texts": list(texts), "scores": list(scores), "boxes": [list(map(list, b)) for b in boxes]}


def _client(post=None, get=None):
    def _post(url, payload, timeout):
        assert url.startswith("http://ml-test:8001/")
        return post(url, payload, timeout)

    def _get(url, timeout):
        assert url.startswith("http://ml-test:8001/")
        return get(url, timeout)

    return MLServiceClient(
        base_url="http://ml-test:8001", post_json=_post, get_json=_get
    )


def test_client_posts_expected_paths():
    seen = []

    def post(url, payload, timeout):
        seen.append((url, payload))
        if url.endswith("/embed"):
            return {"vectors": []}
        return {"texts": [], "scores": [], "boxes": []}

    client = _client(post=post, get=lambda url, timeout: {})
    client.ocr_page(b"%PNGDATA%", lang="en")
    assert seen[0][0] == "http://ml-test:8001/ocr"
    assert seen[0][1]["lang"] == "en"
    assert seen[0][1]["png_base64"]
    client.embed_texts(["a", "b"])
    assert seen[1][0] == "http://ml-test:8001/embed"
    assert seen[1][1] == {"texts": ["a", "b"]}


def test_client_health_returns_payload():
    client = _client(
        post=lambda u, p, t: {}, get=lambda u, t: {"status": "ok"}
    )
    assert client.health() == {"status": "ok"}


def test_client_rejects_bad_input_without_http():
    calls = []
    client = _client(
        post=lambda u, p, t: calls.append(u),
        get=lambda u, t: calls.append(u),
    )
    with pytest.raises(ValueError):
        client.ocr_page(b"", lang="en")
    with pytest.raises(ValueError):
        client.embed_texts([])
    assert calls == []


def test_client_malformed_responses_raise():
    client = _client(
        post=lambda u, p, t: {"nope": True}, get=lambda u, t: [1, 2]
    )
    with pytest.raises(MLServiceError):
        client.ocr_page(b"%PNGDATA%")
    with pytest.raises(MLServiceError):
        client.embed_texts(["a"])
    with pytest.raises(MLServiceError):
        client.health()


def test_http_error_types_preserved():
    assert issubclass(MLServiceUnavailableError, MLServiceError)
    assert issubclass(MLServiceHTTPError, MLServiceError)
    err = MLServiceHTTPError(503, "not ready")
    assert err.status == 503


def _pdf_bytes():
    import pymupdf

    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), "probe")
        return bytes(doc.tobytes())
    finally:
        doc.close()


def test_ml_ocr_success_maps_blocks():
    payload = _ocr_payload()
    provider = MLServiceOCRProvider(
        client=_client(post=lambda u, p, t: payload, get=lambda u, t: {})
    )
    result = provider.extract_page(_pdf_bytes(), 0)
    assert result.provider == "paddleocr"
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "hello world"
    assert result.blocks[0].confidence == 0.9
    assert result.blocks[0].bbox is not None


def test_ml_ocr_unavailable_maps_to_unavailable():
    def post(url, payload, timeout):
        raise MLServiceUnavailableError("refused")

    provider = MLServiceOCRProvider(
        client=_client(post=post, get=lambda u, t: {})
    )
    with pytest.raises(OcrUnavailableError):
        provider.extract_page(_pdf_bytes(), 0)


def test_ml_ocr_503_maps_to_unavailable():
    def post(url, payload, timeout):
        raise MLServiceHTTPError(503, "engine not ready")

    provider = MLServiceOCRProvider(
        client=_client(post=post, get=lambda u, t: {})
    )
    with pytest.raises(OcrUnavailableError):
        provider.extract_page(_pdf_bytes(), 0)


def test_ml_ocr_500_maps_to_runtime():
    def post(url, payload, timeout):
        raise MLServiceHTTPError(500, "boom")

    provider = MLServiceOCRProvider(
        client=_client(post=post, get=lambda u, t: {})
    )
    with pytest.raises(OcrRuntimeError):
        provider.extract_page(_pdf_bytes(), 0)


def test_ml_ocr_garbage_maps_to_malformed():
    provider = MLServiceOCRProvider(
        client=_client(
            post=lambda u, p, t: {"texts": "not-a-list", "scores": [], "boxes": []},
            get=lambda u, t: {},
        )
    )
    with pytest.raises(OcrMalformedError):
        provider.extract_page(_pdf_bytes(), 0)


def test_ml_ocr_validates_input_without_http():
    calls = []
    provider = MLServiceOCRProvider(
        client=_client(
            post=lambda u, p, t: calls.append(u) or _ocr_payload(),
            get=lambda u, t: {},
        )
    )
    with pytest.raises(ValueError):
        provider.extract_page(b"", 0)
    with pytest.raises(ValueError):
        provider.extract_page(_pdf_bytes(), "0")
    assert calls == []


def test_pipeline_prefers_ml_service(monkeypatch):
    from app.extraction import pipeline

    def fake_health(url, timeout):
        return {"status": "ok", "ocr": {"available": True}}

    monkeypatch.setattr(
        pipeline, "MLServiceClient",
        lambda base_url: MLServiceClient(
            base_url=base_url,
            post_json=lambda u, p, t: _ocr_payload(),
            get_json=fake_health,
        ),
    )
    provider = pipeline.get_ocr_provider()
    assert isinstance(provider, MLServiceOCRProvider)


def test_pipeline_falls_back_when_ml_down(monkeypatch):
    from app.extraction import pipeline

    def boom(url, timeout):
        raise MLServiceUnavailableError("refused")

    monkeypatch.setattr(
        pipeline, "MLServiceClient",
        lambda base_url: MLServiceClient(
            base_url=base_url,
            post_json=lambda u, p, t: _ocr_payload(),
            get_json=boom,
        ),
    )
    assert pipeline.get_ocr_provider() is None


def _unit_vec(dim=EMBEDDING_DIM):
    return [1.0 / math.sqrt(dim)] * dim


def test_embedder_available_true_and_false(monkeypatch):
    ok = MLServiceClient(
        base_url="http://x",
        post_json=lambda u, p, t: {},
        get_json=lambda u, t: {"embeddings": {"available": True, "model": "m"}},
    )
    bad = MLServiceClient(
        base_url="http://x",
        post_json=lambda u, p, t: {},
        get_json=lambda u, t: (_ for _ in ()).throw(
            MLServiceUnavailableError("down")
        ),
    )
    monkeypatch.setattr(embeddings_module, "MLServiceClient", lambda base_url: ok)
    assert FactEmbedder.available() == (True, "ML embeddings ready (m)")
    monkeypatch.setattr(embeddings_module, "MLServiceClient", lambda base_url: bad)
    available, reason = FactEmbedder.available()
    assert available is False and reason


def test_embedder_embed_normalizes_and_validates():
    dim = EMBEDDING_DIM
    calls = []

    def post(url, payload, timeout):
        calls.append(payload)
        return {"vectors": [[3.0] * dim, [0.0] * dim], "dim": dim}

    embedder = FactEmbedder(
        _client=MLServiceClient(
            base_url="http://x",
            post_json=post,
            get_json=lambda u, t: {},
        )
    )
    assert embedder.embed([]) == []
    assert calls == []
    vectors = embedder.embed(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(v) == dim for v in vectors)
    assert abs(math.sqrt(sum(x * x for x in vectors[0])) - 1.0) < 1e-9
    assert all(x == 0.0 for x in vectors[1])
    assert calls[0] == {"texts": ["alpha", "beta"]}

    broken = FactEmbedder(
        _client=MLServiceClient(
            base_url="http://x",
            post_json=lambda u, p, t: {"vectors": [["NaN"]]},
            get_json=lambda u, t: {},
        )
    )
    with pytest.raises(RuntimeError):
        broken.embed(["alpha"])


def test_embedder_construction_touches_no_network():
    calls = []
    embedder = FactEmbedder(
        _client=MLServiceClient(
            base_url="http://x",
            post_json=lambda u, p, t: calls.append(u) or {"vectors": []},
            get_json=lambda u, t: calls.append(u) or {},
        )
    )
    assert embedder.model_name
    assert embedder.dim == EMBEDDING_DIM
    assert calls == []
