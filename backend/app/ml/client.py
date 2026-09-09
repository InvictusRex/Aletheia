from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable

__all__ = [
    "MLServiceClient",
    "MLServiceError",
    "MLServiceHTTPError",
    "MLServiceUnavailableError",
    "default_get_json",
    "default_post_json",
]

DEFAULT_BASE_URL = "http://ml:8001"


class MLServiceError(Exception):
    pass


class MLServiceUnavailableError(MLServiceError):
    pass


class MLServiceHTTPError(MLServiceError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"ML service HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


def _read_json_response(response: Any, url: str) -> Any:
    try:
        raw = response.read()
    except Exception as exc:
        raise MLServiceError(f"ML service unreadable response from {url}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise MLServiceError(
            f"ML service non-JSON response from {url}: {exc}"
        ) from exc


def default_post_json(url: str, payload: dict, timeout_s: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            code = getattr(response, "status", 200)
            if code >= 400:
                raise MLServiceHTTPError(code, "")
            return _read_json_response(response, url)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise MLServiceHTTPError(exc.code, body) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise MLServiceUnavailableError(f"ML service unreachable at {url}: {exc}") from exc


def default_get_json(url: str, timeout_s: float) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            code = getattr(response, "status", 200)
            if code >= 400:
                raise MLServiceHTTPError(code, "")
            return _read_json_response(response, url)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise MLServiceHTTPError(exc.code, body) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise MLServiceUnavailableError(f"ML service unreachable at {url}: {exc}") from exc


PostJson = Callable[[str, dict, float], Any]
GetJson = Callable[[str, float], Any]


class MLServiceClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        post_json: PostJson | None = None,
        get_json: GetJson | None = None,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout_s = max(0.1, float(timeout_s))
        self._post_json = post_json or default_post_json
        self._get_json = get_json or default_get_json

    @property
    def base_url(self) -> str:
        return self._base_url

    def health(self, timeout_s: float = 5.0) -> dict:
        payload = self._get_json(f"{self._base_url}/health", timeout_s)
        if not isinstance(payload, dict):
            raise MLServiceError("ML service health is not a JSON object")
        return payload

    def ocr_page(
        self, png_bytes: bytes, lang: str = "en", timeout_s: float = 300.0
    ) -> dict:
        if not isinstance(png_bytes, (bytes, bytearray)) or not png_bytes:
            raise ValueError("ocr_page requires non-empty PNG bytes")
        payload = self._post_json(
            f"{self._base_url}/ocr",
            {
                "png_base64": base64.b64encode(bytes(png_bytes)).decode("ascii"),
                "lang": lang or "en",
            },
            timeout_s,
        )
        if not isinstance(payload, dict):
            raise MLServiceError("ML service /ocr is not a JSON object")
        for key in ("texts", "scores", "boxes"):
            if key not in payload:
                raise MLServiceError(f"ML service /ocr missing key {key!r}")
        if not isinstance(payload["texts"], list):
            raise MLServiceError("ML service /ocr 'texts' is not a list")
        return payload

    def embed_texts(
        self, texts: list[str], timeout_s: float = 120.0
    ) -> dict:
        if not isinstance(texts, list) or not texts:
            raise ValueError("embed_texts requires a non-empty list of strings")
        payload = self._post_json(
            f"{self._base_url}/embed", {"texts": texts}, timeout_s
        )
        if not isinstance(payload, dict):
            raise MLServiceError("ML service /embed is not a JSON object")
        if not isinstance(payload.get("vectors"), list):
            raise MLServiceError("ML service /embed 'vectors' is not a list")
        return payload
