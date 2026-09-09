from __future__ import annotations

from app.extraction.ocr_provider import (
    OcrError,
    OcrMalformedError,
    OcrProvider,
    OcrResult,
    OcrRuntimeError,
    OcrUnavailableError,
)
from app.extraction.paddle_ocr import _parse_predict_output, render_page_to_png
from app.ml.client import (
    DEFAULT_BASE_URL,
    MLServiceClient,
    MLServiceError,
    MLServiceHTTPError,
    MLServiceUnavailableError,
)

__all__ = ["MLServiceOCRProvider"]


class MLServiceOCRProvider(OcrProvider):
    def __init__(
        self,
        client: MLServiceClient | None = None,
        base_url: str | None = None,
        lang: str = "en",
        dpi: int = 200,
        timeout_s: float = 300.0,
    ) -> None:
        self._client = client or MLServiceClient(base_url=base_url or DEFAULT_BASE_URL)
        self._lang = lang
        self._dpi = dpi
        self._timeout_s = timeout_s

    def extract_page(
        self, pdf_bytes: bytes, pdf_page_number: int
    ) -> OcrResult:
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        if not isinstance(pdf_bytes, bytes) or len(pdf_bytes) == 0:
            raise ValueError("extract_page requires non-empty PDF bytes")
        if isinstance(pdf_page_number, bool) or not isinstance(
            pdf_page_number, int
        ):
            raise ValueError(
                "extract_page requires an int pdf_page_number, "
                f"got {type(pdf_page_number).__name__}"
            )

        png_bytes = render_page_to_png(
            pdf_bytes, pdf_page_number, dpi=self._dpi
        )
        try:
            raw = self._client.ocr_page(
                png_bytes, lang=self._lang, timeout_s=self._timeout_s
            )
        except MLServiceUnavailableError as exc:
            raise OcrUnavailableError(f"ML OCR service unavailable: {exc}") from exc
        except MLServiceHTTPError as exc:
            if exc.status == 503:
                raise OcrUnavailableError(
                    f"ML OCR engine not ready (HTTP 503): {exc.body[:200]}"
                ) from exc
            raise OcrRuntimeError(
                f"ML OCR request failed (HTTP {exc.status}): {exc.body[:200]}"
            ) from exc
        except MLServiceError as exc:
            raise OcrMalformedError(f"ML OCR output unusable: {exc}") from exc

        payload = {
            "rec_texts": raw.get("texts"),
            "rec_scores": raw.get("scores"),
            "rec_boxes": raw.get("boxes"),
        }
        try:
            return _parse_predict_output(payload, self._dpi)
        except OcrError:
            raise
        except Exception as exc:  # pragma: no cover - defensive; parser raises OcrError
            raise OcrMalformedError(
                f"ML OCR output failed validation ({type(exc).__name__}): {exc}"
            ) from exc
