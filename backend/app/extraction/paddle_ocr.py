"""PaddleOCR adapter (task T-O2): OCR fallback for scanned PDFs.

Document-agnostic: no dataset, metric, year, or filename logic lives here.
Target engine is PaddleOCR 3.x (``predict`` API).

Lazy-dependency rules (this module must import cleanly without PaddleOCR):
- ``paddleocr`` is imported ONLY inside functions, so importing this module
  never fails when PaddleOCR is not installed. When the engine is actually
  needed but ``paddleocr`` cannot be imported, :class:`OcrUnavailableError`
  is raised with a clear message.
- ``numpy`` (which ships with PaddlePaddle) is imported lazily inside
  :meth:`PaddleOCRProvider.extract_page`.
- :func:`paddle_ocr_available` never triggers a model download; it only
  inspects the local PaddleX model cache and fails closed to native
  extraction on any doubt.

Coordinate convention: Paddle pixel space shares the PDF orientation
(top-left origin), so pixel boxes map to PDF points by dividing by
``dpi / 72`` (see :func:`map_boxes_to_pdf_points`).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pymupdf

try:  # Frozen contract owned by the sibling agent; prefer it when present.
    from app.extraction.ocr_provider import (
        OcrBlock,
        OcrError,
        OcrInitError,
        OcrMalformedError,
        OcrModelMissingError,
        OcrProvider,
        OcrResult,
        OcrRuntimeError,
        OcrUnavailableError,
    )
except ImportError:  # pragma: no cover - concurrent-write shim, see below.
    # Temporary spec-exact shim for the window in which ocr_provider.py has
    # not landed yet. It mirrors the frozen contract
    # (``OcrBlock(text, bbox|None, confidence|None)``,
    # ``OcrResult(blocks, provider)``, ``OcrError`` + five subclasses,
    # ``OcrProvider`` Protocol with ``extract_page``) so that the
    # paddle-independent paths stay importable and testable. Once the real
    # sibling module exists, the ``try`` branch above wins and this shim is
    # never used. Re-verify against ocr_provider.py when it lands.
    from dataclasses import dataclass, field
    from typing import Protocol

    @dataclass
    class OcrBlock:  # type: ignore[no-redef]
        """One OCR text block with optional PDF-space bbox and confidence."""

        text: str
        bbox: tuple[float, float, float, float] | None = None
        confidence: float | None = None

    @dataclass
    class OcrResult:  # type: ignore[no-redef]
        """Per-page OCR result."""

        blocks: list = field(default_factory=list)
        provider: str = "paddleocr"

    class OcrError(Exception):  # type: ignore[no-redef]
        """Base class for all OCR failures."""

    class OcrUnavailableError(OcrError):  # type: ignore[no-redef]
        """The OCR backend (paddleocr package) is not installed."""

    class OcrInitError(OcrError):  # type: ignore[no-redef]
        """The OCR engine failed to initialize."""

    class OcrModelMissingError(OcrError):  # type: ignore[no-redef]
        """Model weights are missing and downloads are not allowed."""

    class OcrRuntimeError(OcrError):  # type: ignore[no-redef]
        """The OCR engine failed at inference time."""

    class OcrMalformedError(OcrError):  # type: ignore[no-redef]
        """The OCR engine returned output this adapter cannot parse."""

    class OcrProvider(Protocol):  # type: ignore[no-redef]
        """Structural contract for per-page OCR providers."""

        def extract_page(
            self, pdf_bytes: bytes, pdf_page_number: int
        ) -> OcrResult: ...


__all__ = [
    "PaddleOCRProvider",
    "map_boxes_to_pdf_points",
    "paddle_ocr_available",
    "render_page_to_png",
]

#: Provider tag recorded on every :class:`OcrResult` from this module.
PROVIDER_TAG = "paddleocr"

#: Substrings (lowercased) suggesting an init failure is really about
#: missing model weights rather than a broken install. Heuristic only;
#: both outcomes are :class:`OcrError` subclasses either way.
_MISSING_MODEL_HINTS = (
    "paddlex",
    "official_models",
    ".pdmodel",
    ".pdiparams",
    "checkpoint",
    "weight",
    "not found",
    "no such file",
    "missing",
    "download",
)


def _to_float(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if unusable.

    Rejects bools, strings/bytes, ``NaN``/``inf``, and anything that is
    not numerically convertible (including ``None``). Accepts int/float
    as well as numeric scalars such as numpy numbers via ``float()``.
    Never raises.
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ArithmeticError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _point_coords(point: object) -> tuple[float, float] | None:
    """Return ``(x, y)`` floats for a 2-element point, else ``None``."""
    if isinstance(point, (str, bytes, bytearray)):
        return None
    try:
        seq = list(point)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(seq) != 2:
        return None
    x = _to_float(seq[0])
    y = _to_float(seq[1])
    if x is None or y is None:
        return None
    return (x, y)


def _map_single_box(box: object, scale: float) -> tuple | None:
    """Map one pixel box to an ordered PDF-point bbox, else ``None``.

    Accepts ``[x_min, y_min, x_max, y_max]`` or a 4-point quad
    ``[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`` (collapsed to its
    axis-aligned bounding rectangle). Never raises.
    """
    if box is None or isinstance(box, (str, bytes, bytearray)):
        return None
    try:
        seq = list(box)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(seq) != 4:
        return None
    if all(isinstance(v, (list, tuple)) for v in seq):
        points = [_point_coords(p) for p in seq]
        if any(p is None for p in points):
            return None
        xs = [p[0] for p in points if p is not None]
        ys = [p[1] for p in points if p is not None]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    else:
        vals = [_to_float(v) for v in seq]
        if any(v is None for v in vals):
            return None
        nums = [v for v in vals if v is not None]
        x0, y0, x1, y1 = nums
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None  # Zero/negative area: never invent geometry.
    return (x0 / scale, y0 / scale, x1 / scale, y1 / scale)


def map_boxes_to_pdf_points(boxes, dpi: int) -> list[tuple | None]:
    """Map Paddle pixel boxes to PDF-point bboxes.

    Pure function: needs no paddle installation and never raises. Paddle
    pixel space shares the PDF orientation (top-left origin), so each
    coordinate is divided by ``dpi / 72`` and ordered so that
    ``x0 <= x1`` and ``y0 <= y1``.

    Each box is either ``[x_min, y_min, x_max, y_max]`` or a 4-point
    quad ``[[x1, y1], ...]``. Per-box failures (wrong length,
    non-numeric entries, ``NaN``/``inf``, zero/negative area) yield
    ``None`` for that box. Non-list top-level input is treated as
    malformed per box: ``None`` per entry when a length is available,
    otherwise ``[]``. An invalid ``dpi`` likewise yields ``None``\\ s
    instead of raising.
    """
    try:
        count = len(boxes)
    except TypeError:
        return []
    if not isinstance(boxes, list):
        return [None] * count
    scale_raw = _to_float(dpi)
    if scale_raw is None or scale_raw <= 0.0:
        return [None] * len(boxes)
    scale = scale_raw / 72.0
    if not math.isfinite(scale) or scale <= 0.0:
        return [None] * len(boxes)
    return [_map_single_box(box, scale) for box in boxes]


def render_page_to_png(
    pdf_bytes: bytes, pdf_page_number: int, dpi: int = 200
) -> bytes:
    """Render a single PDF page to PNG bytes via pymupdf.

    Raises:
        ValueError: On empty/non-bytes input, unopenable PDF bytes, an
            invalid page number or dpi, or an out-of-range page. Mirrors
            the ``ValueError`` conventions of ``pymupdf.py``.
    """
    if isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
    if not isinstance(pdf_bytes, bytes) or len(pdf_bytes) == 0:
        raise ValueError("render_page_to_png requires non-empty PDF bytes")
    if isinstance(pdf_page_number, bool) or not isinstance(
        pdf_page_number, int
    ):
        raise ValueError(
            "render_page_to_png requires an int pdf_page_number, "
            f"got {type(pdf_page_number).__name__}"
        )
    if _to_float(dpi) is None or float(dpi) <= 0:  # type: ignore[arg-type]
        raise ValueError(f"render_page_to_png requires dpi > 0, got {dpi!r}")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(
            f"unable to open PDF stream ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        if pdf_page_number < 0 or pdf_page_number >= len(doc):
            raise ValueError(
                f"page {pdf_page_number} out of range "
                f"(0..{len(doc) - 1} for {len(doc)} pages)"
            )
        pixmap = doc[pdf_page_number].get_pixmap(dpi=dpi)
        return bytes(pixmap.tobytes("png"))
    finally:
        doc.close()


def _paddlex_models_dir() -> Path:
    """Return the expected PaddleX ``official_models`` cache directory."""
    override = os.environ.get("PADDLEX_HOME", "").strip()
    base = Path(override) if override else Path.home() / ".paddlex"
    return base / "official_models"


def _paddlex_models_cached() -> bool:
    """True if the PaddleX model cache holds at least one file.

    Any doubt (missing dir, unreadable tree, traversal error) returns
    ``False`` so callers fail closed to native extraction.
    """
    try:
        models_dir = _paddlex_models_dir()
    except Exception:
        return False
    try:
        if not models_dir.is_dir():
            return False
        for child in models_dir.rglob("*"):
            try:
                if child.is_file():
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def paddle_ocr_available(
    lang: str = "en", allow_model_download: bool = False
) -> tuple[bool, str]:
    """Check whether the PaddleOCR backend can run without downloading.

    Never triggers a model download. Returns ``(False, reason)`` when the
    ``paddleocr`` package is not importable, when model weights are not
    cached and ``allow_model_download`` is ``False``, or when any check
    raises. Returns ``(True, detail)`` only when the package imports and
    either cached weights are found or downloads are explicitly allowed
    (in which case weights may still download on first engine use).
    """
    try:
        try:
            import paddleocr  # noqa: F401  (presence check only, no init)
        except Exception as exc:
            return False, (
                f"paddleocr not importable ({type(exc).__name__}): {exc}; "
                "install PaddleOCR 3.x to enable the OCR fallback "
                f"(lang={lang!r})"
            )
        cached = _paddlex_models_cached()
        models_dir = _paddlex_models_dir()
        if cached:
            return True, (
                "paddleocr importable; cached PaddleX models found under "
                f"{models_dir} (lang={lang!r})"
            )
        if not allow_model_download:
            return False, (
                "PaddleX model weights not cached under "
                f"{models_dir} and allow_model_download=False; refusing to "
                f"download (lang={lang!r}). Failing closed to native "
                "extraction."
            )
        return True, (
            "paddleocr importable; models not cached under "
            f"{models_dir} but allow_model_download=True, so weights may "
            f"download on first engine use (lang={lang!r})"
        )
    except Exception as exc:
        return False, (
            "PaddleOCR availability check failed "
            f"({type(exc).__name__}): {exc}"
        )


def _looks_like_missing_model(exc: BaseException) -> bool:
    """Heuristic: does this init error read like missing model weights?"""
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in message for hint in _MISSING_MODEL_HINTS)


def _payload_field(payload: object, names: tuple[str, ...]) -> object | None:
    """Return the first present named field of a predict-output payload.

    Supports plain dicts, duck-typed mappings with ``.get``, and
    attribute-style result objects. Returns ``None`` when absent.
    """
    getter = getattr(payload, "get", None)
    if isinstance(payload, dict) or callable(getter):
        for name in names:
            try:
                value = payload.get(name)  # type: ignore[union-attr]
            except Exception:
                value = None
            if value is not None:
                return value
    for name in names:
        try:
            value = getattr(payload, name)
        except Exception:
            continue
        if value is not None and not callable(value):
            return value
    return None


def _parse_predict_output(raw: object, dpi: int) -> OcrResult:
    """Parse PaddleOCR 3.x ``predict`` output into an :class:`OcrResult`.

    Expects the ``rec_texts`` / ``rec_scores`` / ``rec_boxes`` shape but
    stays defensive: only the overlap of mismatched-length sequences is
    zipped, missing boxes yield ``None`` bboxes, and empty texts are
    skipped. A zero-text prediction is a valid empty result; output with
    no usable text field at all raises :class:`OcrMalformedError`.
    """
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            raise OcrMalformedError(
                "PaddleOCR predict returned an empty output list"
            )
        try:
            payload = raw[0]
        except Exception as exc:
            raise OcrMalformedError(
                f"PaddleOCR predict output is not indexable "
                f"({type(exc).__name__}): {exc}"
            ) from exc
    else:
        payload = raw
    if payload is None:
        raise OcrMalformedError("PaddleOCR predict returned no output")

    texts = _payload_field(payload, ("rec_texts", "texts"))
    if texts is None:
        raise OcrMalformedError(
            "PaddleOCR predict output has no rec_texts field; "
            f"keys/type: {type(payload).__name__}"
        )
    if isinstance(texts, str):
        texts = [texts]
    try:
        text_list = list(texts)  # type: ignore[arg-type]
    except TypeError as exc:
        raise OcrMalformedError(
            f"PaddleOCR rec_texts is not a sequence "
            f"({type(exc).__name__}): {exc}"
        ) from exc

    scores = _payload_field(payload, ("rec_scores", "scores"))
    boxes = _payload_field(payload, ("rec_boxes", "rec_polys", "boxes"))
    try:
        score_list = list(scores) if scores is not None else []  # type: ignore[arg-type]
    except TypeError:
        score_list = []
    if isinstance(boxes, str) or boxes is None:
        box_items: list = []
    else:
        try:
            box_items = list(boxes)  # type: ignore[arg-type]
        except TypeError:
            box_items = []
    mapped = (
        map_boxes_to_pdf_points(box_items, dpi) if box_items else []
    )

    blocks: list[OcrBlock] = []
    for index, text in enumerate(text_list):
        if text is None:
            continue
        try:
            cleaned = str(text).strip()
        except Exception:
            continue
        if not cleaned:
            continue
        confidence: float | None = None
        if index < len(score_list):
            raw_conf = _to_float(score_list[index])
            # ocr_provider.OcrBlock enforces 0..1 (ValueError otherwise),
            # so out-of-range engine scores degrade to None, never raise.
            if raw_conf is not None and 0.0 <= raw_conf <= 1.0:
                confidence = raw_conf
        bbox = mapped[index] if index < len(mapped) else None
        blocks.append(
            OcrBlock(text=cleaned, bbox=bbox, confidence=confidence)
        )
    return OcrResult(blocks=blocks, provider=PROVIDER_TAG)


def _png_bytes_to_rgb(png_bytes: bytes, np: object) -> object:
    """Decode rendered PNG bytes to a contiguous RGB numpy array.

    Uses only pymupdf + numpy (no cv2/Pillow dependency). Raises
    :class:`OcrRuntimeError` when decoding fails.
    """
    try:
        doc = pymupdf.open(stream=png_bytes, filetype="png")
    except Exception as exc:
        raise OcrRuntimeError(
            f"failed to decode rendered page PNG ({type(exc).__name__}): "
            f"{exc}"
        ) from exc
    try:
        if len(doc) == 0:
            raise OcrRuntimeError("rendered page PNG produced no pages")
        pix = doc[0].get_pixmap()
        try:
            flat = np.frombuffer(pix.samples, dtype=np.uint8)  # type: ignore[union-attr]
        except Exception as exc:
            raise OcrRuntimeError(
                f"failed to wrap rendered pixels ({type(exc).__name__}): "
                f"{exc}"
            ) from exc
        channels = pix.n
        height, width = pix.height, pix.width
        if channels not in (1, 3, 4) or height <= 0 or width <= 0:
            raise OcrRuntimeError(
                "unexpected rendered pixmap geometry "
                f"(w={width}, h={height}, n={channels})"
            )
        if flat.size != height * width * channels:
            raise OcrRuntimeError(
                f"pixel buffer size {flat.size} does not match geometry "
                f"(w={width}, h={height}, n={channels})"
            )
        array = flat.reshape(height, width, channels).copy()
        if channels == 1:
            rgb = np.repeat(array, 3, axis=2)  # type: ignore[union-attr]
        elif channels == 3:
            rgb = array
        else:
            rgb = array[:, :, :3]  # Drop alpha.
        return np.ascontiguousarray(rgb)  # type: ignore[union-attr]
    finally:
        doc.close()


class PaddleOCRProvider(OcrProvider):
    """PaddleOCR 3.x OCR fallback; conforms to the ``OcrProvider`` Protocol.

    Stores ``lang`` / ``dpi`` / ``allow_model_download`` only — the engine
    is constructed lazily on the first :meth:`extract_page` call and then
    cached on ``self``.

    Timeouts / retries: the local PaddleOCR engine exposes no per-call
    timeout knob, so inference here is bounded only by the surrounding
    process (whatever the caller enforces). Retries do not apply locally:
    transient GPU/OOM failures surface as :class:`OcrRuntimeError` and the
    caller must NOT retry them blindly — route back to native evidence or
    mark the page failed instead.
    """

    def __init__(
        self,
        lang: str = "en",
        dpi: int = 200,
        allow_model_download: bool = False,
        enable_mkldnn: bool = False,
    ) -> None:
        self._lang = lang
        self._dpi = dpi
        self._allow_model_download = allow_model_download
        # oneDNN/MKLDNN is intentionally disabled: PaddlePaddle's CPU
        # oneDNN path fails PIR attribute conversion for the OCR models
        # (ConvertPirAttribute2RuntimeAttribute / onednn_instruction.cc),
        # so plain CPU kernels are used instead. Verified plumbing:
        # PaddleOCR validates `enable_mkldnn` as a constructor kwarg and
        # the static runner only calls config.enable_mkldnn() when the
        # run mode requests it.
        self._enable_mkldnn = enable_mkldnn
        self._engine: object | None = None

    def _get_engine(self) -> object:
        """Construct (once) and return the PaddleOCR engine.

        Raises:
            OcrUnavailableError: ``paddleocr`` is not installed.
            OcrModelMissingError: Weights are not cached and downloads
                are disallowed, or init failed for weight-related reasons.
            OcrInitError: Any other engine construction failure.
        """
        if self._engine is not None:
            return self._engine
        try:
            import paddleocr
        except Exception as exc:
            raise OcrUnavailableError(
                "paddleocr is not installed; install PaddleOCR 3.x to "
                f"enable OCR (lang={self._lang!r}). Original import error "
                f"({type(exc).__name__}): {exc}"
            ) from exc
        if not self._allow_model_download and not _paddlex_models_cached():
            raise OcrModelMissingError(
                "PaddleOCR model weights are not cached under "
                f"{_paddlex_models_dir()} (lang={self._lang!r}) and "
                "allow_model_download=False; refusing to download models. "
                "Pre-fetch the PaddleX official models or construct with "
                "allow_model_download=True."
            )
        try:
            engine = paddleocr.PaddleOCR(
                lang=self._lang, enable_mkldnn=self._enable_mkldnn
            )
        except Exception as exc:
            if _looks_like_missing_model(exc):
                raise OcrModelMissingError(
                    "PaddleOCR model weights appear to be missing "
                    f"(lang={self._lang!r}, "
                    f"{type(exc).__name__}): {exc}"
                ) from exc
            raise OcrInitError(
                "failed to initialize PaddleOCR "
                f"(lang={self._lang!r}, {type(exc).__name__}): {exc}"
            ) from exc
        self._engine = engine
        return engine

    def extract_page(
        self, pdf_bytes: bytes, pdf_page_number: int
    ) -> OcrResult:
        """OCR one PDF page and return blocks in PDF-point coordinates.

        Raises:
            ValueError: Invalid input bytes / page number, or the page
                is out of range (mirrors ``pymupdf.py`` conventions).
            OcrUnavailableError: ``paddleocr`` is not installed.
            OcrModelMissingError: Weights missing and downloads disallowed.
            OcrInitError: Engine construction failed.
            OcrRuntimeError: Rendering decode or engine inference failed.
            OcrMalformedError: Engine output could not be parsed at all.
        """
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
        engine = self._get_engine()

        try:
            import numpy as np  # Ships with PaddlePaddle; lazy on purpose.
        except Exception as exc:
            raise OcrRuntimeError(
                "numpy is required to feed page images to PaddleOCR "
                f"({type(exc).__name__}): {exc}"
            ) from exc
        image = _png_bytes_to_rgb(png_bytes, np)

        try:
            raw = engine.predict(image)  # type: ignore[union-attr]
        except OcrError:
            raise
        except Exception as exc:
            raise OcrRuntimeError(
                f"PaddleOCR inference failed on page {pdf_page_number} "
                f"({type(exc).__name__}): {exc}"
            ) from exc
        return _parse_predict_output(raw, self._dpi)
