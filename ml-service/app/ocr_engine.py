from __future__ import annotations

import base64
import binascii
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

PROVIDER_TAG = "paddleocr"
DEFAULT_LANG = os.environ.get("OCR_LANGUAGE", "en")

_engines: dict[str, object] = {}
_engines_lock = threading.Lock()
_allow_download = os.environ.get("OCR_ALLOW_MODEL_DOWNLOAD", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _models_dir() -> Path:
    override = os.environ.get("PADDLEX_HOME", "").strip()
    base = Path(override) if override else Path.home() / ".paddlex"
    return base / "official_models"


def _models_cached() -> bool:
    try:
        models_dir = _models_dir()
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


def init_ocr(lang: str = DEFAULT_LANG, allow_download: bool | None = None) -> tuple[bool, str]:
    lang = (lang or DEFAULT_LANG).strip() or DEFAULT_LANG
    if allow_download is None:
        allow_download = _allow_download
    with _engines_lock:
        if lang in _engines:
            return True, f"paddleocr engine cached (lang={lang!r})"
        try:
            import paddleocr
        except Exception as exc:
            return False, f"paddleocr not installed ({type(exc).__name__}): {exc}"
        if not allow_download and not _models_cached():
            return False, (
                f"PaddleOCR weights not cached under {_models_dir()} "
                f"(lang={lang!r}) and downloads are disallowed"
            )
        try:
            _engines[lang] = paddleocr.PaddleOCR(lang=lang, enable_mkldnn=False)
        except Exception as exc:
            return False, f"PaddleOCR init failed (lang={lang!r}): {exc}"
        return True, f"paddleocr engine ready (lang={lang!r})"


def _decode_png(png_bytes: bytes) -> object:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy is required to decode page PNGs: {exc}") from exc
    try:
        import pymupdf
    except Exception as exc:
        raise RuntimeError(f"pymupdf is required to decode page PNGs: {exc}") from exc
    try:
        doc = pymupdf.open(stream=png_bytes, filetype="png")
    except Exception as exc:
        raise ValueError(f"unable to decode PNG ({type(exc).__name__}): {exc}") from exc
    try:
        if len(doc) == 0:
            raise ValueError("PNG produced no pages")
        pix = doc[0].get_pixmap()
        flat = np.frombuffer(pix.samples, dtype=np.uint8)
        channels, height, width = pix.n, pix.height, pix.width
        if channels not in (1, 3, 4) or height <= 0 or width <= 0:
            raise ValueError(f"unexpected pixmap geometry (w={width}, h={height}, n={channels})")
        if flat.size != height * width * channels:
            raise ValueError("pixel buffer size does not match geometry")
        array = flat.reshape(height, width, channels).copy()
        if channels == 1:
            array = array.repeat(3, axis=2)
        elif channels == 4:
            array = array[:, :, :3]
        return np.ascontiguousarray(array)
    finally:
        doc.close()


def _as_float_list(values: object) -> list:
    if values is None or isinstance(values, str):
        return []
    try:
        import numpy as np

        if isinstance(values, np.ndarray):
            values = values.tolist()
    except Exception:
        pass
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        return []
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            try:
                out.append([
                    [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)
                    for v in item
                ])
            except (TypeError, ValueError):
                out.append(None)
        else:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(None)
    return out


def ocr_png_base64(png_b64: str, lang: str = DEFAULT_LANG) -> dict:
    try:
        png_bytes = base64.b64decode(png_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"png_base64 is not valid base64: {exc}") from exc
    if not png_bytes:
        raise ValueError("png_base64 decodes to empty bytes")
    with _engines_lock:
        engine = _engines.get((lang or DEFAULT_LANG).strip() or DEFAULT_LANG)
    if engine is None:
        raise RuntimeError(f"no OCR engine initialised for lang={lang!r}")
    image = _decode_png(png_bytes)
    try:
        raw = engine.predict(image)  # type: ignore[union-attr]
    except Exception as exc:
        raise RuntimeError(
            f"PaddleOCR inference failed ({type(exc).__name__}): {exc}"
        ) from exc
    payload = raw[0] if isinstance(raw, (list, tuple)) else raw
    if payload is None:
        raise RuntimeError("PaddleOCR predict returned no output")

    def _field(names: tuple[str, ...]) -> object:
        if isinstance(payload, dict):
            for name in names:
                if payload.get(name) is not None:
                    return payload.get(name)
        for name in names:
            try:
                value = getattr(payload, name)
            except Exception:
                continue
            if value is not None and not callable(value):
                return value
        return None

    texts = _field(("rec_texts", "texts"))
    if texts is None:
        raise RuntimeError("PaddleOCR predict output has no rec_texts field")
    if isinstance(texts, str):
        texts = [texts]
    try:
        text_list = [str(t).strip() for t in list(texts)]  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError(f"PaddleOCR rec_texts is not a sequence: {exc}") from exc
    scores = _as_float_list(_field(("rec_scores", "scores")))
    boxes = _as_float_list(_field(("rec_boxes", "rec_polys", "boxes")))
    return {"texts": text_list, "scores": scores, "boxes": boxes}
