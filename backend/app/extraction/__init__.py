"""Extraction package: provenance-first PDF ingestion (native + tables + OCR)."""

from .ocr_provider import OcrProvider, OcrResult
from .paddle_ocr import PaddleOCRProvider
from .pymupdf import PageExtraction, PdfMetadata, TextBlock, extract_pdf
from .tables import StructuredTable, TableCell, extract_tables

__all__ = [
    "OcrProvider",
    "OcrResult",
    "PaddleOCRProvider",
    "PageExtraction",
    "PdfMetadata",
    "StructuredTable",
    "TableCell",
    "TextBlock",
    "extract_pdf",
    "extract_tables",
]
