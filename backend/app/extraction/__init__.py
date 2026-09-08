"""Phase 1 extraction package: provenance-first PDF ingestion."""

from .pymupdf import PageExtraction, PdfMetadata, TextBlock, extract_pdf

__all__ = ["PageExtraction", "PdfMetadata", "TextBlock", "extract_pdf"]
