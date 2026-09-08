"""Extraction package: provenance-first PDF ingestion (native + tables)."""

from .pymupdf import PageExtraction, PdfMetadata, TextBlock, extract_pdf
from .tables import StructuredTable, TableCell, extract_tables

__all__ = [
    "PageExtraction",
    "PdfMetadata",
    "StructuredTable",
    "TableCell",
    "TextBlock",
    "extract_pdf",
    "extract_tables",
]
