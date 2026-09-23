"""Render a stored PDF page to PNG, optionally highlighting one box.

The evidence layer records a bounding box for every unit, but a box is
only inspectable drawn on the page it came from. This closes the
provenance chain a reviewer actually follows: fact -> evidence -> page
-> the rectangle the text was read from.

PyMuPDF is imported lazily so importing this module never requires it.
"""

from __future__ import annotations

__all__ = ["RenderError", "render_page_png"]

#: Accent used for the highlight; matches the frontend's selection colour.
HIGHLIGHT_RGB = (0.886, 0.647, 0.173)

#: Drawn as a translucent fill plus a solid border, so the box is visible
#: over both dense text and whitespace.
HIGHLIGHT_FILL_OPACITY = 0.25
HIGHLIGHT_BORDER_WIDTH = 1.5


class RenderError(Exception):
    """The page could not be rendered (bad PDF, or page out of range)."""


def render_page_png(
    content: bytes,
    pdf_page_number: int,
    bbox: tuple[float, float, float, float] | None = None,
    dpi: int = 110,
) -> bytes:
    """Return ``pdf_page_number`` (0-based) as PNG bytes.

    ``bbox`` is in PDF point coordinates as stored on the evidence unit;
    the rectangle is drawn before rasterisation so it scales with ``dpi``
    automatically. A box outside the page is ignored rather than raising:
    a slightly wrong rectangle should not cost the reviewer the page.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RenderError("pymupdf is not installed") from exc

    if not content:
        raise RenderError("no stored PDF content")
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise RenderError(f"could not open stored PDF: {exc}") from exc
    try:
        if pdf_page_number < 0 or pdf_page_number >= document.page_count:
            raise RenderError(
                f"page {pdf_page_number} out of range "
                f"(document has {document.page_count} pages)"
            )
        page = document.load_page(pdf_page_number)
        if bbox is not None and len(tuple(bbox)) == 4:
            rect = pymupdf.Rect(*bbox)
            if not rect.is_empty and rect.intersects(page.rect):
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=HIGHLIGHT_RGB, fill=HIGHLIGHT_RGB)
                annot.set_opacity(HIGHLIGHT_FILL_OPACITY)
                annot.set_border(width=HIGHLIGHT_BORDER_WIDTH)
                annot.update()
        pixmap = page.get_pixmap(dpi=dpi)
        return bytes(pixmap.tobytes("png"))
    finally:
        document.close()
