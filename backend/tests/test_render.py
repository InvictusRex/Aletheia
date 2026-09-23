"""Page rendering tests: the last link in the provenance chain.

A bounding box is only inspectable drawn on the page it came from, so
these cover the render itself and the failure modes that must not cost
the reviewer the page.
"""

import pymupdf
import pytest

from app.extraction.render import RenderError, render_page_png


def _pdf(pages: int = 2) -> bytes:
    doc = pymupdf.open()
    try:
        for index in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"Synthetic page {index}")
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _is_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def test_renders_a_page_as_png():
    assert _is_png(render_page_png(_pdf(), 0))


def test_renders_with_a_highlight_box():
    assert _is_png(render_page_png(_pdf(), 0, bbox=(60.0, 60.0, 200.0, 90.0)))


def test_highlight_changes_the_rendered_bytes():
    plain = render_page_png(_pdf(), 0)
    boxed = render_page_png(_pdf(), 0, bbox=(60.0, 60.0, 200.0, 90.0))
    assert plain != boxed, "the box must actually be drawn"


def test_a_box_outside_the_page_is_ignored_not_fatal():
    """A slightly wrong rectangle should not cost the reviewer the page."""
    assert _is_png(render_page_png(_pdf(), 0, bbox=(9000.0, 9000.0, 9100.0, 9100.0)))


@pytest.mark.parametrize("page", [-1, 5])
def test_a_page_out_of_range_is_reported(page):
    with pytest.raises(RenderError, match="out of range"):
        render_page_png(_pdf(), page)


def test_empty_content_is_reported():
    with pytest.raises(RenderError, match="no stored PDF"):
        render_page_png(b"", 0)


def test_unopenable_content_is_reported():
    with pytest.raises(RenderError, match="could not open"):
        render_page_png(b"not a pdf at all", 0)


def test_dpi_changes_the_rendered_size():
    small = render_page_png(_pdf(), 0, dpi=50)
    large = render_page_png(_pdf(), 0, dpi=200)
    assert len(large) > len(small)
