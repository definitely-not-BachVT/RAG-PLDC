"""Render original PDF pages so users can check a cited passage against the source."""
from __future__ import annotations

import io
import re
import threading
import unicodedata
from functools import lru_cache

import pypdfium2 as pdfium

from .config import DATA_DIR

_LOCK = threading.Lock()  # pdfium is not thread-safe; Streamlit serves sessions on threads
_NON_WORD = re.compile(r"\W+")
SNIPPET_CHARS = 40


def _squash(text: str) -> str:
    """Compare text ignoring whitespace/punctuation (PDF line breaks and double spaces)."""
    return _NON_WORD.sub("", unicodedata.normalize("NFC", text).lower())


@lru_cache(maxsize=4)
def _doc(filename: str) -> pdfium.PdfDocument:
    path = (DATA_DIR / filename).resolve()
    if path.parent != DATA_DIR.resolve() or path.suffix.lower() != ".pdf":
        raise ValueError(f"not a data PDF: {filename}")
    return pdfium.PdfDocument(str(path))


def page_count(filename: str) -> int:
    with _LOCK:
        return len(_doc(filename))


@lru_cache(maxsize=512)
def _page_text(filename: str, page: int) -> str:
    with _LOCK:
        return _squash(_doc(filename)[page - 1].get_textpage().get_text_range())


def locate_page(filename: str, start: int, end: int, body: str) -> int:
    """The page in [start, end] where the passage begins (falls back to start)."""
    if start >= end:
        return start
    squashed = _squash(body)
    # The chunk may begin with overlap from the previous piece; try a few anchors.
    for offset in (0, len(squashed) // 3, 2 * len(squashed) // 3):
        snippet = squashed[offset:offset + SNIPPET_CHARS]
        if len(snippet) < 15:
            continue
        for page in range(start, end + 1):
            if snippet in _page_text(filename, page):
                return page
    return start


@lru_cache(maxsize=64)
def render_page(filename: str, page: int, scale: float = 1.5) -> bytes:
    """PNG bytes of one page (1-based)."""
    with _LOCK:
        image = _doc(filename)[page - 1].render(scale=scale).to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
