"""Turn parsed textbook sections and slides into retrieval chunks."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PROCESSED_DIR

CHUNK_CHARS = 1000
OVERLAP_CHARS = 150
SKIP_KINDS = {"references", "review_questions"}  # bibliography / unanswered questions add noise


@dataclass
class Chunk:
    id: str
    source: str  # "giaotrinh" | "slide"
    citation: str  # human-readable source shown to the user and the LLM
    text: str  # what gets embedded and searched (includes the heading path)
    body: str  # the passage itself
    chapter: int | None = None
    section_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    slide: int | None = None
    source_file: str | None = None  # PDF in data/, for showing the original page
    pdf_page_start: int | None = None  # 1-based PDF page numbers (not printed page numbers)
    pdf_page_end: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _split(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Pack paragraphs into ~size-char pieces; carry the tail of each piece into the next."""
    paras = [p for p in text.split("\n") if p.strip()]
    pieces: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > size:  # very long paragraph: hard-split on a space
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            paras_part, p = p[:cut], p[cut:].lstrip()
            if cur:
                pieces.append(cur)
            pieces.append(paras_part)
            cur = ""
        if cur and len(cur) + len(p) + 1 > size:
            pieces.append(cur)
            tail = cur[-overlap:]
            cur = tail[tail.find(" ") + 1:] + "\n" + p if overlap else p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        pieces.append(cur)
    return pieces


def _pages(start: int, end: int) -> str:
    return f"tr. {start}" if start == end else f"tr. {start}-{end}"


def textbook_chunks(path: Path) -> list[Chunk]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        if s["kind"] in SKIP_KINDS or not s["text"].strip():
            continue
        heading = " > ".join([f"Chương {s['chapter']}. {s['chapter_title']}"] + s["heading_path"][1:])
        citation = f"Giáo trình, mục {s['section_id']} ({_pages(s['page_start'], s['page_end'])})"
        for n, piece in enumerate(_split(s["text"])):
            if not any(ch.isalpha() for ch in piece):  # e.g. a "-----" rule under a chapter title
                continue
            chunks.append(Chunk(
                id=f"{s['id']}#{n}", source="giaotrinh", citation=citation,
                text=f"{heading}\n{piece}", body=piece, chapter=s["chapter"],
                section_id=s["section_id"], page_start=s["page_start"], page_end=s["page_end"],
                source_file=s["source_file"], pdf_page_start=s["pdf_page_start"],
                pdf_page_end=s["pdf_page_end"],
            ))
    return chunks


def slide_chunks(path: Path) -> list[Chunk]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        chunks.append(Chunk(
            id=s["id"], source="slide", citation=f"Slide bài giảng, trang {s['slide']}",
            text=f"Slide: {s['title']}\n{s['text']}", body=s["text"], slide=s["slide"],
            source_file=s["source_file"], pdf_page_start=s["slide"], pdf_page_end=s["slide"],
        ))
    return chunks


def load_all_chunks(processed_dir: Path = PROCESSED_DIR) -> list[Chunk]:
    return (textbook_chunks(processed_dir / "textbook_sections.jsonl")
            + slide_chunks(processed_dir / "slides.jsonl"))
