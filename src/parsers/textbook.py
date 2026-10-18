"""Parser for the PTIT lecture notes (giáo trình), split into numbered sections.

Each heading ("1.1", "1.1.2", "1.1.2.1", "CÂU HỎI ÔN TẬP CHƯƠNG n", ...) starts a new
section that runs until the next heading. The book's own numbering has mistakes
(e.g. "1.6.1.2" under "1.1.6", "7.1.2" used twice), so we nest by depth, not by value.
"""
from __future__ import annotations

import re
from pathlib import Path

import pypdfium2 as pdfium

from .common import TextbookSection, clean_inline, clean_text

SOURCE = "giaotrinh"
CHAPTER_RE = re.compile(r"^CHƯƠNG\s*(\d+)\s*:?\s*(.*)$")
SECTION_RE = re.compile(r"^(\d+(?:\s?\.\s?\d+){1,3})[\s.]+([^\W\d_].*)$")  # also "1.2.1.2 .Bản chất"
REVIEW_RE = re.compile(r"^CÂU HỎI ÔN TẬP CHƯƠNG\s*\d*", re.I)
REFS_RE = re.compile(r"^TÀI LIỆU THAM KHẢO\s*$")
TOC_LEADER_RE = re.compile(r"\.{5,}|…{2,}")
PAGE_NO_RE = re.compile(r"^\d{1,3}$")
BULLET_RE = re.compile(r"^([-+*•●○■□▪]|[a-zđ]\)|\d+\)|Điều \d+)")
MAX_HEADING_LEN = 160


def _page_lines(doc: pdfium.PdfDocument) -> list[tuple[int, int | None, list[str]]]:
    """(pdf page, printed page, body lines) for every page."""
    out = []
    for i in range(len(doc)):
        lines = [ln for ln in clean_text(doc[i].get_textpage().get_text_range()).split("\n") if ln]
        printed = None
        for idx in (0, -1):  # page number is the first or last extracted line
            if lines and PAGE_NO_RE.match(lines[idx]):
                printed = int(lines.pop(idx))
                break
        out.append((i + 1, printed, lines))
    return out


def _join_paragraphs(lines: list[str]) -> str:
    paras: list[str] = []
    for ln in lines:
        new_para = (not paras or BULLET_RE.match(ln)
                    or (paras[-1].endswith((".", ":", ";", "?", "!")) and ln[:1].isupper()))
        if new_para:
            paras.append(ln)
        else:
            paras[-1] += " " + ln
    return "\n".join(paras)


def _section_id(raw: str) -> str:
    return re.sub(r"\s", "", raw).rstrip(".")


def parse(pdf: Path) -> list[TextbookSection]:
    doc = pdfium.PdfDocument(str(pdf))
    pages = _page_lines(doc)

    # Body starts at the first "CHƯƠNG 1" that is not on a table-of-contents page.
    start = next(i for i, (_, _, lines) in enumerate(pages)
                 if any(CHAPTER_RE.match(l) and l.startswith("CHƯƠNG 1") for l in lines)
                 and not any(TOC_LEADER_RE.search(l) for l in lines))

    sections: list[dict] = []
    chapter, chapter_title = 0, ""
    path: list[tuple[int, str]] = []  # (depth, "1.1 Title")
    cur: dict | None = None

    def open_section(sec_id: str, title: str, kind: str, depth: int, pdf_page: int, printed: int | None):
        nonlocal cur, path
        path = [p for p in path if p[0] < depth] + [(depth, f"{sec_id} {title}".strip())]
        cur = {"section_id": sec_id, "title": title, "kind": kind, "chapter": chapter,
               "chapter_title": chapter_title, "heading_path": [p[1] for p in path],
               "lines": [], "pages": [(pdf_page, printed)]}
        sections.append(cur)

    for pdf_page, printed, lines in pages[start:]:
        if cur and cur["pages"][-1][0] != pdf_page:
            cur["pages"].append((pdf_page, printed))
        i = 0
        while i < len(lines):
            ln = lines[i]
            m_ch = CHAPTER_RE.match(ln)
            m_sec = SECTION_RE.match(ln)
            if m_ch:
                chapter = int(m_ch.group(1))
                title = m_ch.group(2)
                # Chapter titles wrap onto further all-caps lines.
                while i + 1 < len(lines) and lines[i + 1].isupper() and not SECTION_RE.match(lines[i + 1]):
                    i += 1
                    title += " " + lines[i]
                chapter_title = clean_inline(title).strip(" ,-")
                open_section(f"{chapter}", chapter_title, "chapter_intro", 1, pdf_page, printed)
            elif REVIEW_RE.match(ln):
                open_section(f"{chapter}.review", "Câu hỏi ôn tập", "review_questions", 2, pdf_page, printed)
            elif REFS_RE.match(ln):
                open_section(f"{chapter}.refs", "Tài liệu tham khảo", "references", 2, pdf_page, printed)
            elif (m_sec and len(ln) <= MAX_HEADING_LEN and m_sec.group(2)[0].isupper()
                  and _section_id(m_sec.group(1)).split(".")[0] == str(chapter)):
                sec_id = _section_id(m_sec.group(1))
                title = m_sec.group(2).strip()
                # Long headings wrap; a lowercase next line is the rest of the title.
                if len(ln) > 70 and i + 1 < len(lines) and lines[i + 1][:1].islower():
                    i += 1
                    title += " " + lines[i]
                open_section(sec_id, title, "content", sec_id.count(".") + 1, pdf_page, printed)
            elif cur is not None:
                cur["lines"].append(ln)
            i += 1

    out: list[TextbookSection] = []
    for n, s in enumerate(sections):
        text = _join_paragraphs(s["lines"])
        if not text:  # parent headings with no own text live on in children's heading_path
            continue
        printed = [p for _, p in s["pages"] if p is not None]
        out.append(TextbookSection(
            id=f"{SOURCE}-{n:04d}-{s['section_id']}", source=SOURCE, source_file=pdf.name,
            chapter=s["chapter"], chapter_title=s["chapter_title"], section_id=s["section_id"],
            title=s["title"], heading_path=s["heading_path"], kind=s["kind"], text=text,
            page_start=printed[0] if printed else 0, page_end=printed[-1] if printed else 0,
            pdf_page_start=s["pages"][0][0], pdf_page_end=s["pages"][-1][0],
        ))
    return out
