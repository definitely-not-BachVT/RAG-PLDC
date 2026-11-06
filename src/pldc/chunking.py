"""Turn parsed textbook sections and slides into retrieval chunks."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PROCESSED_DIR

CHUNK_CHARS = 1000
OVERLAP_CHARS = 200  # whole sentences only, so usually one or two
MIN_TAIL_CHARS = 250  # a shorter last piece of a section is merged into the one before
SKIP_KINDS = {"references", "review_questions"}  # bibliography / unanswered questions add noise
SENTENCE_END_RE = re.compile(r"(?<=[.;!?])\s+(?=[\"“(]?[A-ZĐÀ-Ỹ0-9])")
# Sub-parts inside a section: "a. Tài sản ...", "*. Quyền chiếm hữu ...", "• Điều kiện hưởng ..."
SUBHEAD_RE = re.compile(r"^([a-zđ]\.|\*\.?|•)\s*(\S.*)")  # not "a)": that lists a law's clauses
SUBHEAD_MAX_WORDS = 10
# A slide whose first line is a bullet or a sentence has no real title of its own.
SLIDE_FRAGMENT_RE = re.compile(r"^([❖➢•*+\-]|Là\s|[a-zđ])")


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


def _sentences(text: str, size: int) -> list[str]:
    """Split a paragraph into sentences; a sentence longer than size is cut on a space."""
    out = []
    parts = SENTENCE_END_RE.split(text)
    for i in range(len(parts) - 1, 0, -1):  # "a. Bồi thường..." is a list marker, not a sentence
        if len(parts[i - 1]) <= 3:
            parts[i - 1:i + 1] = [parts[i - 1] + " " + parts[i]]
    for s in parts:
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            out.append(s[:cut])
            s = s[cut:].lstrip()
        if s:
            out.append(s)
    return out


def _subheading(para: str) -> tuple[int, str] | None:
    """(level, title) if the paragraph opens a sub-part inside a section, e.g. "a. Tài sản Tài
    sản là vật..." -> (1, "Tài sản"). The PDF runs the title into the first sentence, so the title
    ends before the next capitalized word."""
    m = SUBHEAD_RE.match(para)
    if not m:
        return None
    words = m.group(2).split()
    n = 1
    while n < len(words) and not words[n][:1].isupper() and not words[n - 1].endswith(":"):
        n += 1
    title = " ".join(words[:n]).rstrip(":")
    # A bullet of plain content ("• Bằng 50% mức tiền lương ... khác.") is not a title.
    if n > SUBHEAD_MAX_WORDS or title.endswith((".", ";", ",")):
        return None
    return (2 if m.group(1) == "•" else 1), title


def _split(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[tuple[str, str]]:
    """Pack paragraphs into ~size-char pieces of whole sentences.

    Returns (context, piece) pairs: context names the sub-part the piece starts in (a long
    section such as "8.2.9.8 Bảo hiểm xã hội bắt buộc" spans 20 pages of "a. Chế độ ốm đau",
    "b. Chế độ thai sản"...), so a piece from the middle still says what it is about. The last
    sentences of each piece (up to `overlap` chars) are repeated at the start of the next.
    """
    units: list[Unit] = []
    heads: dict[int, str] = {}
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if sub := _subheading(para):
            level, title = sub
            heads = {lv: t for lv, t in heads.items() if lv < level} | {level: title}
        context = " > ".join(heads[lv] for lv in sorted(heads))
        units += [Unit(s, context, i == 0) for i, s in enumerate(_sentences(para, size))]

    pieces: list[tuple[list[Unit], int]] = []  # (units, how many lead units repeat the last piece)
    cur: list[Unit] = []
    carried = 0
    for u in units:
        if len(cur) > carried and len(_join(cur + [u])) > size:
            pieces.append((cur, carried))
            carried = 0
            while carried < len(cur) and len(_join(cur[len(cur) - carried - 1:])) <= overlap:
                carried += 1
            cur = cur[len(cur) - carried:]
        cur.append(u)
    if len(cur) > carried:
        pieces.append((cur, carried))
    # A short tail reads better as the end of the previous piece than as a chunk of its own.
    if len(pieces) > 1 and len(_join(pieces[-1][0][pieces[-1][1]:])) < MIN_TAIL_CHARS:
        tail, skip = pieces.pop()
        pieces[-1] = (pieces[-1][0] + tail[skip:], pieces[-1][1])
    return [(us[0].context, _join(us)) for us, _ in pieces]


@dataclass(frozen=True)
class Unit:
    sentence: str
    context: str  # sub-headings open at this sentence
    para_start: bool


def _join(units: list[Unit]) -> str:
    return "".join(("\n" if u.para_start else " ") + u.sentence for u in units).strip()


def _pages(start: int, end: int) -> str:
    return f"tr. {start}" if start == end else f"tr. {start}-{end}"


def _chapter_heading(chapter: int, title: str) -> str:
    return f"Chương {chapter}. {title}"


def textbook_chunks(path: Path) -> list[Chunk]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        if s["kind"] in SKIP_KINDS or not s["text"].strip():
            continue
        heading = [_chapter_heading(s["chapter"], s["chapter_title"])] + s["heading_path"][1:]
        citation = f"Giáo trình, mục {s['section_id']} ({_pages(s['page_start'], s['page_end'])})"
        for n, (context, piece) in enumerate(_split(s["text"])):
            if not any(ch.isalpha() for ch in piece):  # e.g. a "-----" rule under a chapter title
                continue
            last = context.rsplit(" > ", 1)[-1]
            opens_part = last and last in piece[:len(last) + 4]  # piece starts at its sub-heading
            path = heading + ([context] if context and not opens_part else [])
            chunks.append(Chunk(
                id=f"{s['id']}#{n}", source="giaotrinh", citation=citation,
                text=" > ".join(path) + f"\n{piece}", body=piece, chapter=s["chapter"],
                section_id=s["section_id"], page_start=s["page_start"], page_end=s["page_end"],
                source_file=s["source_file"], pdf_page_start=s["pdf_page_start"],
                pdf_page_end=s["pdf_page_end"],
            ))
    return chunks


def slide_chunks(path: Path, chapter_titles: dict[int, str] | None = None) -> list[Chunk]:
    """One chunk per slide, headed by its chapter and topic.

    Many slides are short bullet lists whose title alone is ambiguous ("Đối tượng điều chỉnh"
    appears for four branches of law), and continuation slides start with a bullet instead of a
    title; those take the title of the last slide that had one.
    """
    chunks, topic = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        title = s["title"]
        if " là " in title:  # a definition slide: "Tội phạm là hành vi..." is about "Tội phạm"
            topic = title.split(" là ", 1)[0]
        elif not SLIDE_FRAGMENT_RE.match(title):
            topic = title
        parts = ["Slide"]
        if s.get("chapter") and chapter_titles and s["chapter"] in chapter_titles:
            parts.append(_chapter_heading(s["chapter"], chapter_titles[s["chapter"]]))
        if topic and not s["text"].startswith(topic):
            parts.append(topic)
        chunks.append(Chunk(
            id=s["id"], source="slide", citation=f"Slide bài giảng, trang {s['slide']}",
            text=" > ".join(parts) + f"\n{s['text']}", body=s["text"], chapter=s.get("chapter"),
            slide=s["slide"], source_file=s["source_file"], pdf_page_start=s["slide"],
            pdf_page_end=s["slide"],
        ))
    return chunks


def load_all_chunks(processed_dir: Path = PROCESSED_DIR) -> list[Chunk]:
    sections = processed_dir / "textbook_sections.jsonl"
    titles = {}
    for line in sections.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        titles.setdefault(s["chapter"], s["chapter_title"])
    return textbook_chunks(sections) + slide_chunks(processed_dir / "slides.jsonl", titles)
