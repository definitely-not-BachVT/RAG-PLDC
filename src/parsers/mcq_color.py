"""Parser for the question bank whose correct option is printed in red (no text marker).

We stream characters with their fill colour, then split the stream into questions
("Câu c.s.n.") and options ("A." .. "D."). The answer is the option whose
non-space characters are mostly red.
"""
from __future__ import annotations

import re
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTAnno, LTChar, LTTextContainer, LTTextLine

from .common import Option, Question, clean_inline

SOURCE = "nganhang_tn"
# Usually at line start; a few are glued to the previous option ("... trước. Câu 1.3.14. ...").
QUESTION_RE = re.compile(
    r"(?:^[ \t]*Câu\s+(\d+)\.(\d+)\.(\d+)\s*[.:]?"  # line start: "Câu 2.1.16:" / "Câu  1.3.14."
    r"|(?<=\s)Câu\s+(\d+)\.(\d+)\.(\d+)\s*[.:])",  # glued: must end with '.' or ':'
    re.M,
)
CHAPTER_RE = re.compile(r"^Chương (\d+)\.\s*(.+)$", re.M)
FOOTER_RE = re.compile(r"\n[^\n]*, ngày\s+tháng.*", re.S)  # "Thái Nguyên, ngày tháng 9 năm 2023 ..."
LABELS = "ABCD"
RED_RATIO = 0.5


def _is_red(ch: LTChar) -> bool:
    c = ch.graphicstate.ncolor
    # Two shades are used for answers: (1,0,0) and (0.753,0,0). Orange (0.89,0.42,..) is not.
    return isinstance(c, (tuple, list)) and len(c) == 3 and c[0] > 0.5 and c[1] < 0.1 and c[2] < 0.1


def _char_stream(pdf: Path) -> tuple[str, list[bool], list[int]]:
    chars: list[str] = []
    red: list[bool] = []
    pages: list[int] = []
    for pno, page in enumerate(extract_pages(str(pdf)), start=1):
        lines = [ln for el in page if isinstance(el, LTTextContainer)
                 for ln in el if isinstance(ln, LTTextLine)]
        lines.sort(key=lambda l: (-round(l.y0), l.x0))
        for ln in lines:
            for item in ln:
                if isinstance(item, LTChar):
                    t, r = item.get_text(), _is_red(item)
                elif isinstance(item, LTAnno):  # inferred spaces/newlines
                    t, r = item.get_text(), False
                else:
                    continue
                chars.extend(t)
                red.extend([r] * len(t))
                pages.extend([pno] * len(t))
            if not chars or chars[-1] != "\n":
                chars.append("\n"); red.append(False); pages.append(pno)
    return "".join(chars), red, pages


def _find_label(text: str, label: str, pos: int, end: int) -> re.Match | None:
    """Prefer a label at line start; fall back to one after whitespace (inline 'B.')."""
    m = re.compile(rf"(?m)^[ \t]*{label}\.").search(text, pos, end)
    if m:
        return m
    return re.compile(rf"(?<=\s){label}\.").search(text, pos, end)


def _red_ratio(red: list[bool], text: str, start: int, end: int) -> float:
    idx = [i for i in range(start, end) if not text[i].isspace()]
    return sum(red[i] for i in idx) / len(idx) if idx else 0.0


def parse(pdf: Path) -> list[Question]:
    text, red, pages = _char_stream(pdf)
    footer = FOOTER_RE.search(text)
    if footer:
        text = text[:footer.start()]

    chapters = [(m.start(), int(m.group(1)), clean_inline(m.group(2))) for m in CHAPTER_RE.finditer(text)]
    heads = list(QUESTION_RE.finditer(text))
    questions = []
    for k, h in enumerate(heads):
        start, end = h.end(), heads[k + 1].start() if k + 1 < len(heads) else len(text)
        # A chapter heading between two questions belongs to the next one, not this block.
        for cpos, _, _ in chapters:
            if start < cpos < end:
                end = cpos
        chap = next(((n, t) for cpos, n, t in reversed(chapters) if cpos < h.start()), (None, None))

        marks, pos = [], start
        for label in LABELS:
            m = _find_label(text, label, pos, end)
            if not m:
                break
            marks.append((label, m.start(), m.end()))
            pos = m.end()

        flags = []
        stem_end = marks[0][1] if marks else end
        options, answers = [], []
        for i, (label, mstart, mend) in enumerate(marks):
            oend = marks[i + 1][1] if i + 1 < len(marks) else end
            options.append(Option(label, clean_inline(text[mend:oend])))
            if _red_ratio(red, text, mend, oend) > RED_RATIO:
                answers.append(label)
        if len(options) != len(LABELS):
            flags.append(f"option_count_{len(options)}")
        if len(answers) != 1:
            flags.append(f"answer_count_{len(answers)}")
        if flags:
            flags.append("needs_review")

        chap_no, sec_no, q_no = [g for g in h.groups() if g is not None]
        questions.append(Question(
            id=f"{SOURCE}-{chap_no}.{sec_no}.{q_no}", source=SOURCE, source_file=pdf.name,
            number=int(q_no), question=clean_inline(text[start:stem_end]),
            options=options, answers=answers, chapter=chap[0], chapter_title=chap[1],
            page=pages[h.start()], flags=flags,
        ))
    return questions
