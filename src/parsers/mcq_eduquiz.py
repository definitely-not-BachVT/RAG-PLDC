"""Parser for the EduQuiz export (430 questions, correct options prefixed with '*').

The export has no A/B/C/D labels and long lines wrap, so plain text cannot tell a
wrapped line from a new option. We use layout instead:
  * within a page, the stem is separated from the first option by a larger gap (~21pt
    vs ~18pt between lines);
  * an option line starts a new option for sure if it begins with '*' or the previous
    line was short (did not reach the wrap width). After a full-width line it is
    ambiguous: the source genuinely has full-width options followed by new options.
    We resolve ambiguous breaks so that the question ends up with 4 options when
    possible, preferring breaks where the next line starts with an uppercase letter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextContainer, LTTextLine

from .common import Option, Question, clean_inline, label_for

SOURCE = "eduquiz"
HEADER_RE = re.compile(r"^Câu (\d+)\W*(Một|Nhiều) đáp án")
BOILERPLATE_RE = re.compile(r"^(Phần \d+|Pháp Luật Đại Cương \(PTIT\).*|EduQuiz.*)$")
STEM_OPTION_GAP = 20.0  # pt; >= this between same-page lines means "options start"
FULL_LINE_X1 = 530.0  # wrapped lines end at x1 ~550-584; anything shorter did not wrap
EXPECTED_OPTIONS = 4
BODY_FONT_MIN = 10.0  # header/footer are 8pt
INLINE_LABEL_RE = re.compile(r"\s[A-D]\.\s?(?=[A-ZĐ])")  # e.g. "... công tác C.Người có ..."


@dataclass
class _Line:
    page: int
    y0: float
    x1: float
    first_word_w: float
    text: str


def _first_word_width(chars: list[LTChar]) -> float:
    w = 0.0
    for c in chars:
        if c.get_text().isspace():
            break
        w += c.width
    return w


def _read_lines(pdf: Path) -> tuple[list[_Line], float, float]:
    lines: list[_Line] = []
    space_w = 0.0
    for pno, page in enumerate(extract_pages(str(pdf)), start=1):
        page_lines = []
        for el in page:
            if not isinstance(el, LTTextContainer):
                continue
            for ln in el:
                if not isinstance(ln, LTTextLine):
                    continue
                chars = [c for c in ln if isinstance(c, LTChar)]
                if not chars or chars[0].size < BODY_FONT_MIN:
                    continue
                text = ln.get_text().strip()
                if not text or BOILERPLATE_RE.match(clean_inline(text)):
                    continue
                if not space_w:
                    sp = [c.width for c in chars if c.get_text() == " "]
                    space_w = sp[0] if sp else 0.0
                page_lines.append(_Line(pno, ln.y0, ln.x1, _first_word_width(chars), text))
        page_lines.sort(key=lambda l: -l.y0)
        lines.extend(page_lines)
    right_margin = max(l.x1 for l in lines)
    return lines, right_margin, space_w or 3.5


def _segment_options(lines: list[_Line], right_margin: float,
                     space_w: float) -> tuple[list[str], list[str]]:
    """Group option lines into options (see module docstring).

    Returns (options, forced) where forced lists the lines we *guessed* to start an option.
    """
    if not lines:
        return [], []
    # A few source lines hold two options glued by an inline label ("...công tác C.Người...").
    split: list[_Line] = []
    for ln in lines:
        parts = INLINE_LABEL_RE.split(ln.text)
        for j, part in enumerate(parts):
            last = j == len(parts) - 1
            # x1=0 on all but the last piece forces the next piece to start a new option.
            split.append(_Line(ln.page, ln.y0, ln.x1 if last else 0.0,
                               ln.first_word_w if j == 0 else 0.0, part.strip()))
    lines = split
    starts = {0}
    ambiguous: list[tuple[float, int]] = []
    for i in range(1, len(lines)):
        prev, ln = lines[i - 1], lines[i]
        if ln.text.startswith("*") or prev.x1 < FULL_LINE_X1:
            starts.add(i)
        else:
            slack = right_margin - (prev.x1 + space_w + ln.first_word_w)
            score = 2.0 * ln.text[:1].isupper() + slack / 50.0
            ambiguous.append((score, i))
    missing = EXPECTED_OPTIONS - len(starts)
    forced: list[int] = []
    if missing > 0:
        ambiguous.sort(reverse=True)
        forced = [i for _, i in ambiguous[:missing]]
        starts.update(forced)
    options: list[str] = []
    for i, ln in enumerate(lines):
        if i in starts:
            options.append(ln.text)
        else:
            options[-1] += " " + ln.text
    return options, [lines[i].text for i in forced]


def parse(pdf: Path) -> list[Question]:
    lines, right_margin, space_w = _read_lines(pdf)

    raw: list[dict] = []
    cur: dict | None = None
    prev: _Line | None = None
    for ln in lines:
        m = HEADER_RE.match(clean_inline(ln.text))
        if m:
            cur = {"number": int(m.group(1)), "multi": m.group(2) == "Nhiều",
                   "page": ln.page, "stem": [], "option_lines": []}
            raw.append(cur)
            prev = None
            state = "stem"
        elif cur is None:
            continue
        elif state == "stem":
            if not cur["stem"]:
                cur["stem"].append(ln.text)
            else:
                if prev.page == ln.page:
                    starts_options = (prev.y0 - ln.y0) >= STEM_OPTION_GAP
                else:  # gap is meaningless across a page break
                    starts_options = prev.x1 < FULL_LINE_X1
                if starts_options or ln.text.startswith("*"):
                    state = "options"
                    cur["option_lines"].append(ln)
                else:
                    cur["stem"].append(ln.text)
        else:
            cur["option_lines"].append(ln)
        prev = ln

    questions = []
    for q in raw:
        opts, answers = [], []
        texts, forced = _segment_options(q["option_lines"], right_margin, space_w)
        for i, t in enumerate(texts):
            t = clean_inline(t)
            if t.startswith("*"):
                answers.append(label_for(i))
                t = t[1:].strip()
            opts.append(Option(label_for(i), t))
        flags = []
        if len(opts) != EXPECTED_OPTIONS:
            flags.append(f"option_count_{len(opts)}")
        if q["multi"] and len(answers) < 2:
            flags.append("multi_answer_lt2")
        if not q["multi"] and len(answers) != 1:
            flags.append(f"answer_count_{len(answers)}")
        if any(INLINE_LABEL_RE.search(ln.text) for ln in q["option_lines"]):
            flags.append("split_inline_label")
        if any(t[:1].islower() for t in forced):  # a guessed option starting mid-sentence
            flags.append("forced_split_lowercase")
        if flags:
            flags.append("needs_review")
        if forced:  # informational: option boundary was guessed, verified by eye at build time
            flags.append("forced_split")
        questions.append(Question(
            id=f"{SOURCE}-{q['number']:03d}", source=SOURCE, source_file=pdf.name,
            number=q["number"], question=clean_inline(" ".join(q["stem"])),
            options=opts, answers=answers, multi_answer=q["multi"],
            page=q["page"], flags=flags,
        ))
    return questions
