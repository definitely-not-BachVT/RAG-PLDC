"""Parser for the lecture slides: one record per slide, template boilerplate removed.

The deck was built on another course's template, so every slide repeats lines such as
"LỊCH SỬ CÁC HỌC THUYẾT KINH TẾ" and "TS. NGUYỄN VĂN A". We drop known template
lines, "Trang N" footers and anything that appears on a large share of slides.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from .common import Slide, clean_inline, clean_text

SOURCE = "slide"
TEMPLATE_RE = re.compile(
    r"^(BÀI GIẢNG MÔN|(LỊCH SỬ ?)?(CÁC )?HỌC THUYẾT KINH TẾ|PHÁP LU[ẬậẠ]T ĐẠI CƯƠNG|Mẫu \d+|"
    r"Mẫu tham khảo|trang bìa bài giảng|Slide show|Điện thoại/E-mail.*|Học kỳ/Năm biên soạn.*|"
    r".*GIẢNG VIÊN\s*:.*|.*NGUYỄN VĂN A.*|BỘ MÔN:?\s*KINH TẾ - KHOA QTKD\s*1.*|"
    r"Kinh tế - Khoa QTKD ?1|Bộ môn: Kinh tế - Khoa QTKD1|.*@ptit\.edu\.vn|TS\. Trần Đoàn Hạnh|"
    r"Trang\s*(\d+|#))$",
    re.I,
)
FREQUENT_SHARE = 0.3  # lines on more than 30% of slides are template, not content


def _join_wrapped(lines: list[str]) -> str:
    """Re-join lines broken inside a text box (next line starts lowercase)."""
    out: list[str] = []
    for ln in lines:
        if out and ln[:1].islower():
            out[-1] += " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def parse(pdf: Path) -> list[Slide]:
    doc = pdfium.PdfDocument(str(pdf))
    pages = [[ln for ln in clean_text(doc[i].get_textpage().get_text_range()).split("\n") if ln]
             for i in range(len(doc))]
    freq = Counter(ln for p in pages for ln in set(p))
    frequent = {ln for ln, n in freq.items() if n > FREQUENT_SHARE * len(pages)}

    slides = []
    for no, lines in enumerate(pages, start=1):
        is_cover = any(ln.startswith("HỌC VIỆN CÔNG NGHỆ") for ln in lines)
        body = [ln for ln in lines if ln not in frequent and not TEMPLATE_RE.match(ln)]
        if not body or is_cover:
            continue
        # Titles wrap inside their text box: "Tiền đề ra đời của" / "nhà nước".
        title, i = body[0], 1
        while i < len(body) and body[i][:1].islower() and len(title) < 80:
            title += " " + body[i]
            i += 1
        slides.append(Slide(
            id=f"{SOURCE}-{no:03d}", source=SOURCE, source_file=pdf.name, slide=no,
            chapter=None, title=clean_inline(title), text=_join_wrapped(body),
        ))
    return slides
