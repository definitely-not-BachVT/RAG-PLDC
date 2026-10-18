"""Shared helpers for all PDF parsers."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# pdfminer spams warnings about broken font descriptors in the EduQuiz PDF.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

LATEX_MARK_RE = re.compile(r"\s*\[latext\]\s*")
DOLLAR_RE = re.compile(r"\$([^$]*)\$")
SPACES_RE = re.compile(r"[ \t ]+")


def clean_text(text: str) -> str:
    """NFC-normalize Vietnamese text and collapse whitespace on each line."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r", "").replace("￾", "(").replace("﻿", "")
    text = LATEX_MARK_RE.sub(" ", text)
    text = DOLLAR_RE.sub(r"\1", text)
    lines = [SPACES_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def clean_inline(text: str) -> str:
    """Like clean_text but joins everything into one line."""
    return " ".join(clean_text(text).split())


@dataclass
class Option:
    label: str
    text: str


@dataclass
class Question:
    id: str
    source: str
    source_file: str
    number: int
    question: str
    options: list[Option]
    answers: list[str]
    multi_answer: bool = False
    chapter: int | None = None
    chapter_title: str | None = None
    page: int | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class TextbookSection:
    id: str
    source: str
    source_file: str
    chapter: int
    chapter_title: str
    section_id: str
    title: str
    heading_path: list[str]
    kind: str  # "content" | "review_questions"
    text: str
    page_start: int  # printed page number
    page_end: int
    pdf_page_start: int
    pdf_page_end: int


@dataclass
class Slide:
    id: str
    source: str
    source_file: str
    slide: int
    chapter: int | None
    title: str
    text: str


def write_jsonl(path: Path, records: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def label_for(i: int) -> str:
    return "ABCDEFGH"[i]
