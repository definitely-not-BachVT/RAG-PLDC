"""Multiple-choice question bank and AI explanations."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import OVERRIDES_FILE, PROCESSED_DIR
from .llm import FallbackLLM, Message, StreamResult
from .prompts import EXPLAIN_SYSTEM, format_context
from .search import Hit, Retriever

BANKS = {
    "nganhang_tn": ("Ngân hàng câu hỏi (theo chương)", "questions_nganhang_tn.jsonl"),
    "eduquiz": ("Đề ôn PTIT (EduQuiz 2025)", "questions_eduquiz.jsonl"),
}


@dataclass
class Question:
    id: str
    source: str
    question: str
    options: list[dict]  # [{"label": "A", "text": ...}]
    answers: list[str]
    multi_answer: bool = False
    chapter: int | None = None
    chapter_title: str | None = None
    flags: list[str] = field(default_factory=list)

    def option_text(self, label: str) -> str:
        return next((o["text"] for o in self.options if o["label"] == label), "")

    def is_correct(self, chosen: list[str]) -> bool:
        return sorted(chosen) == sorted(self.answers)


class QuestionBank:
    def __init__(self, processed_dir: Path = PROCESSED_DIR, overrides_file: Path = OVERRIDES_FILE):
        overrides = {}
        if overrides_file.exists():
            overrides = json.loads(overrides_file.read_text(encoding="utf-8"))["questions"]
        self.questions: list[Question] = []
        for _, filename in BANKS.values():
            for line in (processed_dir / filename).read_text(encoding="utf-8").splitlines():
                raw = json.loads(line)
                fix = overrides.get(raw["id"], {})
                raw.update({k: fix[k] for k in ("options", "answers") if k in fix})
                q = Question(**{k: raw.get(k) for k in Question.__dataclass_fields__})
                if q.answers:  # questions without a key cannot be graded
                    self.questions.append(q)
        self.by_id = {q.id: q for q in self.questions}

    def chapters(self, source: str | None = None) -> list[tuple[int, str]]:
        seen = {(q.chapter, q.chapter_title) for q in self.questions
                if q.chapter is not None and (source is None or q.source == source)}
        return sorted(seen)

    def select(self, source: str | None = None, chapter: int | None = None,
               shuffle: bool = True, seed: int | None = None) -> list[str]:
        ids = [q.id for q in self.questions
               if (source is None or q.source == source) and (chapter is None or q.chapter == chapter)]
        if shuffle:
            random.Random(seed).shuffle(ids)
        return ids


def _format_question(q: Question) -> str:
    return q.question + "\n" + "\n".join(f"{o['label']}. {o['text']}" for o in q.options)


def explain(q: Question, chosen: list[str], retriever: Retriever, llm: FallbackLLM,
            k: int = 5) -> tuple[list[Hit], StreamResult]:
    correct_text = " ".join(q.option_text(a) for a in q.answers)
    chosen_text = " ".join(q.option_text(c) for c in chosen if c not in q.answers)
    hits = retriever.search(f"{q.question}\n{correct_text}\n{chosen_text}".strip(), k=k)
    status = "ĐÚNG" if q.is_correct(chosen) else "SAI"
    user = (f"{format_context(hits)}\n\n<cau_hoi>\n{_format_question(q)}\n</cau_hoi>\n"
            f"Đáp án đúng: {', '.join(q.answers)}\n"
            f"Người học đã chọn: {', '.join(chosen) or '(không chọn)'} ({status})")
    messages: list[Message] = [{"role": "system", "content": EXPLAIN_SYSTEM},
                               {"role": "user", "content": user}]
    return hits, llm.stream(messages)
