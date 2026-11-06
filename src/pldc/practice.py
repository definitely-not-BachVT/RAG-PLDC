"""A practice session (UI-independent): question order, answers, summary, retry, review mode.

The session is a plain JSON-able dict so it can be saved with Store.save_quiz().
"""
from __future__ import annotations

from dataclasses import dataclass

from .quiz import Question, QuestionBank
from .store import Store

PRACTICE, REVIEW = "practice", "review"


def new_session(bank: QuestionBank, store: Store, sid: str, mode: str = PRACTICE,
                source: str | None = None, chapter: int | None = None, shuffle: bool = True) -> dict:
    if mode == REVIEW:
        order = [qid for qid in store.due_reviews(sid) if qid in bank.by_id]
    else:
        order = bank.select(source, chapter, shuffle)
    return {"mode": mode, "source": source, "chapter": chapter, "shuffle": shuffle,
            "order": order, "pos": 0, "submitted": {}, "explanations": {},
            "view": "question", "label": None}


def load_session(bank: QuestionBank, store: Store, sid: str) -> dict:
    """The saved session, cleaned of questions that no longer exist; a new one if none."""
    s = store.load_quiz(sid)
    if not s or "mode" not in s:  # nothing saved, or saved by the old Streamlit page
        return new_session(bank, store, sid)
    s["order"] = [qid for qid in s["order"] if qid in bank.by_id]
    s["submitted"] = {q: c for q, c in s["submitted"].items() if q in bank.by_id}
    s["pos"] = min(s["pos"], max(len(s["order"]) - 1, 0))
    return s


def current(bank: QuestionBank, s: dict) -> Question | None:
    return bank.by_id[s["order"][s["pos"]]] if s["order"] else None


def submit(bank: QuestionBank, store: Store, sid: str, s: dict, chosen: list[str]) -> bool:
    q = current(bank, s)
    s["submitted"][q.id] = chosen
    correct = q.is_correct(chosen)
    store.record_answer(sid, q.id, correct)  # wrong answers enter the review queue
    return correct


def explanation_key(qid: str, chosen: list[str]) -> str:
    return f"{qid}|{','.join(chosen)}"


@dataclass
class Group:
    name: str
    correct: int
    done: int


@dataclass
class Summary:
    done: int
    correct: int
    groups: list[Group]
    wrong: list[str]


def summarize(bank: QuestionBank, s: dict) -> Summary:
    groups: dict[str, Group] = {}
    sort_key: dict[str, int] = {}
    wrong = []
    for qid, chosen in s["submitted"].items():
        q = bank.by_id[qid]
        name = f"Chương {q.chapter}" if q.chapter else "Đề ôn PTIT"
        sort_key[name] = q.chapter or 99
        g = groups.setdefault(name, Group(name, 0, 0))
        g.done += 1
        if q.is_correct(chosen):
            g.correct += 1
        else:
            wrong.append(qid)
    ordered = sorted(groups.values(), key=lambda g: sort_key[g.name])
    return Summary(done=sum(g.done for g in ordered), correct=sum(g.correct for g in ordered),
                   groups=ordered, wrong=wrong)


def retry_wrong(s: dict, wrong: list[str]) -> None:
    s.update(order=list(wrong), pos=0, submitted={}, view="question", label="Làm lại câu sai")
