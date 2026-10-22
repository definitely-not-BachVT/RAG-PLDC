"""Question answering over the course material."""
from __future__ import annotations

from dataclasses import dataclass

from .config import get_settings
from .llm import FallbackLLM, Message, StreamResult
from .prompts import CHAT_SYSTEM, format_context
from .search import Hit, Retriever

MAX_HISTORY_TURNS = 6  # previous messages sent to the LLM
SHORT_FOLLOWUP_WORDS = 8


@dataclass
class ChatAnswer:
    hits: list[Hit]
    stream: StreamResult


def retrieval_query(question: str, history: list[Message]) -> str:
    """Short follow-ups ("còn trường hợp khác thì sao?") need the previous question for context."""
    prev = [m["content"] for m in history if m["role"] == "user"]
    if prev and len(question.split()) < SHORT_FOLLOWUP_WORDS:
        return f"{prev[-1]}\n{question}"
    return question


def answer(question: str, history: list[Message], retriever: Retriever, llm: FallbackLLM,
           k: int | None = None) -> ChatAnswer:
    hits = retriever.search(retrieval_query(question, history), k=k or get_settings().chat_top_k)
    messages: list[Message] = [{"role": "system", "content": CHAT_SYSTEM}]
    messages += history[-MAX_HISTORY_TURNS:]
    messages.append({"role": "user", "content": f"{format_context(hits)}\n\nCâu hỏi: {question}"})
    return ChatAnswer(hits, llm.stream(messages))
