"""Process-wide resources and per-user identity for the NiceGUI app."""
from __future__ import annotations

import logging
import uuid
from functools import cache

from nicegui import app

from pldc.config import INDEX_DIR
from pldc.embeddings import GeminiEmbedder
from pldc.llm import AllProvidersFailed, FallbackLLM, build_llm
from pldc.quiz import QuestionBank
from pldc.search import Retriever
from pldc.store import Store

log = logging.getLogger(__name__)


@cache
def bank() -> QuestionBank:
    return QuestionBank()


@cache
def store() -> Store:
    return Store()


@cache
def llm() -> FallbackLLM:
    return build_llm()


@cache
def retriever() -> Retriever:
    if not (INDEX_DIR / "manifest.json").exists():
        raise FileNotFoundError("Chưa có chỉ mục tìm kiếm. Chạy: python src/build_index.py")
    try:
        embedder = GeminiEmbedder()
    except RuntimeError:
        embedder = None  # no Gemini key: keyword search only
    return Retriever(embedder)


def user_id() -> str:
    """Per-browser id kept in NiceGUI's signed user storage (a cookie), so reloads keep progress."""
    return app.storage.user.setdefault("sid", uuid.uuid4().hex)


def error_message(exc: Exception) -> str:
    """Plain-language explanation of a failed AI call; says what happened and what to do."""
    log.warning("AI call failed: %r", exc)
    text = str(exc)
    if isinstance(exc, AllProvidersFailed) and "Chưa cấu hình" in text:
        return "Chưa có API key cho mô hình AI. Thêm key vào file .env rồi khởi động lại."
    if isinstance(exc, AllProvidersFailed) or any(s in text for s in ("503", "429", "UNAVAILABLE")):
        return "Các mô hình AI đang quá tải hoặc không kết nối được. Thử lại sau vài giây."
    return "Không tạo được lời giải do lỗi hệ thống. Thử lại, nếu vẫn lỗi hãy xem log máy chủ."
