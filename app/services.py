"""Shared resources and UI helpers for the Streamlit pages."""
from __future__ import annotations

import logging
import re
import uuid

import streamlit as st

from pldc.config import INDEX_DIR
from pldc.embeddings import GeminiEmbedder
from pldc import pages
from pldc.llm import AllProvidersFailed, FallbackLLM, build_llm
from pldc.quiz import QuestionBank
from pldc.search import Hit, Retriever
from pldc.store import Store

log = logging.getLogger(__name__)
SID_RE = re.compile(r"^[0-9a-f]{32}$")


@st.cache_resource
def get_llm() -> FallbackLLM:
    return build_llm()


@st.cache_resource
def get_retriever() -> Retriever:
    if not (INDEX_DIR / "manifest.json").exists():
        raise FileNotFoundError("Chưa có chỉ mục tìm kiếm. Hãy chạy: python src/build_index.py")
    try:
        embedder = GeminiEmbedder()
    except RuntimeError:
        embedder = None  # no Gemini key: keyword search only
    return Retriever(embedder)


@st.cache_resource
def get_bank() -> QuestionBank:
    return QuestionBank()


@st.cache_resource
def get_store() -> Store:
    return Store()


def require(factory):
    """Call a resource factory; show the error and stop the page if it fails."""
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - surface setup problems in the UI
        st.error(str(exc))
        st.stop()


def get_sid() -> str:
    """Session id kept in the URL (?sid=...) so a page reload restores the user's data.

    Switching pages clears query params, so we also keep it in session_state and put it back.
    """
    sid = st.session_state.get("sid") or st.query_params.get("sid", "")
    if not SID_RE.match(sid):
        sid = uuid.uuid4().hex
    st.session_state.sid = sid
    if st.query_params.get("sid") != sid:
        st.query_params["sid"] = sid
    return sid


# ---- errors ----------------------------------------------------------------------------
def friendly_error(exc: Exception, retry_hint: str) -> None:
    """Explain a failure in plain Vietnamese; technical details go in a collapsed box."""
    log.warning("LLM call failed: %r", exc)
    text = str(exc)
    if isinstance(exc, AllProvidersFailed) and "Chưa cấu hình" in text:
        msg = "Chưa cấu hình API key cho mô hình AI. Hãy kiểm tra file `.env`."
    elif isinstance(exc, AllProvidersFailed) or any(s in text for s in ("503", "429", "UNAVAILABLE")):
        msg = "Các mô hình AI đang quá tải hoặc tạm thời không kết nối được. Vui lòng thử lại sau vài giây."
    else:
        msg = "Đã có lỗi khi tạo câu trả lời."
    st.warning(f"{msg} {retry_hint}", icon="⏳")
    with st.expander("Chi tiết kỹ thuật"):
        st.code(text or type(exc).__name__, language=None)


# ---- sources & feedback ------------------------------------------------------------------
def to_sources(hits: list[Hit]) -> list[dict]:
    """Plain dicts so sources can be stored in SQLite and shown again after a reload."""
    return [{"citation": h.chunk.citation, "body": h.chunk.body, "file": h.chunk.source_file,
             "page_start": h.chunk.pdf_page_start, "page_end": h.chunk.pdf_page_end}
            for h in hits]


def _short(citation: str) -> str:
    """'Giáo trình, mục 6.1.3 (tr. 130-133)' -> 'Mục 6.1.3'; 'Slide bài giảng, trang 99' -> 'Slide 99'."""
    if m := re.search(r"mục ([\w.]+)", citation):
        return f"Mục {m.group(1)}"
    if m := re.search(r"trang (\d+)", citation):
        return f"Slide {m.group(1)}"
    return citation[:20]


@st.dialog("📚 Nguồn tham khảo", width="large")
def _source_dialog(num: int, src: dict) -> None:
    st.markdown(f"**[{num}] {src['citation']}**")
    tab_text, tab_page = st.tabs(["Đoạn trích", "Trang gốc (PDF)"])
    with tab_text:
        st.text(src["body"])
    with tab_page:
        if not src.get("file"):
            st.info("Nguồn này được lưu trước khi có tính năng xem trang gốc. Hãy hỏi lại để xem.")
            return
        start, end = src["page_start"], src["page_end"]
        located = pages.locate_page(src["file"], start, end, src["body"])
        total = pages.page_count(src["file"])
        page = st.number_input(f"Trang trong file PDF (đoạn trích bắt đầu ở trang {located})",
                               min_value=1, max_value=total, value=located, step=1,
                               key=f"page_{src['file']}_{start}_{num}")
        st.image(pages.render_page(src["file"], int(page)), width="stretch")


def show_sources(sources: list[dict], key: str) -> None:
    """A row of [n] buttons matching the [n] markers in the answer; each opens a dialog."""
    if not sources:
        return
    with st.container(horizontal=True, gap="small", key=f"src_{key}"):
        st.caption("📚 Nguồn:")
        for i, src in enumerate(sources, start=1):
            if st.button(f"[{i}] {_short(src['citation'])}", key=f"src_{key}_{i}",
                         help=src["citation"], type="tertiary"):
                _source_dialog(i, src)


def feedback_row(ref: str, kind: str, prompt: str, response: str, provider: str | None) -> None:
    """👍/👎 under an AI answer, saved immediately; plus a small provider badge."""
    store, sid = get_store(), get_sid()
    key = f"fb_{ref}"

    def _save():
        store.save_feedback(sid, ref, kind, st.session_state[key], prompt, response, provider)

    left, right = st.columns([1, 4], vertical_alignment="center")
    with left:
        st.feedback("thumbs", key=key, default=store.get_rating(sid, ref), on_change=_save)
    if provider:
        right.caption(f"🤖 {provider}")
