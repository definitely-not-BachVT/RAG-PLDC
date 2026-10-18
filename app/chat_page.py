"""Chatbot: Q&A grounded in the PTIT course material."""
import streamlit as st

from pldc.chat import answer
from services import (feedback_row, friendly_error, get_llm, get_retriever, get_sid, get_store,
                      require, show_sources, to_sources)

SAMPLE_QUESTIONS = [
    ("🏛️", "Nhà nước ra đời từ những tiền đề nào?"),
    ("📜", "Quy phạm pháp luật gồm những bộ phận nào?"),
    ("⚖️", "Vi phạm pháp luật có những dấu hiệu cơ bản nào?"),
    ("🗳️", "Quốc hội có những nhiệm vụ và quyền hạn gì?"),
    ("🏠", "Di chúc hợp pháp cần những điều kiện gì?"),
    ("👷", "Hợp đồng lao động có những loại nào?"),
]

st.title("💬 Hỏi đáp Pháp luật đại cương")
st.caption("Trả lời dựa trên giáo trình và slide bài giảng PTIT, có dẫn nguồn để bạn đối chiếu.")

sid = get_sid()
store = get_store()
if "chat" not in st.session_state:
    st.session_state.chat = store.load_chat(sid)  # restore after a page reload
chat: list[dict] = st.session_state.chat


def ask(question: str) -> None:
    """Queue a question; it is answered further down in this same script run."""
    msg = {"role": "user", "content": question}
    chat.append(msg)
    store.add_chat_message(sid, len(chat) - 1, msg)
    st.session_state.chat_pending = question
    st.session_state.chat_error = None


with st.sidebar:
    if st.button("🗑️ Xóa hội thoại", width="stretch", disabled=not chat):
        chat.clear()
        store.clear_chat(sid)
        st.session_state.chat_error = None
        st.rerun()

# ---- empty state: sample questions -------------------------------------------------------
if not chat:
    st.markdown("##### Bạn có thể bắt đầu với một câu hỏi gợi ý:")
    cols = st.columns(2)
    for i, (icon, q) in enumerate(SAMPLE_QUESTIONS):
        cols[i % 2].button(f"{icon} {q}", key=f"sample_{i}", width="stretch",
                           on_click=ask, args=(q,))

# ---- history -----------------------------------------------------------------------------
for idx, msg in enumerate(chat):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), key=f"chat{idx}")
            feedback_row(f"chat-{idx}", "chat", chat[idx - 1]["content"] if idx else "",
                         msg["content"], msg.get("provider"))

if question := st.chat_input("Ví dụ: Thừa kế theo di chúc có những điều kiện gì?"):
    ask(question)
    with st.chat_message("user"):
        st.markdown(question)

# ---- answer the pending question -------------------------------------------------------------
pending = st.session_state.get("chat_pending")
if pending:
    st.session_state.chat_pending = None
    history = [{"role": m["role"], "content": m["content"]} for m in chat[:-1]]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang tra cứu giáo trình…"):
                result = answer(pending, history, require(get_retriever), require(get_llm))
            text = st.write_stream(result.stream.chunks)
        except Exception as exc:  # noqa: BLE001 - shown as a friendly message with retry
            st.session_state.chat_error = (pending, exc)
        else:
            msg = {"role": "assistant", "content": text, "sources": to_sources(result.hits),
                   "provider": result.stream.provider}
            chat.append(msg)
            store.add_chat_message(sid, len(chat) - 1, msg)
            st.rerun()  # redraw through the history loop so feedback buttons appear

# ---- failed answer: explain and offer a retry ---------------------------------------------
if err := st.session_state.get("chat_error"):
    failed_question, exc = err
    with st.chat_message("assistant"):
        friendly_error(exc, "Câu hỏi của bạn vẫn được giữ lại.")
        if st.button("🔄 Thử lại", type="primary"):
            st.session_state.chat_error = None
            st.session_state.chat_pending = failed_question
            st.rerun()
