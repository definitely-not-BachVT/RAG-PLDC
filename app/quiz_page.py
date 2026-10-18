"""Multiple-choice practice: option cards, "ask AI", spaced review of wrong answers, summary."""
import altair as alt
import pandas as pd
import streamlit as st

from pldc.quiz import BANKS, explain
from services import (feedback_row, friendly_error, get_bank, get_llm, get_retriever, get_sid,
                      get_store, require, show_sources, to_sources)

PRACTICE, REVIEW = "📚 Luyện tập", "🔁 Ôn câu sai"
ALL = "all"  # selectbox sentinel: Streamlit renders a None option as an empty placeholder
BAR_COLOR = "#2a78d6"  # single neutral series hue (not the red brand color, which reads as "bad")

# Left-align option cards (buttons inside the keyed container) and give them some height.
st.html("""<style>
.st-key-opts button { justify-content: flex-start; text-align: left; min-height: 3rem; }
.st-key-opts button div, .st-key-opts button span { justify-content: flex-start; }
.st-key-opts button p { text-align: left; }
</style>""")

st.title("📝 Luyện trắc nghiệm")
bank = require(get_bank)
sid, store = get_sid(), get_store()

if "quiz" not in st.session_state:  # first run in this browser session: restore after a reload
    st.session_state.quiz = store.load_quiz(sid)
# Streamlit drops widget state when another page is shown, so re-seed the filter widgets
# from the quiz state; otherwise coming back would look like a filter change and reset it.
saved = st.session_state.quiz
saved_filters = [PRACTICE, ALL, ALL, True]
if saved and len(saved["filters"]) == 4:
    # Review mode stores no practice filters; keep practice defaults for when the user switches back.
    saved_filters = saved["filters"] if saved["filters"][0] == PRACTICE else [REVIEW, ALL, ALL, True]
for _k, _v in zip(("f_mode", "f_source", "f_chapter", "f_shuffle"), saved_filters):
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---- sidebar: mode & filters ---------------------------------------------------------
stats = store.review_stats(sid)
with st.sidebar:
    mode = st.radio("Chế độ", [PRACTICE, REVIEW], key="f_mode",
                    captions=["", f"{stats['due']} câu đến hạn · {stats['mastered']} câu đã thuộc"])
    source = chapter = ALL
    shuffle = False
    if mode == PRACTICE:
        source = st.selectbox("Bộ câu hỏi", [ALL, *BANKS], key="f_source",
                              format_func=lambda s: "Tất cả" if s == ALL else BANKS[s][0])
        if source == "nganhang_tn":
            chapter = st.selectbox("Chương", [ALL, *[c for c, _ in bank.chapters(source)]],
                                   key="f_chapter",
                                   format_func=lambda c: "Tất cả" if c == ALL else
                                   f"Chương {c}. {dict(bank.chapters(source))[c]}")
        shuffle = st.toggle("Xáo trộn câu hỏi", key="f_shuffle")
    restart = st.button("🔄 Làm bộ mới" if mode == PRACTICE else "🔄 Tải lại câu đến hạn",
                        width="stretch")


def new_order() -> list[str]:
    if mode == REVIEW:
        return [qid for qid in store.due_reviews(sid) if qid in bank.by_id]
    return bank.select(None if source == ALL else source, None if chapter == ALL else chapter, shuffle)


filters = [mode, source, chapter, shuffle]  # a list, so it round-trips through JSON
state = st.session_state.quiz
if state is None or state["filters"] != filters or restart:
    state = {"filters": filters, "order": new_order(), "pos": 0, "submitted": {},
             "explanations": {}, "view": "question", "label": None}
    store.save_quiz(sid, state)  # so a reload keeps the new mode/filters
# Questions can disappear from the bank (e.g. after re-parsing); drop stale ids.
state["order"] = [qid for qid in state["order"] if qid in bank.by_id]
state["pos"] = min(state["pos"], max(len(state["order"]) - 1, 0))
state.setdefault("view", "question")
st.session_state.quiz = state


def save() -> None:
    store.save_quiz(sid, state)


order = state["order"]
answered = {qid: c for qid, c in state["submitted"].items() if qid in bank.by_id}
n_correct = sum(bank.by_id[qid].is_correct(c) for qid, c in answered.items())
with st.sidebar:
    st.metric("Điểm bộ hiện tại", f"{n_correct}/{len(answered)}", help="Số câu đúng / số câu đã làm")
    if st.button("📊 Xem tổng kết", width="stretch", disabled=not answered):
        state["view"] = "summary"
        save()
        st.rerun()

if not order:
    if mode == REVIEW:
        st.success(f"🎉 Không có câu nào đến hạn ôn. Đang theo dõi {stats['active']} câu, "
                   f"đã thuộc {stats['mastered']} câu. Câu làm sai ở chế độ Luyện tập sẽ xuất hiện ở đây.")
    else:
        st.info("Không có câu hỏi nào với bộ lọc này.")
    st.stop()


# ---- summary view ----------------------------------------------------------------------
def group_name(q) -> str:
    return f"Chương {q.chapter}" if q.chapter else "Đề ôn PTIT"


def render_summary() -> None:
    st.subheader("📊 Tổng kết" + (f" · {state['label']}" if state.get("label") else ""))
    total = len(answered)
    c1, c2, c3 = st.columns(3)
    c1.metric("Đã làm", f"{total}/{len(order)}")
    c2.metric("Số câu đúng", n_correct)
    c3.metric("Tỉ lệ đúng", f"{n_correct / total:.0%}" if total else "–")

    rows = {}
    for qid, chosen in answered.items():
        q = bank.by_id[qid]
        r = rows.setdefault(group_name(q), {"Nhóm": group_name(q), "Đúng": 0, "Đã làm": 0,
                                           "order": q.chapter or 99})
        r["Đã làm"] += 1
        r["Đúng"] += q.is_correct(chosen)
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["order"])).drop(columns="order")
    df["Tỉ lệ đúng (%)"] = (100 * df["Đúng"] / df["Đã làm"]).round(0)
    df["Nhãn"] = df["Đúng"].astype(str) + "/" + df["Đã làm"].astype(str)

    if len(df) > 1:
        st.markdown("##### Tỉ lệ đúng theo chương")
        base = alt.Chart(df).encode(
            y=alt.Y("Nhóm:N", sort=list(df["Nhóm"]), title=None),
            x=alt.X("Tỉ lệ đúng (%):Q", scale=alt.Scale(domain=[0, 100]), title="Tỉ lệ đúng (%)",
                    axis=alt.Axis(grid=True, tickCount=5)),
            tooltip=["Nhóm", "Đúng", "Đã làm", "Tỉ lệ đúng (%)"],
        )
        bars = base.mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height={"band": 0.6})
        labels = base.mark_text(align="left", dx=4).encode(text="Nhãn:N")
        st.altair_chart((bars + labels).properties(height=36 * len(df) + 40), width="stretch")
    with st.expander("Bảng số liệu"):
        st.dataframe(df.drop(columns="Nhãn"), hide_index=True, width="stretch")

    wrong = [qid for qid, c in answered.items() if not bank.by_id[qid].is_correct(c)]
    st.markdown(f"##### Câu làm sai ({len(wrong)})")
    if not wrong:
        st.success("Bạn chưa làm sai câu nào trong bộ này.")
    for qid in wrong:
        q = bank.by_id[qid]
        with st.expander(f"❌ {q.question[:110]}{'…' if len(q.question) > 110 else ''}"):
            st.markdown(f"**Bạn chọn:** {', '.join(f'{c}. {q.option_text(c)}' for c in answered[qid])}")
            st.markdown(f"**Đáp án đúng:** {', '.join(f'{a}. {q.option_text(a)}' for a in q.answers)}")
            if st.button("Xem lại câu này", key=f"goto_{qid}"):
                state["pos"], state["view"] = order.index(qid), "question"
                save()
                st.rerun()

    left, right = st.columns(2)
    if left.button("⬅️ Quay lại làm bài", width="stretch"):
        state["view"] = "question"
        save()
        st.rerun()
    if right.button(f"🔁 Làm lại {len(wrong)} câu sai", type="primary", width="stretch",
                    disabled=not wrong):
        state.update(order=wrong, pos=0, submitted={}, view="question", label="Làm lại câu sai")
        save()
        st.rerun()


if state["view"] == "summary":
    render_summary()
    st.stop()

# ---- question view ---------------------------------------------------------------------
pos = state["pos"]
q = bank.by_id[order[pos]]
st.progress((pos + 1) / len(order), text=f"Câu {pos + 1}/{len(order)}"
            + (f" · {state['label']}" if state.get("label") else ""))
meta = BANKS[q.source][0] + (f" · Chương {q.chapter}" if q.chapter else "")
st.caption(f"{meta} · mã {q.id}")
st.markdown(f"#### {q.question}")

labels = [o["label"] for o in q.options]
fmt = lambda lab: f"**{lab}.** {q.option_text(lab)}"  # noqa: E731
pick_key = f"pick_{q.id}"


def toggle(lab: str) -> None:
    picked = st.session_state.get(pick_key, [])
    if q.multi_answer:
        st.session_state[pick_key] = [x for x in picked if x != lab] if lab in picked else picked + [lab]
    else:
        st.session_state[pick_key] = [] if picked == [lab] else [lab]


if q.id not in answered:
    picked = st.session_state.get(pick_key, [])
    if q.multi_answer:
        st.caption("Câu này có nhiều đáp án đúng, hãy chọn tất cả.")
    with st.container(key="opts"):
        for lab in labels:
            st.button(fmt(lab), key=f"opt_{q.id}_{lab}", width="stretch",
                      type="primary" if lab in picked else "secondary", on_click=toggle, args=(lab,))
    if st.button("Nộp bài", type="primary", disabled=not picked):
        state["submitted"][q.id] = picked
        store.record_answer(sid, q.id, q.is_correct(picked))  # feeds the review queue
        save()
        st.rerun()
else:
    chosen = answered[q.id]
    for lab in labels:
        if lab in q.answers:
            st.success(fmt(lab), icon="✅")
        elif lab in chosen:
            st.error(fmt(lab), icon="❌")
        else:
            st.container(border=True).markdown(fmt(lab))
    if q.is_correct(chosen):
        st.markdown("🎉 **Chính xác!**")
    else:
        st.markdown(f"Bạn chọn **{', '.join(chosen)}**, đáp án đúng là **{', '.join(q.answers)}**."
                    + (" Câu này đã được thêm vào danh sách ôn." if mode == PRACTICE else ""))

    # ---- ask AI ----------------------------------------------------------------------
    key = f"{q.id}|{','.join(chosen)}"
    cached = state["explanations"].get(key)
    if cached is None and st.button("🤖 Hỏi AI vì sao", type="primary"):
        with st.container(border=True):
            try:
                with st.spinner("Đang tra cứu giáo trình…"):
                    hits, result = explain(q, chosen, require(get_retriever), require(get_llm))
                text = st.write_stream(result.chunks)
            except Exception as exc:  # noqa: BLE001 - friendly message; the button stays for a retry
                friendly_error(exc, "Bấm **🤖 Hỏi AI vì sao** để thử lại.")
            else:
                state["explanations"][key] = {"text": text, "sources": to_sources(hits),
                                              "provider": result.provider}
                save()
                st.rerun()  # redraw from the cache so the source and feedback buttons appear
    elif cached is not None:
        with st.container(border=True):
            st.markdown(cached["text"])
            show_sources(cached["sources"], key=f"exp_{q.id}")
            feedback_row(f"explain-{key}", "explain", f"{q.id} chọn {', '.join(chosen)}",
                         cached["text"], cached["provider"])

# ---- navigation ----------------------------------------------------------------------
left, right = st.columns(2)
if left.button("⬅️ Câu trước", disabled=pos == 0, width="stretch"):
    state["pos"] -= 1
    save()
    st.rerun()
if pos < len(order) - 1:
    if right.button("Câu tiếp ➡️", width="stretch"):
        state["pos"] += 1
        save()
        st.rerun()
elif right.button("📊 Xem tổng kết", type="primary", width="stretch", disabled=not answered,
                  key="summary_end"):
    state["view"] = "summary"
    save()
    st.rerun()
