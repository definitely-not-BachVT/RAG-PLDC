"""Q&A page laid out like a law Q&A handbook: "Hỏi" (violet, the learner) / "Đáp" (red margin, the tutor)."""
from __future__ import annotations

import html

from nicegui import run, ui

import shell
from pldc.chat import answer
from services import error_message, llm, retriever, store, user_id
from sources import SourcePanel, with_cites

SAMPLE_QUESTIONS = [
    "Nhà nước ra đời từ những tiền đề nào?",
    "Quy phạm pháp luật gồm những bộ phận nào?",
    "Vi phạm pháp luật có những dấu hiệu cơ bản nào?",
    "Quốc hội có những nhiệm vụ và quyền hạn gì?",
    "Di chúc hợp pháp cần những điều kiện gì?",
    "Người lao động được đơn phương chấm dứt hợp đồng khi nào?",
]


def to_sources(hits) -> list[dict]:
    return [{"citation": h.chunk.citation, "body": h.chunk.body, "file": h.chunk.source_file,
             "page_start": h.chunk.pdf_page_start, "page_end": h.chunk.pdf_page_end} for h in hits]


def build() -> None:
    sid, db = user_id(), store()
    msgs: list[dict] = db.load_chat(sid)  # {"role", "content", "sources", "provider"}
    state = {"busy": False}
    panel = SourcePanel()
    drawer = shell.frame()

    # ---- rail ------------------------------------------------------------------------
    @ui.refreshable
    def rail() -> None:
        with ui.column().classes("w-full gap-1 p-1"):
            shell.nav("chat", db.review_stats(sid)["due"])
            ui.label("Cuộc trò chuyện").classes("group-label")
            ui.label("Câu trả lời dựa trên giáo trình PTIT (2019) và slide bài giảng; "
                     "luôn đối chiếu nguồn được dẫn.").classes("text-sm px-2 text-[var(--graphite)]")
            if msgs:
                ui.button("Xóa cuộc trò chuyện", on_click=clear) \
                    .props("flat no-caps dense align=left").classes("chapter link mt-2")

    def clear() -> None:
        msgs.clear()
        db.clear_chat(sid)
        rail.refresh()
        thread.refresh()

    # ---- messages ----------------------------------------------------------------------
    def render_answer(idx: int, m: dict) -> None:
        sources = m.get("sources") or []
        with ui.element("div").classes("qa answer").props(f'data-scope="chat-{idx}"'):
            ui.html('<span class="who">Đáp</span>', sanitize=False).classes("contents")
            ui.markdown(with_cites(m["content"], len(sources))).classes("body")
            if sources:
                with ui.element("div").classes("sources"):
                    ui.label("Nguồn trích dẫn")
                    for i, src in enumerate(sources, start=1):
                        ui.html(f'<span class="num">{i}</span>{html.escape(src["citation"])}', sanitize=False) \
                            .props('role="button" tabindex="0"').classes("cursor-pointer") \
                            .on("click", lambda i=i, src=src: panel.show(i, src))
            ref = f"chat-{idx}"
            rating = db.get_rating(sid, ref)
            question = msgs[idx - 1]["content"] if idx else ""
            with ui.row().classes("items-center gap-1 mt-3 text-sm text-[var(--graphite)]"):
                ui.label("Câu trả lời này có hữu ích không?")
                for value, icon, aria in ((1, "thumb_up", "Có"), (0, "thumb_down", "Không")):
                    ui.button(icon=icon, on_click=lambda v=value: rate(ref, v, question, m)) \
                        .props(f'flat round dense size=sm aria-label="{aria}" '
                               f'color={"primary" if rating == value else "grey-7"}')
                if m.get("provider"):
                    ui.label(f"Trả lời bởi {m['provider']}").classes("ml-3 text-xs")

    def render_question(text: str) -> None:
        with ui.element("div").classes("qa ask"):
            ui.html(f'<span class="who">Hỏi</span><div class="body">{html.escape(text)}</div>',
                    sanitize=False).classes("contents")

    def rate(ref: str, value: int, question: str, m: dict) -> None:
        db.save_feedback(sid, ref, "chat", value, question, m["content"], m.get("provider"))
        ui.notify("Đã ghi nhận đánh giá", position="bottom", timeout=1500)
        thread.refresh()

    @ui.refreshable
    def thread() -> None:
        if not msgs:
            empty_state()
            return
        for idx, m in enumerate(msgs):
            if m["role"] == "user":
                render_question(m["content"])
            else:
                render_answer(idx, m)
        if msgs[-1]["role"] == "user" and not state["busy"]:  # e.g. the answer failed, then a reload
            with ui.element("div").classes("notice"):
                ui.label("Câu hỏi này chưa có câu trả lời.")
                ui.button("Trả lời câu này", on_click=generate).props("flat no-caps").classes("btn btn-red mt-2")

    def empty_state() -> None:
        ui.html('<h1 class="question">Hỏi điều bạn chưa rõ trong môn học.</h1>'
                '<p class="lede">Câu trả lời trích từ giáo trình và slide bài giảng PTIT, kèm số trang '
                'để bạn mở trang gốc đối chiếu.</p>', sanitize=False)
        with ui.element("div").classes("suggest"):
            for q in SAMPLE_QUESTIONS:
                with ui.element("button").props('type="button"').on("click", lambda q=q: send(q)):
                    ui.html(html.escape(q), sanitize=False).classes("contents")

    # ---- sending & streaming -------------------------------------------------------------
    async def send(text: str) -> None:
        # The clicked button (a suggestion, "Thử lại") is deleted by the redraws below;
        # run in the page's context so later UI calls don't use a dead element's slot.
        with page:
            text = text.strip()
            if not text or state["busy"]:
                return
            state["busy"] = True  # before the redraw, so it doesn't show "chưa có câu trả lời"
            composer.value = ""
            msgs.append({"role": "user", "content": text})
            db.add_chat_message(sid, len(msgs) - 1, msgs[-1])
            thread.refresh()
            rail.refresh()
            await generate()

    async def generate() -> None:
        """Answer the last user message (also used by "Thử lại" after a failure)."""
        # The clicked button (a suggestion, "Thử lại") is deleted by the redraws below;
        # run in the page's context so later UI calls don't use a dead element's slot.
        with page:
            state["busy"] = True
            send_btn.disable()
            question = msgs[-1]["content"]
            history = [{"role": m["role"], "content": m["content"]} for m in msgs[:-1]]
            live.clear()
            with live, ui.element("div").classes("qa answer"):
                ui.html('<span class="who">Đáp</span>', sanitize=False).classes("contents")
                body = ui.markdown("_Đang tra giáo trình…_").classes("body")
            scroll_to_end()
            try:
                res = await run.io_bound(answer, question, history, retriever(), llm())
                text, it = "", iter(res.stream.chunks)
                while (chunk := await run.io_bound(next, it, None)) is not None:
                    text += chunk
                    body.set_content(text)
            except Exception as exc:  # noqa: BLE001 - plain-language notice with a retry button
                live.clear()
                with live, ui.element("div").classes("notice"):
                    ui.label(error_message(exc) + " Câu hỏi của bạn vẫn được giữ lại.")
                    ui.button("Thử lại", on_click=generate).props("flat no-caps").classes("btn btn-red mt-2")
            else:
                msgs.append({"role": "assistant", "content": text, "sources": to_sources(res.hits),
                             "provider": res.stream.provider})
                db.add_chat_message(sid, len(msgs) - 1, msgs[-1])
                live.clear()
                thread.refresh()
                rail.refresh()
            finally:
                state["busy"] = False
                send_btn.enable()
                scroll_to_end()

    def scroll_to_end() -> None:
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

    async def on_cite(e) -> None:
        scope, n = e.args.get("scope", ""), int(e.args["n"])
        if not scope.startswith("chat-"):
            return
        idx = int(scope.split("-")[1])
        sources = (msgs[idx].get("sources") or []) if idx < len(msgs) else []
        if 1 <= n <= len(sources):
            await panel.show(n, sources[n - 1])

    ui.on("cite", on_cite)

    # ---- layout ------------------------------------------------------------------------------
    with drawer:
        rail()
    with ui.element("main").classes("sheet") as page:
        ui.html('<div class="sheet-head"><span>Hỏi đáp</span><span>Giáo trình và slide PTIT</span></div>'
                '<div class="progress" style="background:var(--rule)"></div>', sanitize=False)
        thread()
        live = ui.element("div")
        with ui.element("div").classes("composer"):
            composer = ui.textarea(placeholder="Ví dụ: Thừa kế theo di chúc có những điều kiện gì?") \
                .props('autogrow borderless input-style="max-height: 9rem" aria-label="Câu hỏi của bạn"') \
                .classes("w-full")
            composer.on("keydown.enter.exact.prevent", lambda: send(composer.value))
            with ui.row().classes("w-full items-center justify-between mt-1"):
                ui.html('<span class="hint"><kbd>Enter</kbd> gửi, <kbd>Shift</kbd>+<kbd>Enter</kbd> xuống dòng</span>',
                        sanitize=False)
                send_btn = ui.button("Gửi câu hỏi", on_click=lambda: send(composer.value)) \
                    .props("unelevated no-caps").classes("btn btn-violet")
    if msgs:
        scroll_to_end()
