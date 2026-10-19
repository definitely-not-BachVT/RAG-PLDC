"""Practice page: the exercise sheet, the red-pen grading, the teacher's comment (AI)."""
from __future__ import annotations

import html

from nicegui import run, ui

from pldc import practice
from pldc.quiz import BANKS, explain
from services import bank, error_message, llm, retriever, store, user_id
from sources import SourcePanel, with_cites
from theme import CIRCLE_SVG, CROSS_SVG, TICK_SVG

SOURCE_LABELS = {None: "Tất cả câu hỏi", "nganhang_tn": "Ngân hàng theo chương", "eduquiz": "Đề ôn PTIT 2025"}


def build() -> None:
    sid, qbank, db = user_id(), bank(), store()
    s = practice.load_session(qbank, db, sid)
    db.save_quiz(sid, s)  # a brand-new session must survive a reload too (keeps its shuffle)
    view = {"picked": [], "just_submitted": None, "busy": False}
    panel = SourcePanel()

    def save() -> None:
        db.save_quiz(sid, s)

    def restart(**changes) -> None:
        nonlocal s
        opts = {"mode": s["mode"], "source": s["source"], "chapter": s["chapter"],
                "shuffle": s["shuffle"], **changes}
        s = practice.new_session(qbank, db, sid, **opts)
        view["picked"] = []
        save()
        rail.refresh()
        sheet.refresh()

    def go(pos: int) -> None:
        s["pos"], s["view"] = pos, "question"
        view["picked"] = []
        save()
        sheet.refresh()

    # ---- left rail ---------------------------------------------------------------
    # Filled at the end of build(), once every handler it references exists.
    # value=None lets Quasar decide: open on desktop, closed (menu button) on phones.
    drawer = ui.left_drawer(value=None, fixed=True).classes("rail").props("width=280 breakpoint=900")

    @ui.refreshable
    def rail() -> None:
        with ui.column().classes("w-full gap-1 p-1"):
            rail_content()

    def rail_content() -> None:
        ui.html('<div class="wordmark">Pháp luật<br>đại cương</div>'
                '<div class="tagline">Ôn thi theo giáo trình PTIT</div>', sanitize=False).classes("mb-5")
        due = db.review_stats(sid)["due"]
        nav = [("Luyện tập", practice.PRACTICE, ""),
               ("Ôn câu sai", practice.REVIEW, f"{due} câu" if due else "")]
        for label, mode, count in nav:
            active = " active" if s["mode"] == mode else ""
            ui.html(f'<span>{label}</span><span class="count">{count}</span>', sanitize=False) \
                .classes(f"nav-link{active}").props('role="button" tabindex="0"') \
                .on("click", lambda m=mode: restart(mode=m))
        ui.html('<span>Hỏi đáp</span><span class="count" style="color:var(--graphite)">sắp có</span>',
                sanitize=False).classes("nav-link soon")

        if s["mode"] == practice.PRACTICE:
            ui.label("Bộ câu hỏi").classes("group-label")
            for key, label in SOURCE_LABELS.items():
                active = " active" if s["source"] == key else ""
                ui.button(label, on_click=lambda k=key: restart(source=k, chapter=None)) \
                    .props(f"flat no-caps align=left color={'primary' if active else 'secondary'}") \
                    .classes(f"chapter{active}")
            if s["source"] == "nganhang_tn":
                ui.label("Chương").classes("group-label")
                chapters = [(None, "Tất cả chương")] + [
                    (n, f"{n}. {t}") for n, t in qbank.chapters("nganhang_tn")]
                for n, label in chapters:
                    active = " active" if s["chapter"] == n else ""
                    ui.button(label, on_click=lambda c=n: restart(chapter=c)) \
                        .props(f"flat no-caps align=left color={'primary' if active else 'secondary'}") \
                        .classes(f"chapter{active}")
            ui.switch("Xáo trộn thứ tự", value=s["shuffle"],
                      on_change=lambda e: restart(shuffle=e.value)).classes("mt-3 text-sm")

        summ = practice.summarize(qbank, s)
        with ui.column().classes("mt-6 gap-1"):
            if summ.done:
                ui.label(f"Đúng {summ.correct} / {summ.done} câu đã làm").classes("score pl-2")
                ui.button("Xem tổng kết", on_click=show_summary).props("flat no-caps dense align=left") \
                    .classes("chapter link")
            ui.button("Làm lại từ đầu", on_click=lambda: restart()).props("flat no-caps dense align=left") \
                .classes("chapter link")

    with ui.element("div").classes("mobile-bar"):
        ui.button(icon="menu", on_click=drawer.toggle).props('flat round aria-label="Mở menu"')
        ui.label("Pháp luật đại cương").classes("font-semibold")

    def show_summary() -> None:
        s["view"] = "summary"
        save()
        sheet.refresh()
        drawer.value = drawer.value  # keep drawer state on mobile

    # ---- actions -----------------------------------------------------------------------
    def pick(label: str) -> None:
        q = practice.current(qbank, s)
        if q.id in s["submitted"]:
            return
        p = view["picked"]
        if q.multi_answer:
            view["picked"] = [x for x in p if x != label] if label in p else p + [label]
        else:
            view["picked"] = [] if p == [label] else [label]
        sheet.refresh()

    def submit() -> None:
        q = practice.current(qbank, s)
        if not view["picked"] or q.id in s["submitted"]:
            return
        practice.submit(qbank, db, sid, s, view["picked"])
        view["picked"], view["just_submitted"] = [], q.id
        save()
        sheet.refresh()
        view["just_submitted"] = None  # the circle animates once, not on later redraws
        rail.refresh()

    async def ask_ai(container) -> None:
        q = practice.current(qbank, s)
        chosen = s["submitted"].get(q.id)
        if view["busy"] or not chosen:
            return
        view["busy"] = True
        container.clear()
        with container:
            with ui.element("div").classes("comment"):
                ui.html("<h3>Lời giải</h3>", sanitize=False)
                body = ui.markdown("_Đang tra giáo trình…_").classes("body")
        try:
            hits, result = await run.io_bound(explain, q, chosen, retriever(), llm())
            text, it = "", iter(result.chunks)
            while (chunk := await run.io_bound(next, it, None)) is not None:
                text += chunk
                body.set_content(text)
        except Exception as exc:  # noqa: BLE001 - shown as a plain-language notice with retry
            container.clear()
            with container, ui.element("div").classes("notice"):
                ui.label(error_message(exc))
                ui.button("Thử lại", on_click=lambda: ask_ai(container)) \
                    .props("flat no-caps").classes("btn btn-red mt-2")
        else:
            sources = [{"citation": h.chunk.citation, "body": h.chunk.body, "file": h.chunk.source_file,
                        "page_start": h.chunk.pdf_page_start, "page_end": h.chunk.pdf_page_end} for h in hits]
            s["explanations"][practice.explanation_key(q.id, chosen)] = {
                "text": text, "sources": sources, "provider": result.provider}
            save()
            container.clear()
            with container:
                comment(q.id, chosen)
        finally:
            view["busy"] = False

    def comment(qid: str, chosen: list[str]) -> None:
        key = practice.explanation_key(qid, chosen)
        exp = s["explanations"][key]
        sources = exp["sources"]
        view["sources"] = sources
        with ui.element("div").classes("comment"):
            ui.html("<h3>Lời giải</h3>", sanitize=False)
            ui.markdown(with_cites(exp["text"], len(sources))).classes("body")
            with ui.element("div").classes("sources"):
                ui.label("Nguồn trích dẫn")
                for i, src in enumerate(sources, start=1):
                    ui.html(f'<span class="num">{i}</span>{html.escape(src["citation"])}', sanitize=False) \
                        .props('role="button" tabindex="0"').classes("cursor-pointer") \
                        .on("click", lambda i=i, src=src: panel.show(i, src))
            rating = db.get_rating(sid, f"explain-{key}")
            with ui.row().classes("items-center gap-1 mt-3 text-sm text-[var(--graphite)]"):
                ui.label("Lời giải này có giúp bạn hiểu không?")
                for value, icon, aria in ((1, "thumb_up", "Có"), (0, "thumb_down", "Không")):
                    ui.button(icon=icon, on_click=lambda v=value: rate(key, v, exp)) \
                        .props(f'flat round dense size=sm aria-label="{aria}" '
                               f'color={"primary" if rating == value else "grey-7"}')
                ui.label(f"Trả lời bởi {exp['provider']}").classes("ml-3 text-xs")

    def rate(key: str, value: int, exp: dict) -> None:
        db.save_feedback(sid, f"explain-{key}", "explain", value, key, exp["text"], exp["provider"])
        ui.notify("Đã ghi nhận đánh giá", position="bottom", timeout=1500)
        sheet.refresh()

    async def on_cite(e) -> None:
        n = int(e.args)
        sources = view.get("sources") or []
        if 1 <= n <= len(sources):
            await panel.show(n, sources[n - 1])

    ui.on("cite", on_cite)

    # ---- the sheet ---------------------------------------------------------------------
    @ui.refreshable
    def sheet() -> None:
        with ui.element("main").classes("sheet"):
            if not s["order"]:
                empty_state()
            elif s["view"] == "summary":
                summary()
            else:
                question_view()

    def empty_state() -> None:
        if s["mode"] == practice.REVIEW:
            ui.html('<p class="question">Chưa có câu nào cần ôn.</p>', sanitize=False)
            ui.label("Câu bạn làm sai khi luyện tập sẽ được đưa vào đây và hỏi lại sau 1, 3, 7, 16 ngày.") \
                .classes("mb-4")
            ui.button("Luyện tập", on_click=lambda: restart(mode=practice.PRACTICE)) \
                .props("unelevated no-caps").classes("btn btn-violet")
        else:
            ui.html('<p class="question">Không có câu hỏi nào trong bộ này.</p>', sanitize=False)

    def question_view() -> None:
        q = practice.current(qbank, s)
        n, pos = len(s["order"]), s["pos"]
        chosen = s["submitted"].get(q.id)
        done = chosen is not None
        picked = chosen if done else view["picked"]
        where = f"Chương {q.chapter}" if q.chapter else BANKS[q.source][0]
        label = f"{s['label']}: câu" if s.get("label") else "Câu"
        ui.html(f'<div class="sheet-head"><span>{html.escape(label)} {pos + 1} / {n}</span>'
                f'<span>{html.escape(where)}</span></div>'
                f'<div class="progress"><div style="width:{100 * (pos + 1) / n:.1f}%"></div></div>'
                f'<h1 class="question">{html.escape(q.question)}</h1>', sanitize=False)

        with ui.element("div").classes("options").props('role="group" aria-label="Các phương án"'):
            for o in q.options:
                lab = o["label"]
                cls = ["option", "locked"] if done else ["option"]
                mark = circle = ""
                if lab in picked:
                    cls.append("picked")
                if done and lab in q.answers:
                    mark = TICK_SVG
                    cls.append("correct")
                    circle = CIRCLE_SVG.format(cls=" draw" if view["just_submitted"] == q.id else "")
                elif done and lab in chosen:
                    mark, _ = CROSS_SVG, cls.append("struck")
                btn = ui.element("button").classes(" ".join(cls)).props(
                    f'type="button" aria-pressed="{str(lab in picked).lower()}"'
                    + (' aria-disabled="true"' if done else ""))
                with btn:
                    ui.html(f'{mark}<span class="letter">{lab}{circle}</span>'
                            f'<span class="text">{html.escape(o["text"])}</span>', sanitize=False) \
                        .classes("contents")
                if not done:
                    btn.on("click", lambda lab=lab: pick(lab))

        if not done:
            if q.multi_answer:
                ui.label("Câu này có nhiều đáp án đúng, hãy chọn tất cả.").classes("text-sm mt-3")
            ui.button("Nộp bài", on_click=submit).props("unelevated no-caps") \
                .classes("btn btn-violet mt-5").set_enabled(bool(picked))
        else:
            if q.is_correct(chosen):
                verdict = "Đúng."
            else:
                verdict = f"Chưa đúng. Đáp án là {', '.join(q.answers)}."
                if s["mode"] == practice.PRACTICE:
                    verdict += " Câu này sẽ quay lại trong phần Ôn câu sai."
            ui.html(f'<p class="verdict">{verdict}</p>', sanitize=False)
            expl = ui.element("div")
            view["expl_container"] = expl
            with expl:
                if practice.explanation_key(q.id, chosen) in s["explanations"]:
                    comment(q.id, chosen)
                else:
                    ui.button("Hỏi AI vì sao", on_click=lambda: ask_ai(expl)) \
                        .props("flat no-caps").classes("btn btn-red mt-4")

        with ui.row().classes("w-full justify-between mt-8"):
            ui.button("Câu trước", icon="west", on_click=lambda: go(pos - 1)) \
                .props("flat no-caps").classes("btn btn-plain").set_enabled(pos > 0)
            if pos < n - 1:
                ui.button("Câu tiếp", on_click=lambda: go(pos + 1)) \
                    .props("flat no-caps icon-right=east").classes("btn btn-plain")
            else:
                ui.button("Xem tổng kết", on_click=show_summary).props("flat no-caps") \
                    .classes("btn btn-plain").set_enabled(bool(s["submitted"]))
        ui.html('<div class="hints"><span><kbd>A</kbd>–<kbd>D</kbd> chọn</span>'
                '<span><kbd>Enter</kbd> nộp bài</span><span><kbd>←</kbd> <kbd>→</kbd> chuyển câu</span>'
                '<span><kbd>H</kbd> hỏi AI</span></div>', sanitize=False)

    def summary() -> None:
        summ = practice.summarize(qbank, s)
        pct = round(100 * summ.correct / summ.done) if summ.done else 0
        ui.html(f'<div class="sheet-head"><span>Tổng kết{html.escape(": " + s["label"].lower()) if s.get("label") else ""}'
                f'</span><span>{summ.done} / {len(s["order"])} câu đã làm</span></div>'
                f'<div class="progress"><div style="width:{100 * summ.done / len(s["order"]):.1f}%"></div></div>'
                f'<p class="headline">Bạn làm đúng {summ.correct} trên {summ.done} câu ({pct}%).</p>',
                sanitize=False)
        if len(summ.groups) > 1:
            rows = "".join(
                f'<span class="name">{g.name}</span>'
                f'<div class="track" title="{g.name}: đúng {g.correct}/{g.done}">'
                f'<div class="fill" style="width:{100 * g.correct / g.done:.0f}%"></div></div>'
                f'<span class="val">{g.correct}/{g.done}</span>' for g in summ.groups)
            ui.html(f'<div class="bars" role="img" aria-label="Tỉ lệ đúng theo chương">{rows}</div>',
                    sanitize=False)
        ui.html(f'<h2 class="question" style="font-size:1.125rem;margin-top:1rem">'
                f'Câu làm sai ({len(summ.wrong)})</h2>', sanitize=False)
        if not summ.wrong:
            ui.label("Không có câu sai trong bộ này.")
        for qid in summ.wrong:
            q = qbank.by_id[qid]
            mine = ", ".join(s["submitted"][qid])
            with ui.element("div").classes("wrong-item w-full"):
                ui.html(f'<div class="q">{html.escape(q.question)}</div>'
                        f'<div class="a">Bạn chọn {mine}, đáp án đúng là {", ".join(q.answers)}: '
                        f'{html.escape(q.option_text(q.answers[0]))}</div>', sanitize=False)
                ui.button("Xem lại câu này", on_click=lambda qid=qid: go(s["order"].index(qid))) \
                    .props("flat no-caps dense").classes("btn btn-plain mt-1").style("padding-left:0")
        with ui.row().classes("mt-8 gap-3"):
            ui.button(f"Làm lại {len(summ.wrong)} câu sai", on_click=retry) \
                .props("unelevated no-caps").classes("btn btn-violet").set_enabled(bool(summ.wrong))
            ui.button("Quay lại làm bài", on_click=lambda: go(s["pos"])) \
                .props("flat no-caps").classes("btn btn-plain")

    def retry() -> None:
        practice.retry_wrong(s, practice.summarize(qbank, s).wrong)
        view["picked"] = []
        save()
        rail.refresh()
        sheet.refresh()

    with drawer:
        rail()
    sheet()

    # ---- keyboard shortcuts ----------------------------------------------------------
    async def on_key(e) -> None:
        if not e.action.keydown or e.modifiers.ctrl or e.modifiers.alt or e.modifiers.meta:
            return
        if not s["order"] or s["view"] != "question":
            return
        q = practice.current(qbank, s)
        key = e.key.name
        labels = [o["label"] for o in q.options]
        if key.upper() in labels:
            pick(key.upper())
        elif key in "1234" and int(key) <= len(labels):
            pick(labels[int(key) - 1])
        elif key == "Enter":
            submit()
        elif key == "ArrowRight" and s["pos"] < len(s["order"]) - 1:
            go(s["pos"] + 1)
        elif key == "ArrowLeft" and s["pos"] > 0:
            go(s["pos"] - 1)
        elif key.lower() == "h" and q.id in s["submitted"] and view.get("expl_container") is not None:
            if practice.explanation_key(q.id, s["submitted"][q.id]) not in s["explanations"]:
                await ask_ai(view["expl_container"])

    ui.keyboard(on_key=on_key)
