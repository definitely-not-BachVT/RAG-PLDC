"""Citations: [n] markers in AI text, and the side panel showing the original PDF page."""
from __future__ import annotations

import base64
import html
import re

from nicegui import run, ui

from pldc import pages

CITE_RE = re.compile(r"\[(\d{1,2})\]")
# Models sometimes write arrows as LaTeX despite the prompt; markdown would show them raw.
LATEX_ARROWS = {r"$\rightarrow$": "→", r"$\Rightarrow$": "⇒", r"$\leftarrow$": "←", r"$\to$": "→"}


def with_cites(text: str, n_sources: int) -> str:
    """Turn [n] into clickable superscripts (only for n that exist); tidy stray LaTeX arrows."""
    for latex, char in LATEX_ARROWS.items():
        text = text.replace(latex, char)

    def sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f'<sup class="cite" data-n="{n}">{n}</sup>' if 1 <= n <= n_sources else m.group(0)
    return CITE_RE.sub(sub, text)


class SourcePanel:
    """Right-hand drawer: excerpt + rendered PDF page with page stepping."""

    def __init__(self) -> None:
        self.src: dict | None = None
        self.num = 0
        self.page = 1
        self.total = 1
        with ui.right_drawer(value=False, fixed=True).classes("source-panel").props(
                "width=520 overlay bordered") as self.drawer:
            self.body = ui.column().classes("w-full gap-3 p-4")

    async def show(self, num: int, src: dict) -> None:
        self.src, self.num = src, num
        if src.get("file"):
            self.page = await run.io_bound(pages.locate_page, src["file"], src["page_start"],
                                           src["page_end"], src["body"])
            self.total = await run.io_bound(pages.page_count, src["file"])
        await self._render()
        self.drawer.show()

    async def step(self, delta: int) -> None:
        self.page = min(max(self.page + delta, 1), self.total)
        await self._render()

    async def _render(self) -> None:
        src = self.src
        self.body.clear()
        with self.body:
            with ui.row().classes("w-full items-start justify-between no-wrap"):
                ui.html(f'<div style="font-family:var(--serif);font-weight:600">'
                        f'<span class="cite" style="font-size:1em">[{self.num}]</span> '
                        f'{html.escape(src["citation"])}</div>', sanitize=False)
                ui.button(icon="close", on_click=self.drawer.hide).props(
                    'flat round dense aria-label="Đóng"').classes("text-[var(--graphite)]")
            ui.html(f'<div class="excerpt">{html.escape(src["body"])}</div>', sanitize=False)
            if not src.get("file"):
                ui.label("Nguồn này không kèm trang PDF gốc.").classes("text-sm")
                return
            png = await run.io_bound(pages.render_page, src["file"], self.page)
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"Trang {self.page} / {self.total} trong file PDF").classes("text-sm")
                with ui.row().classes("gap-1"):
                    ui.button(icon="chevron_left", on_click=lambda: self.step(-1)).props(
                        'flat round dense aria-label="Trang trước"').set_enabled(self.page > 1)
                    ui.button(icon="chevron_right", on_click=lambda: self.step(1)).props(
                        'flat round dense aria-label="Trang sau"').set_enabled(self.page < self.total)
            ui.image("data:image/png;base64," + base64.b64encode(png).decode()).classes("w-full")
