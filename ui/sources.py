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


def cite_refs(text: str, n_sources: int) -> list[int]:
    """Source number behind each citation marker, in order of appearance (footnote k -> refs[k-1])."""
    return [n for n in (int(m) for m in CITE_RE.findall(text)) if 1 <= n <= n_sources]


def with_cites(text: str, n_sources: int) -> str:
    """Turn [n] into clickable footnotes numbered 1, 2, 3... by appearance (not by source number,
    so one source cited five times is not five identical "1"s); tidy stray LaTeX arrows."""
    for latex, char in LATEX_ARROWS.items():
        text = text.replace(latex, char)
    seen = 0

    def sub(m: re.Match) -> str:
        nonlocal seen
        if not 1 <= int(m.group(1)) <= n_sources:
            return m.group(0)
        seen += 1
        return f'<sup class="cite" data-n="{seen}">{seen}</sup>'
    return CITE_RE.sub(sub, text)


def answer_markdown(text: str, n_sources: int) -> ui.markdown:
    """Markdown with footnotes. 'cuddled-lists': models write "**Heading:**
* item" with no blank
    line, which plain markdown2 leaves as a paragraph with literal asterisks."""
    return ui.markdown(with_cites(text, n_sources), extras=[*ui.markdown.default_extras, "cuddled-lists"])


def source_rows(text: str, sources: list[dict]) -> list[tuple[list[int], int, dict]]:
    """Cited sources, one row each: (footnote numbers, source number, source) in order of first citation.
    When the text cites nothing (older answers), all sources are listed as before."""
    refs = cite_refs(text, len(sources))
    if not refs:
        return [([i], i, src) for i, src in enumerate(sources, start=1)]
    rows: dict[int, list[int]] = {}
    for k, n in enumerate(refs, start=1):
        rows.setdefault(n, []).append(k)
    return [(ks, n, sources[n - 1]) for n, ks in rows.items()]


def source_list(text: str, sources: list[dict], panel: "SourcePanel") -> None:
    """The "Nguồn trích dẫn" list under an answer; each row shows its footnote numbers."""
    with ui.element("div").classes("sources"):
        ui.label("Nguồn trích dẫn")
        for ks, n, src in source_rows(text, sources):
            label = ", ".join(map(str, ks))
            ui.html(f'<span class="num">{label}</span>{html.escape(src["citation"])}', sanitize=False)                 .props('role="button" tabindex="0"').classes("cursor-pointer")                 .on("click", lambda n=n, src=src: panel.show(n, src))


def source_for_footnote(text: str, sources: list[dict], k: int) -> tuple[int, dict] | None:
    """(source number, source) behind footnote k, or None."""
    refs = cite_refs(text, len(sources))
    return (refs[k - 1], sources[refs[k - 1] - 1]) if 1 <= k <= len(refs) else None


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
