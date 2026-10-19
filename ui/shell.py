"""Page frame shared by every screen: left rail (wordmark + navigation) and the phone menu bar."""
from __future__ import annotations

from typing import Callable

from nicegui import ui

# key, label, url. Practice and review live on the same page, switched by ?mode=.
NAV = [("practice", "Luyện tập", "/?mode=practice"),
       ("review", "Ôn câu sai", "/?mode=review"),
       ("chat", "Hỏi đáp", "/hoi-dap")]


def frame() -> ui.left_drawer:
    """Create the rail drawer and the phone menu bar; fill the drawer later with `with drawer:`."""
    # value=None lets Quasar decide: open on desktop, closed (menu button) on phones.
    drawer = ui.left_drawer(value=None, fixed=True).classes("rail").props("width=280 breakpoint=900")
    with ui.element("div").classes("mobile-bar"):
        ui.button(icon="menu", on_click=drawer.toggle).props('flat round aria-label="Mở menu"')
        ui.label("Pháp luật đại cương").classes("font-semibold")
    return drawer


def nav(current: str, due: int, on_mode: Callable[[str], None] | None = None) -> None:
    """Wordmark and the three destinations. On the practice page, switching between
    practice and review happens in place via `on_mode`; elsewhere these are plain links."""
    ui.html('<div class="wordmark">Pháp luật<br>đại cương</div>'
            '<div class="tagline">Ôn thi theo giáo trình PTIT</div>', sanitize=False).classes("mb-5")
    for key, label, url in NAV:
        count = f"{due} câu" if key == "review" and due else ""
        inner = f'<span>{label}</span><span class="count">{count}</span>'
        cls = "nav-link active" if key == current else "nav-link"
        if on_mode and key in ("practice", "review"):
            ui.html(inner, sanitize=False).classes(cls).props('role="button" tabindex="0"') \
                .on("click", lambda k=key: on_mode(k)).on("keydown.enter", lambda k=key: on_mode(k))
        else:
            with ui.element("a").classes(cls).props(f'href="{url}"'):
                ui.html(inner, sanitize=False).classes("contents")
