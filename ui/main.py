"""NiceGUI front end.  Run:  python ui/main.py   (then open http://localhost:8080)"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from nicegui import ui

sys.path.insert(0, str(Path(__file__).parent))  # sibling modules (theme, services, ...)

import quiz_page  # noqa: E402
from pldc.config import DATA_DIR  # noqa: E402
from theme import apply_theme  # noqa: E402


def storage_secret() -> str:
    """Signs the per-browser cookie; generated once and kept out of git."""
    if env := os.environ.get("UI_STORAGE_SECRET"):
        return env
    path = DATA_DIR / ".ui_secret"
    if not path.exists():
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


@ui.page("/")
def practice_page() -> None:
    ui.page_title("Luyện trắc nghiệm – Pháp luật đại cương")
    apply_theme()
    quiz_page.build()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Pháp luật đại cương", port=int(os.environ.get("UI_PORT", 8080)),
           storage_secret=storage_secret(), language="vi", favicon="⚖️",
           reload=os.environ.get("UI_RELOAD") == "1", show=False)
