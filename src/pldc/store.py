"""Local SQLite persistence: chat history, quiz progress and 👍/👎 feedback, keyed by session id.

The session id lives in the page URL (?sid=...), so a reload restores the user's state.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .config import DATA_DIR
from .review import MASTERED, schedule

DB_PATH = DATA_DIR / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    sid TEXT NOT NULL, idx INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
    sources TEXT, provider TEXT, created REAL NOT NULL, PRIMARY KEY (sid, idx));
CREATE TABLE IF NOT EXISTS quiz_state (
    sid TEXT PRIMARY KEY, state TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS review (
    sid TEXT NOT NULL, qid TEXT NOT NULL, box INTEGER NOT NULL, due REAL NOT NULL,
    wrong_count INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL, PRIMARY KEY (sid, qid));
CREATE TABLE IF NOT EXISTS feedback (
    sid TEXT NOT NULL, ref TEXT NOT NULL, kind TEXT NOT NULL, rating INTEGER,
    prompt TEXT, response TEXT, provider TEXT, updated REAL NOT NULL, PRIMARY KEY (sid, ref));
"""


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        # A fresh connection per call: Streamlit serves each session from its own thread.
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _run(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with closing(self._conn()) as conn, conn:
            return conn.execute(sql, params).fetchall()

    # ---- chat --------------------------------------------------------------------
    def load_chat(self, sid: str) -> list[dict]:
        rows = self._run("SELECT * FROM chat_messages WHERE sid=? ORDER BY idx", (sid,))
        return [{"role": r["role"], "content": r["content"],
                 "sources": json.loads(r["sources"]) if r["sources"] else [],
                 "provider": r["provider"]} for r in rows]

    def add_chat_message(self, sid: str, idx: int, msg: dict) -> None:
        self._run("INSERT OR REPLACE INTO chat_messages VALUES (?,?,?,?,?,?,?)",
                  (sid, idx, msg["role"], msg["content"],
                   json.dumps(msg.get("sources") or [], ensure_ascii=False),
                   msg.get("provider"), time.time()))

    def clear_chat(self, sid: str) -> None:
        self._run("DELETE FROM chat_messages WHERE sid=?", (sid,))

    # ---- quiz --------------------------------------------------------------------
    def load_quiz(self, sid: str) -> dict | None:
        rows = self._run("SELECT state FROM quiz_state WHERE sid=?", (sid,))
        return json.loads(rows[0]["state"]) if rows else None

    def save_quiz(self, sid: str, state: dict) -> None:
        self._run("INSERT OR REPLACE INTO quiz_state VALUES (?,?,?)",
                  (sid, json.dumps(state, ensure_ascii=False), time.time()))

    # ---- spaced repetition ------------------------------------------------------
    def record_answer(self, sid: str, qid: str, correct: bool, now: float | None = None) -> None:
        now = time.time() if now is None else now
        rows = self._run("SELECT box, wrong_count FROM review WHERE sid=? AND qid=?", (sid, qid))
        box, wrong = (rows[0]["box"], rows[0]["wrong_count"]) if rows else (None, 0)
        if box is not None and box >= MASTERED:
            box = None  # mastered earlier: treat like a fresh question
        nxt = schedule(box, correct, now)
        if nxt is None:
            return
        new_box, due = nxt
        self._run("INSERT OR REPLACE INTO review VALUES (?,?,?,?,?,?)",
                  (sid, qid, new_box, due if due != float("inf") else 1e18,
                   wrong + (not correct), now))

    def due_reviews(self, sid: str, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        rows = self._run("SELECT qid FROM review WHERE sid=? AND box<? AND due<=? ORDER BY due",
                         (sid, MASTERED, now))
        return [r["qid"] for r in rows]

    def review_stats(self, sid: str, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        rows = self._run(
            "SELECT SUM(box<? AND due<=?) AS due, SUM(box<?) AS active, SUM(box>=?) AS mastered "
            "FROM review WHERE sid=?", (MASTERED, now, MASTERED, MASTERED, sid))
        r = rows[0]
        return {"due": r["due"] or 0, "active": r["active"] or 0, "mastered": r["mastered"] or 0}

    # ---- feedback ----------------------------------------------------------------
    def get_rating(self, sid: str, ref: str) -> int | None:
        rows = self._run("SELECT rating FROM feedback WHERE sid=? AND ref=?", (sid, ref))
        return rows[0]["rating"] if rows else None

    def save_feedback(self, sid: str, ref: str, kind: str, rating: int | None,
                      prompt: str, response: str, provider: str | None) -> None:
        self._run("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?,?,?)",
                  (sid, ref, kind, rating, prompt, response, provider, time.time()))
