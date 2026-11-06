"""Settings read from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"  # Gemini-embedding index (tracked in git)


def index_dir_for(model_id: str) -> Path:
    """Each embedder keeps its own index, so switching EMBEDDER needs no rebuild of the other."""
    return INDEX_DIR if model_id.startswith("gemini:") else DATA_DIR / f"index-{model_id.split(':')[1]}"
OVERRIDES_FILE = DATA_DIR / "answer_overrides.json"


@dataclass
class Settings:
    google_api_key: str
    gemini_models: list[str]  # tried in order
    gemini_embed_model: str
    embed_dim: int
    embedder: str  # "gemini" or "local" (bge-m3 on CPU)
    chat_top_k: int  # passages given to the chat model
    llm_timeout: float
    gemini_thinking_budget: int | None  # None: model default; 0: no thinking (flash models)
    reranker: str  # "auto" (only on a GPU), "on" or "off"
    rerank_candidates: int  # first-stage hits the reranker re-scores


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    return Settings(
        google_api_key=env("GOOGLE_API_KEY", "").strip(),
        gemini_models=[m.strip() for m in env(
            "GEMINI_MODELS", "gemini-flash-lite-latest,gemini-3.8-flash,gemini-flash-latest").split(",")
            if m.strip()],
        gemini_embed_model=env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        embed_dim=int(env("EMBED_DIM", "768")),
        embedder=env("EMBEDDER", "gemini").strip().lower(),
        chat_top_k=int(env("CHAT_TOP_K", "6")),
        llm_timeout=float(env("LLM_TIMEOUT", "30")),
        gemini_thinking_budget=int(b) if (b := env("GEMINI_THINKING_BUDGET", "0").strip()) else None,
        reranker=env("RERANKER", "auto").strip().lower(),
        rerank_candidates=int(env("RERANK_CANDIDATES", "20")),
    )
