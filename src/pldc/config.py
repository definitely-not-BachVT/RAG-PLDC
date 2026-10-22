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


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    return Settings(
        google_api_key=env("GOOGLE_API_KEY", "").strip(),
        gemini_models=[m.strip() for m in env(
            "GEMINI_MODELS", "gemini-flash-latest,gemini-3.8-flash,gemini-flash-lite-latest").split(",")
            if m.strip()],
        gemini_embed_model=env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        embed_dim=int(env("EMBED_DIM", "768")),
        embedder=env("EMBEDDER", "gemini").strip().lower(),
        chat_top_k=int(env("CHAT_TOP_K", "6")),
        llm_timeout=float(env("LLM_TIMEOUT", "30")),
    )
