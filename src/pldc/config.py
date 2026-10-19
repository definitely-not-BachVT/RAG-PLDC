"""Settings read from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
OVERRIDES_FILE = DATA_DIR / "answer_overrides.json"


@dataclass
class Settings:
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    google_api_key: str
    gemini_models: list[str]  # tried in order
    gemini_embed_model: str
    embed_dim: int
    llm_timeout: float


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    return Settings(
        deepseek_api_key=env("DEEPSEEK_API_KEY", "").strip(),
        deepseek_model=env("DEEPSEEK_MODEL", "deepseek-flash"),
        deepseek_base_url=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        google_api_key=env("GOOGLE_API_KEY", "").strip(),
        gemini_models=[m.strip() for m in env(
            "GEMINI_MODELS", "gemini-flash-latest,gemini-3.8-flash,gemini-flash-lite-latest").split(",")
            if m.strip()],
        gemini_embed_model=env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        embed_dim=int(env("EMBED_DIM", "768")),
        llm_timeout=float(env("LLM_TIMEOUT", "30")),
    )
