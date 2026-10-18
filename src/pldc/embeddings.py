"""Text embeddings. DeepSeek has no embedding API, so this always uses Gemini.

Swap in another class with the same `embed()` signature (e.g. local bge-m3) if needed.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import Settings, get_settings

log = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_ATTEMPTS = 6  # indexing: wait out free-tier rate limits
QUERY_ATTEMPTS = 2  # queries: fail fast, the retriever falls back to BM25


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        if not s.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY chưa được cấu hình trong .env")
        self.client = genai.Client(
            api_key=s.google_api_key,
            http_options=genai_types.HttpOptions(timeout=int(s.llm_timeout * 1000)))
        self.model = s.gemini_embed_model
        self.dim = s.embed_dim

    @property
    def model_id(self) -> str:
        return f"gemini:{self.model}:{self.dim}"

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """L2-normalized float32 matrix of shape (len(texts), dim)."""
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            out.extend(self._embed_batch(texts[i:i + BATCH_SIZE], task,
                                         QUERY_ATTEMPTS if is_query else MAX_ATTEMPTS))
            if not is_query:
                log.info("embedded %d/%d", min(i + BATCH_SIZE, len(texts)), len(texts))
        vecs = np.asarray(out, dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)

    def _embed_batch(self, batch: list[str], task: str, attempts: int) -> list[list[float]]:
        config = genai_types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        for attempt in range(attempts):
            try:
                resp = self.client.models.embed_content(model=self.model, contents=batch, config=config)
                return [e.values for e in resp.embeddings]
            except genai_errors.APIError as exc:
                # Free tier hits 429 easily when indexing; back off and retry.
                if exc.code not in (429, 500, 503) or attempt == attempts - 1:
                    raise
                wait = 2 ** attempt * 5
                log.warning("embed %s, retrying in %ss", exc.code, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")
