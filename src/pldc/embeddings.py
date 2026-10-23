"""Text embeddings.

Two interchangeable embedders with the same `embed()` / `model_id` interface:
  * GeminiEmbedder  - API; free tier allows ~100 texts/minute and ~1000/day.
  * LocalEmbedder   - BAAI/bge-m3 via onnxruntime (GPU through DirectML, else CPU); no quota, no network.
Pick one with EMBEDDER=gemini|local (see build_embedder); each has its own index directory.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import ROOT, Settings, get_settings

log = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_ATTEMPTS = 6  # indexing: wait out free-tier rate limits
QUERY_ATTEMPTS = 2  # queries: fail fast, the retriever falls back to BM25
QUERY_TIMEOUT_S = 8  # a hung query embedding delays the whole answer (golden test caught a 32 s wait)


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
        if task == "RETRIEVAL_QUERY":
            config.http_options = genai_types.HttpOptions(timeout=QUERY_TIMEOUT_S * 1000)
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


def onnx_session(path: Path, threads: int | None = None):
    """An onnxruntime session on the GPU through DirectML when available (onnxruntime-directml
    on Windows), else on the CPU. Returns (session, on_gpu)."""
    import onnxruntime as ort  # imported lazily: only needed for the local models

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    if "DmlExecutionProvider" in ort.get_available_providers():
        opts.enable_mem_pattern = False  # both required by the DirectML provider
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = ort.InferenceSession(str(path), opts,
                                           providers=["DmlExecutionProvider", "CPUExecutionProvider"])
            return session, True
        except Exception as exc:  # noqa: BLE001 - e.g. no usable GPU: the CPU still works
            log.warning("DirectML unavailable (%s); using the CPU", exc)
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"]), False


class LocalEmbedder:
    """BAAI/bge-m3 dense embeddings (1024-d) with onnxruntime (GPU if available, else CPU).

    Uses the ONNX export from the model repo (no pickled weights, no torch). Dense vectors are
    the normalized [CLS] state, as in the reference implementation; queries need no prefix.
    """

    MAX_TOKENS = 512  # chunks are ~1000 chars (~300 tokens); longer text is truncated
    BATCH = 8

    def __init__(self, model_dir: Path | None = None, threads: int | None = None):
        from tokenizers import Tokenizer  # imported lazily: only needed when this embedder is used

        model_dir = model_dir or ROOT / "models" / "bge-m3" / "onnx"
        if not (model_dir / "model.onnx").exists():
            raise RuntimeError(f"Chưa có model bge-m3 tại {model_dir}. Xem README để tải.")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self.session, self.on_gpu = onnx_session(model_dir / "model.onnx", threads)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.dim = 1024
        self._lock = threading.Lock()  # one batch at a time (eval calls from many threads)

    @property
    def model_id(self) -> str:
        return "local:bge-m3:1024"

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        with self._lock:
            return self._embed(texts, is_query)

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.BATCH):
            enc = self.tokenizer.encode_batch(texts[i:i + self.BATCH])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            feed = {k: v for k, v in feed.items() if k in self.input_names}
            result = dict(zip(self.output_names, self.session.run(None, feed)))
            if "sentence_embedding" in result:  # the repo's ONNX export pools for us
                vecs = result["sentence_embedding"]
            else:  # plain encoder export: bge-m3 dense = [CLS] token
                vecs = result["token_embeddings"][:, 0]
            out.append(vecs.astype(np.float32))
            if not is_query and len(texts) > self.BATCH:
                log.info("embedded %d/%d", min(i + self.BATCH, len(texts)), len(texts))
        vecs = np.concatenate(out)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


def build_embedder(settings: Settings | None = None):
    """The configured embedder (EMBEDDER=gemini|local)."""
    s = settings or get_settings()
    return LocalEmbedder() if s.embedder == "local" else GeminiEmbedder(s)
