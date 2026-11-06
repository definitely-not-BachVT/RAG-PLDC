"""Cross-encoder reranking of retrieved passages: BAAI/bge-reranker-v2-m3 via onnxruntime.

The first stage (BM25 + vectors) scores query and passage separately; the reranker reads them
together, so it can tell "who may NOT marry" from "who may marry". It is a 568M-parameter model:
~40 ms per passage on a laptop GPU (DirectML), but ~1.2 s on the CPU, so by default it is only
switched on when a GPU is available (RERANKER=auto, see build_reranker).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from .config import ROOT, Settings, get_settings
from .embeddings import GPU_LOCK, onnx_session

log = logging.getLogger(__name__)

MODEL_DIR = ROOT / "models" / "bge-reranker-v2-m3"


class Reranker:
    MAX_TOKENS = 512  # query + passage; chunks are ~1000 chars (~330 tokens)
    BATCH = 16

    def __init__(self, model_dir: Path = MODEL_DIR, prefer_gpu: bool = True):
        from tokenizers import Tokenizer  # imported lazily: only needed when reranking

        # fp16 is fastest on the GPU; int8 is the CPU build (fp16 has no fast CPU kernels).
        names = ["model_fp16.onnx", "model_int8.onnx"] if prefer_gpu else ["model_int8.onnx"]
        path = next((model_dir / "onnx" / n for n in names if (model_dir / "onnx" / n).exists()), None)
        if path is None:
            raise RuntimeError(f"Chưa có model reranker tại {model_dir}. Xem README để tải.")
        self.session, self.on_gpu = onnx_session(path)
        if self.on_gpu and path.name != "model_fp16.onnx":
            log.warning("reranker: %s on the GPU; model_fp16.onnx is faster", path.name)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self.input_names = {i.name for i in self.session.get_inputs()}
        # one batch at a time (eval calls from many threads); on the GPU, one model at a time
        self._lock = GPU_LOCK if self.on_gpu else threading.Lock()

    @property
    def model_id(self) -> str:
        return "local:bge-reranker-v2-m3"

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logits, one per passage (higher = more relevant; not a probability)."""
        out = []
        with self._lock:
            for i in range(0, len(passages), self.BATCH):
                enc = self.tokenizer.encode_batch([(query, p) for p in passages[i:i + self.BATCH]])
                feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
                logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]
                out.append(logits.reshape(len(enc), -1)[:, 0].astype(np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def build_reranker(settings: Settings | None = None) -> Reranker | None:
    """The configured reranker, or None.

    RERANKER=auto (default): use it only on a GPU, where it costs ~1 s per question.
    RERANKER=on: always (on the CPU expect ~25 s per question). RERANKER=off: never.
    """
    mode = (settings or get_settings()).reranker
    if mode == "off":
        return None
    try:
        reranker = Reranker()
    except Exception as exc:  # noqa: BLE001 - missing model / runtime: search still works without it
        log.warning("reranker unavailable: %s", exc)
        return None
    if mode == "auto" and not reranker.on_gpu:
        log.info("reranker off: no GPU (set RERANKER=on to use it on the CPU anyway)")
        return None
    return reranker
