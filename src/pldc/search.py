"""Hybrid retrieval: BM25 (keywords) + dense embeddings, fused with reciprocal rank fusion.

If the embedding API is unavailable at query time, we degrade to BM25 only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, load_all_chunks
from .config import INDEX_DIR

log = logging.getLogger(__name__)

WORD_RE = re.compile(r"\w+")
RRF_K = 60
CANDIDATES = 50
DENSE_COOLDOWN_S = 300  # after an embedding failure, use BM25 only for this long


def tokenize(text: str) -> list[str]:
    """Vietnamese words are multi-syllable, so index syllables plus syllable bigrams."""
    syl = WORD_RE.findall(unicodedata.normalize("NFC", text).lower())
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.lens.mean()) if docs else 0.0
        df = Counter(t for d in self.tfs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / (self.avg or 1))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    similarity: float | None = None  # cosine to the query (None when only BM25 was used)


def corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in chunks),
                    encoding="utf-8")


def _old_vectors(index_dir: Path, model_id: str) -> dict[tuple[str, str], np.ndarray]:
    """Vectors from the previous index, keyed by (chunk id, text), if built with the same model."""
    try:
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("embed_model") != model_id:
            return {}
        old = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        vecs = np.load(index_dir / "embeddings.npy")
    except FileNotFoundError:
        return {}
    return {(c["id"], c["text"]): v for c, v in zip(old, vecs)}


RESUME_BATCH = 50


def _embed_resumable(embedder, chunks: list[Chunk], index_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    """Embed in batches, saving each batch to a side file, so an interrupted build (daily quota
    used up, process killed) keeps what it paid for; the next build continues from there."""
    path = index_dir / "partial_vectors.npz"
    key = lambda c: hashlib.sha256(f"{embedder.model_id}|{c.id}|{c.text}".encode()).hexdigest()[:24]  # noqa: E731
    done: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as data:
            done = {k: data[k] for k in data.files}
    todo = [c for c in chunks if key(c) not in done]
    if done and chunks:
        log.info("resuming: %d of %d already embedded", len(chunks) - len(todo), len(chunks))
    for i in range(0, len(todo), RESUME_BATCH):
        batch = todo[i:i + RESUME_BATCH]
        done.update(zip(map(key, batch), embedder.embed([c.text for c in batch])))
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, **done)
    return {(c.id, c.text): done[key(c)] for c in chunks}


def build_index(embedder, index_dir: Path = INDEX_DIR, force: bool = False) -> str:
    """Embed chunks and save the index. Incremental: unchanged chunks reuse their old vector
    (the free embedding tier allows only ~1000 texts per day).

    Returns "unchanged" or "built (N embedded)".
    """
    chunks = load_all_chunks()
    chunks_path = index_dir / "chunks.jsonl"
    reuse = {} if force else _old_vectors(index_dir, embedder.model_id)
    if reuse and len(reuse) == len(chunks) and chunks_path.exists():
        old = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines()]
        if old == [c.to_dict() for c in chunks]:
            return "unchanged"
    missing = [c for c in chunks if (c.id, c.text) not in reuse]
    fresh = _embed_resumable(embedder, missing, index_dir)
    vecs = np.stack([reuse.get((c.id, c.text), fresh.get((c.id, c.text))) for c in chunks])
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", vecs.astype(np.float32))
    _write_chunks(chunks_path, chunks)
    digest = corpus_hash(chunks)
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "corpus_hash": digest, "embed_model": embedder.model_id, "count": len(chunks),
    }, indent=2), encoding="utf-8")
    (index_dir / "partial_vectors.npz").unlink(missing_ok=True)
    return f"built ({len(missing)} embedded)"


class Retriever:
    def __init__(self, embedder=None, index_dir: Path = INDEX_DIR, reranker=None,
                 rerank_candidates: int = 20):
        lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        self.chunks = [Chunk(**json.loads(l)) for l in lines]
        self.vecs = np.load(index_dir / "embeddings.npy")
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if embedder is not None and manifest["embed_model"] != embedder.model_id:
            raise RuntimeError(f"Index built with {manifest['embed_model']}, but embedder is "
                               f"{embedder.model_id}. Run: python src/build_index.py --force")
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        self.last_mode = "hybrid"
        self._dense_off_until = 0.0

    def search(self, query: str, k: int = 6) -> list[Hit]:
        hits, self.last_mode = self.search_with_mode(query, k)
        return hits

    def search_with_mode(self, query: str, k: int = 6, min_similarity: float | None = None,
                         margin: float | None = None) -> tuple[list[Hit], str]:
        """Like search(), also returning the mode actually used ("hybrid" or "bm25", plus
        "+rerank" when the reranker re-ordered the hits); safe to call from several threads
        (doesn't rely on last_mode).

        With a reranker, the best `rerank_candidates` fused hits are re-scored and the top k kept
        (Hit.score is then the reranker's logit).

        Optional gating (hybrid only): drop hits whose cosine to the query is below
        `min_similarity`, or more than `margin` below the best hit. May return fewer than k."""
        rankings = [np.argsort(-self.bm25.scores(tokenize(query)))[:CANDIDATES]]
        mode = "bm25"
        sims = None
        if self.embedder is not None and time.monotonic() >= self._dense_off_until:
            try:
                q = self.embedder.embed([query], is_query=True)[0]
                sims = self.vecs @ q
                rankings.append(np.argsort(-sims)[:CANDIDATES])
                mode = "hybrid"
            except Exception as exc:  # noqa: BLE001 - keep chat working on keywords alone
                # e.g. daily free quota used up: don't make every question wait for a retry.
                self._dense_off_until = time.monotonic() + DENSE_COOLDOWN_S
                log.warning("embedding failed, BM25 only for %ss: %s", DENSE_COOLDOWN_S, exc)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])
        if self.reranker is not None:
            pool = [i for i, _ in best[:max(k, self.rerank_candidates)]]
            try:
                scores = self.reranker.scores(query, [self.chunks[i].text for i in pool])
                best = sorted(zip(pool, map(float, scores)), key=lambda kv: -kv[1])
                mode += "+rerank"
            except Exception as exc:  # noqa: BLE001 - keep the first-stage order
                log.warning("rerank failed, using fused order: %s", exc)
        hits = [Hit(self.chunks[i], s, float(sims[i]) if sims is not None else None)
                for i, s in best[:k]]
        if sims is not None and hits and (min_similarity is not None or margin is not None):
            top = max(h.similarity for h in hits)
            hits = [h for h in hits
                    if (min_similarity is None or h.similarity >= min_similarity)
                    and (margin is None or h.similarity >= top - margin)]
        return hits, mode


def build_retriever() -> Retriever:
    """The app's retriever: hybrid with the configured embedder (EMBEDDER=gemini|local), then the
    reranker when one is configured (RERANKER, see pldc.rerank.build_reranker).

    Falls back to the Gemini index if the local model/index is missing, and to keywords only if
    there is no Gemini key.
    """
    from .config import get_settings, index_dir_for  # local imports: keep module import light
    from .embeddings import GeminiEmbedder, LocalEmbedder
    from .rerank import build_reranker

    s = get_settings()
    rerank = {"reranker": build_reranker(s), "rerank_candidates": s.rerank_candidates}
    if s.embedder == "local":
        try:
            local = LocalEmbedder()
            if (index_dir_for(local.model_id) / "manifest.json").exists():
                return Retriever(local, index_dir=index_dir_for(local.model_id), **rerank)
            log.warning("bge-m3 index missing; run: python src/build_index.py --embedder local")
        except RuntimeError as exc:
            log.warning("local embedder unavailable (%s); using Gemini", exc)
    if not (INDEX_DIR / "manifest.json").exists():
        raise FileNotFoundError("Chưa có chỉ mục tìm kiếm. Chạy: python src/build_index.py")
    try:
        embedder = GeminiEmbedder()
    except RuntimeError:
        embedder = None  # no Gemini key: keyword search only
    return Retriever(embedder, **rerank)
