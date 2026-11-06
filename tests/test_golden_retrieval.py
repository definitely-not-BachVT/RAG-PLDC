"""Retrieval quality on the golden questions: is the textbook section that holds the answer found?

Offline (BM25 only, no API). Thresholds are set just under the measured values, so a change that
makes retrieval worse fails here. The bge-m3 hybrid variant runs with `pytest -m slow`.
"""
import time

import pytest

from golden import GROUNDED, retrieval_query
from pldc.answer_checks import section_matches
from pldc.config import index_dir_for
from pldc.search import Retriever

K = 6
MIN_RECALL = 1.0  # measured 9/9
MIN_MRR = 0.75  # measured ~0.84
MAX_AVG_MS = 200  # measured ~36 ms


def first_rank(hits, expected) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.chunk.section_id and section_matches({h.chunk.section_id}, expected):
            return rank
    return None


def evaluate(retriever: Retriever):
    ranks, slow = {}, []
    for case in GROUNDED:
        t0 = time.perf_counter()
        hits = retriever.search(retrieval_query(case), k=K)
        slow.append(time.perf_counter() - t0)
        ranks[case["id"]] = first_rank(hits, case["expected_sections"])
    return ranks, 1000 * sum(slow) / len(slow)


def check(ranks: dict, label: str) -> None:
    missed = [cid for cid, r in ranks.items() if r is None]
    recall = 1 - len(missed) / len(ranks)
    mrr = sum(1 / r for r in ranks.values() if r) / len(ranks)
    assert recall >= MIN_RECALL, f"{label}: expected section not in top {K} for {missed}"
    assert mrr >= MIN_MRR, f"{label}: MRR {mrr:.2f} < {MIN_MRR}; ranks {ranks}"


@pytest.fixture(scope="module")
def bm25():
    return Retriever(None)


def test_bm25_finds_the_answer_section(bm25):
    ranks, _ = evaluate(bm25)
    check(ranks, "bm25")


def test_bm25_is_fast(bm25):
    _, avg_ms = evaluate(bm25)
    assert avg_ms <= MAX_AVG_MS, f"BM25 search took {avg_ms:.0f} ms on average"


@pytest.mark.slow
def test_local_hybrid_finds_the_answer_section():
    pytest.importorskip("onnxruntime")
    from pldc.embeddings import LocalEmbedder
    try:
        emb = LocalEmbedder()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    index = index_dir_for(emb.model_id)
    if not (index / "manifest.json").exists():
        pytest.skip("bge-m3 index not built (python src/build_index.py --embedder local)")
    ranks, avg_ms = evaluate(Retriever(emb, index_dir=index))
    check(ranks, "bm25+bge-m3")
    assert avg_ms <= 3000, f"hybrid search with local embedding took {avg_ms:.0f} ms on average"
