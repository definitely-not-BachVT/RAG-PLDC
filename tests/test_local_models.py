"""The local ONNX models (bge-m3, bge-reranker-v2-m3) used from several threads at once, as the
web app and the evaluation scripts do. On DirectML, two models running at the same time used to
fail with "Catastrophic failure" or crash the process. Run with `pytest -m slow`."""
import threading

import pytest

from pldc.embeddings import LocalEmbedder
from pldc.rerank import Reranker

pytestmark = pytest.mark.slow


def test_embedder_and_reranker_can_run_concurrently():
    try:
        embedder, reranker = LocalEmbedder(), Reranker()
    except RuntimeError as exc:
        pytest.skip(str(exc))  # models not downloaded
    errors = []

    def embed():
        for _ in range(10):
            embedder.embed(["quyền thừa kế theo di chúc"] * 4, is_query=True)

    def rerank():
        for _ in range(10):
            reranker.scores("thừa kế", ["di chúc hợp pháp cần điều kiện gì"] * 8)

    def guarded(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(fn,)) for fn in (embed, rerank, embed, rerank)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
