import numpy as np

from pldc.chunking import _split, load_all_chunks
from pldc.quiz import QuestionBank
from pldc.search import BM25, Retriever, tokenize


def test_tokenize_keeps_diacritics_and_bigrams():
    assert tokenize("Thừa kế") == ["thừa", "kế", "thừa_kế"]


def test_bm25_ranks_matching_doc_first():
    docs = ["quyền sở hữu tài sản", "thừa kế theo di chúc", "hợp đồng lao động"]
    bm = BM25([tokenize(d) for d in docs])
    assert bm.scores(tokenize("di chúc")).argmax() == 1


def test_split_respects_size():
    text = "\n".join(["câu " * 60] * 10)
    assert all(len(p) <= 1000 for _, p in _split(text))


def test_split_keeps_whole_sentences_and_overlaps_by_sentence():
    sents = [f"Câu số {i} nói về quyền và nghĩa vụ của công dân." for i in range(60)]
    pieces = [p for _, p in _split(" ".join(sents))]
    assert len(pieces) > 1
    for p in pieces:
        assert p.startswith("Câu số") and p.endswith(".")
    first_of_next = pieces[1].split(". ")[0] + "."
    assert first_of_next in pieces[0]  # the next piece repeats the previous one's last sentence


def test_split_names_the_sub_part_a_piece_starts_in():
    filler = " ".join(f"Người lao động được hưởng chế độ thứ {i}." for i in range(40))
    text = f"a. Chế độ ốm đau Đối tượng áp dụng là người lao động.\n{filler}\nb) Mức hưởng khác."
    contexts = [c for c, _ in _split(text)]
    assert contexts[0] == "Chế độ ốm đau" and contexts[-1] == "Chế độ ốm đau"  # "b)" is a clause


def test_slides_carry_chapter_and_topic():
    slides = [c for c in load_all_chunks() if c.source == "slide"]
    assert all(c.chapter for c in slides)
    by_no = {c.slide: c for c in slides}
    assert "Chương 3." in by_no[39].text and "Vi phạm pháp luật" in by_no[39].text.split("\n")[0]


def test_chunks_cover_textbook_and_slides():
    chunks = load_all_chunks()
    sources = {c.source for c in chunks}
    assert sources == {"giaotrinh", "slide"}
    assert all(c.citation and c.body for c in chunks)


def test_bank_applies_overrides_and_drops_unanswerable():
    bank = QuestionBank()
    q = bank.by_id["nganhang_tn-1.3.16"]
    assert q.answers == ["D"] and len(q.options) == 4
    assert "nganhang_tn-2.3.39" not in bank.by_id  # no key in the source PDF
    assert "eduquiz-407" not in bank.by_id
    assert all(set(q.answers) <= {o["label"] for o in q.options} for q in bank.questions)


class _FakeReranker:
    """Scores passages by how often they mention "di chúc"; raises when told to."""

    def __init__(self, fail: bool = False):
        self.fail, self.seen = fail, 0

    def scores(self, query, passages):
        if self.fail:
            raise RuntimeError("GPU lost")
        self.seen = len(passages)
        return np.array([p.lower().count("di chúc") for p in passages], dtype=np.float32)


def test_reranker_reorders_the_candidate_pool():
    fake = _FakeReranker()
    hits, mode = Retriever(None, reranker=fake, rerank_candidates=15).search_with_mode(
        "điều kiện thừa kế", k=4)
    assert mode == "bm25+rerank" and fake.seen == 15 and len(hits) == 4
    counts = [h.chunk.text.lower().count("di chúc") for h in hits]
    assert counts == sorted(counts, reverse=True) and hits[0].score == counts[0]


def test_failing_reranker_keeps_first_stage_order():
    plain, _ = Retriever(None).search_with_mode("điều kiện thừa kế", k=4)
    hits, mode = Retriever(None, reranker=_FakeReranker(fail=True)).search_with_mode(
        "điều kiện thừa kế", k=4)
    assert mode == "bm25" and [h.chunk.id for h in hits] == [h.chunk.id for h in plain]


class _FlakyEmbedder:
    """Fails on its `fail_at`-th call, like a quota running out mid-build."""

    model_id = "test:fake:4"

    def __init__(self, fail_at: int | None = None):
        self.calls, self.texts, self.fail_at = 0, 0, fail_at

    def embed(self, texts, is_query=False):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("429 quota")
        self.texts += len(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


def test_interrupted_index_build_resumes_without_re_embedding(tmp_path):
    from pldc.search import build_index

    n = len(load_all_chunks())
    first = _FlakyEmbedder(fail_at=3)
    try:
        build_index(first, index_dir=tmp_path)
    except RuntimeError:
        pass
    assert first.texts == 100 and (tmp_path / "partial_vectors.npz").exists()
    second = _FlakyEmbedder()
    assert build_index(second, index_dir=tmp_path) == f"built ({n} embedded)"
    assert second.texts == n - 100  # the first two batches were kept
    assert not (tmp_path / "partial_vectors.npz").exists()
