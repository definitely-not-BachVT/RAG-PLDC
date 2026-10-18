from pldc.chunking import _split, load_all_chunks
from pldc.quiz import QuestionBank
from pldc.search import BM25, tokenize


def test_tokenize_keeps_diacritics_and_bigrams():
    assert tokenize("Thừa kế") == ["thừa", "kế", "thừa_kế"]


def test_bm25_ranks_matching_doc_first():
    docs = ["quyền sở hữu tài sản", "thừa kế theo di chúc", "hợp đồng lao động"]
    bm = BM25([tokenize(d) for d in docs])
    assert bm.scores(tokenize("di chúc")).argmax() == 1


def test_split_respects_size():
    text = "\n".join(["câu " * 60] * 10)
    assert all(len(p) <= 1000 for p in _split(text))


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
