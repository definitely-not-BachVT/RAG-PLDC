from pldc.evaluation import Row, parse_letters, sample, summarize
from pldc.quiz import QuestionBank


def test_parse_letters():
    v = ["A", "B", "C", "D"]
    assert parse_letters("B", v) == ["B"]
    assert parse_letters("Đáp án: C.", v) == ["C"]
    assert parse_letters("A, C", v) == ["A", "C"]
    assert parse_letters("E", v) == []  # not an option
    assert parse_letters("B\nVì A sai…", v) == ["B"]  # only the first line counts


def test_sample_is_stratified_and_reproducible():
    bank = QuestionBank()
    s1 = sample(bank, 60, seed=1)
    assert len(s1) == 60 and [q.id for q in s1] == [q.id for q in sample(bank, 60, seed=1)]
    assert {q.source for q in s1} == {"nganhang_tn", "eduquiz"}


def test_summarize_helped_hurt():
    def row(qid, mode, ok):
        return Row(qid, "eduquiz", None, mode, ["A"], ["A"], ok, "A", "P", 1.0, [])
    rows = [row("q1", "none", False), row("q1", "hybrid", True),
            row("q2", "none", True), row("q2", "hybrid", False)]
    s = summarize(rows)
    assert s["modes"]["hybrid"]["vs_none"] == {"helped": ["q1"], "hurt": ["q2"]}
    assert s["modes"]["none"]["accuracy"] == 0.5
