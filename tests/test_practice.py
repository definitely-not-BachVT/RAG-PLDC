from pldc import practice
from pldc.quiz import QuestionBank
from pldc.store import Store

SID = "e" * 32


def test_session_flow(tmp_path):
    bank, store = QuestionBank(), Store(tmp_path / "p.db")
    s = practice.new_session(bank, store, SID, source="nganhang_tn", chapter=6, shuffle=False)
    assert s["order"] and all(bank.by_id[q].chapter == 6 for q in s["order"])

    q1 = practice.current(bank, s)
    wrong = next(o["label"] for o in q1.options if o["label"] not in q1.answers)
    assert practice.submit(bank, store, SID, s, [wrong]) is False
    s["pos"] = 1
    q2 = practice.current(bank, s)
    assert practice.submit(bank, store, SID, s, q2.answers) is True

    summ = practice.summarize(bank, s)
    assert (summ.done, summ.correct, summ.wrong) == (2, 1, [q1.id])
    assert summ.groups[0].name == "Chương 6"

    # the wrong answer is now due for review
    review = practice.new_session(bank, store, SID, mode=practice.REVIEW)
    assert review["order"] == [q1.id]

    practice.retry_wrong(s, summ.wrong)
    assert s["order"] == [q1.id] and s["submitted"] == {}

    store.save_quiz(SID, s)
    assert practice.load_session(bank, store, SID)["order"] == [q1.id]
