from pldc.review import DAY, MASTERED, schedule
from pldc.store import Store

SID = "c" * 32


def test_schedule_progression():
    assert schedule(None, True, 0) is None  # right first time: not queued
    assert schedule(None, False, 0) == (1, 0)  # wrong: due now
    assert schedule(1, True, 0) == (2, 1 * DAY)
    assert schedule(4, False, 0) == (1, 0)  # wrong again: back to box 1
    assert schedule(5, True, 0)[0] == MASTERED


def test_store_review_queue(tmp_path):
    s = Store(tmp_path / "r.db")
    s.record_answer(SID, "q1", correct=False, now=100)
    s.record_answer(SID, "q2", correct=True, now=100)  # never queued
    assert s.due_reviews(SID, now=100) == ["q1"]

    s.record_answer(SID, "q1", correct=True, now=200)  # box 2, due in 1 day
    assert s.due_reviews(SID, now=200) == []
    assert s.due_reviews(SID, now=200 + DAY) == ["q1"]
    assert s.review_stats(SID, now=200) == {"due": 0, "active": 1, "mastered": 0}

    for t in range(4):  # boxes 2 -> 3 -> 4 -> 5 -> mastered
        s.record_answer(SID, "q1", correct=True, now=10**9 + t)
    assert s.review_stats(SID, now=10**9) == {"due": 0, "active": 0, "mastered": 1}
