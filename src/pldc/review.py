"""Spaced repetition for wrong answers (Leitner boxes).

A wrongly answered question enters box 1 (due immediately, review the same day).
Each correct review moves it up a box and pushes the next review further out;
a wrong answer sends it back to box 1. After a correct answer in the last box it is
considered mastered and leaves the review queue.
"""
from __future__ import annotations

DAY = 86_400
# box -> days until the next review
INTERVAL_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 16}
MASTERED = len(INTERVAL_DAYS) + 1


def schedule(box: int | None, correct: bool, now: float) -> tuple[int, float] | None:
    """New (box, due timestamp) after an answer, or None if the question needs no review.

    `box` is None for a question that is not in the review queue yet.
    """
    if box is None:
        if correct:
            return None  # answered right the first time: nothing to review
        box = 1
    elif correct:
        box += 1
        if box >= MASTERED:
            return MASTERED, float("inf")
    else:
        box = 1
    return box, now + INTERVAL_DAYS[box] * DAY
