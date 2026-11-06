"""The checkers themselves must be right, or the golden tests prove nothing."""
from pldc.answer_checks import (cited_numbers, cited_sections, invalid_citations, missing_facts,
                                section_matches, stated_correct_letter)
from pldc.chunking import Chunk
from pldc.search import Hit


def hit(section: str) -> Hit:
    return Hit(Chunk(id=section, source="giaotrinh", citation=f"mục {section}", text="", body="",
                     section_id=section), 0.0)


def test_missing_facts_any_of_and_accents():
    answer = "Thời giờ làm việc không quá 8 giờ/ngày và 48 giờ/tuần do Toà án…"
    assert missing_facts(answer, [["08 giờ", "8 giờ"], ["48 giờ"]]) == []
    assert missing_facts(answer, [["tòa án"]]) == []  # òa / oà spellings are equivalent
    assert missing_facts(answer, [["chế tài"]]) == [["chế tài"]]


def test_citations():
    answer = "Ý một [1]. Ý hai [2], [3]. Không phải nguồn: [12 tháng]"
    assert cited_numbers(answer) == [1, 2, 3]
    assert invalid_citations(answer, n_sources=2) == [3]
    assert cited_sections("theo [2]", [hit("6.1.3"), hit("6.1.3.1")]) == {"6.1.3.1"}


def test_section_matches_nested():
    assert section_matches({"6.1.3.1"}, ["6.1.3"])
    assert not section_matches({"6.1.30"}, ["6.1.3"])
    assert not section_matches(set(), ["6.1.3"])


def test_stated_correct_letter():
    assert stated_correct_letter("### 1. Vì sao đáp án A là đúng?") == "A"
    assert stated_correct_letter("Đáp án đúng là **D** vì…") == "D"
    assert stated_correct_letter("Đáp án: (B)") == "B"
    assert stated_correct_letter("Không nói rõ đáp án nào.") is None
