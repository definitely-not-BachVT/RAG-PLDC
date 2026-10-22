"""Footnotes in AI answers are numbered by appearance, and each maps back to its source."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "ui"))

from sources import cite_refs, source_for_footnote, source_rows, with_cites  # noqa: E402

SOURCES = [{"citation": f"mục {i}"} for i in range(1, 6)]
TEXT = "A [1]. B [1]. C [1]. D [2]. E [5]. F [9]."  # [9] does not exist


def test_footnotes_number_each_citation_not_each_source():
    out = with_cites(TEXT, len(SOURCES))
    assert [f'data-n="{k}">{k}<' in out for k in range(1, 6)] == [True] * 5
    assert "[9]" in out  # unknown source left as plain text


def test_footnote_maps_back_to_its_source():
    assert cite_refs(TEXT, 5) == [1, 1, 1, 2, 5]
    assert source_for_footnote(TEXT, SOURCES, 4) == (2, SOURCES[1])
    assert source_for_footnote(TEXT, SOURCES, 6) is None


def test_source_list_groups_footnotes_and_skips_uncited_sources():
    rows = source_rows(TEXT, SOURCES)
    assert [(ks, n) for ks, n, _ in rows] == [([1, 2, 3], 1), ([4], 2), ([5], 5)]


def test_answer_without_citations_lists_all_sources():
    assert [n for _, n, _ in source_rows("không dẫn nguồn", SOURCES)] == [1, 2, 3, 4, 5]
