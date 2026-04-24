"""Tests for page-level PDF chunking.

Fixture: tests/fixtures/three_pages.pdf — regenerable via
scripts/make_test_fixtures.py. Page i contains the literal string "Page i:"
followed by a section title.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperqa import Passage, chunk_by_page

FIXTURE = Path(__file__).parent / "fixtures" / "three_pages.pdf"


def test_chunk_by_page_returns_one_passage_per_page() -> None:
    passages = chunk_by_page(FIXTURE)
    assert len(passages) == 3


def test_chunk_by_page_numbers_pages_from_one() -> None:
    # WHY: citations in the wild are 1-indexed; the off-by-one fix lives in
    # chunk_by_page. If this ever flips to 0, user-facing citations break.
    passages = chunk_by_page(FIXTURE)
    assert [p.page_number for p in passages] == [1, 2, 3]


def test_chunk_by_page_preserves_page_content() -> None:
    passages = chunk_by_page(FIXTURE)
    assert "Introduction" in passages[0].text
    assert "Method" in passages[1].text
    assert "Conclusion" in passages[2].text


def test_passage_is_hashable_and_frozen() -> None:
    # WHY: downstream code may use Passage as a dict key or dedup by identity.
    # The frozen dataclass guarantee is part of the contract in ADR-0002.
    p = Passage(source_path=FIXTURE, page_number=1, text="hello")
    assert hash(p) == hash(p)
    with pytest.raises(AttributeError):
        p.page_number = 2  # type: ignore[misc]


def test_chunk_by_page_preserves_source_path() -> None:
    passages = chunk_by_page(FIXTURE)
    assert all(p.source_path == FIXTURE for p in passages)
