"""Tests for the offline measurement harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paperqa import PaperQA, StubAnswerer
from paperqa.evaluation import (
    GoldItem,
    load_gold_set,
    recall_at_k,
    run_report,
    score_citations,
)

GOLD_PATH = Path(__file__).parent / "eval" / "gold.json"
PAPER_FIXTURE = Path(__file__).parent / "fixtures" / "attention_is_all_you_need.pdf"
SIMPLE_FIXTURE = Path(__file__).parent / "fixtures" / "three_pages.pdf"


class KeywordEmbedder:
    """One dim per keyword, L2-normalised — same pattern used elsewhere."""

    def __init__(self, keywords: list[str]) -> None:
        self._keywords = keywords

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vec = np.array(
                [1.0 if kw in lowered else 0.0 for kw in self._keywords],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
        return np.vstack(rows)


def test_recall_at_k_basic() -> None:
    assert recall_at_k([3, 1, 2], frozenset({1, 2}), k=3) == 1.0
    assert recall_at_k([3, 1, 2], frozenset({1, 2}), k=1) == 0.0
    assert recall_at_k([3, 1], frozenset({1, 2}), k=5) == 0.5


def test_recall_at_k_handles_empty_gold() -> None:
    # WHY: a malformed entry should not divide-by-zero the whole run.
    assert recall_at_k([1, 2], frozenset(), k=3) == 0.0


def test_score_citations_counts_only_relevant() -> None:
    assert score_citations([1, 2, 4], frozenset({1, 2})) == pytest.approx(2 / 3)
    assert score_citations([], frozenset({1})) == 1.0  # nothing unfaithful
    assert score_citations([7], frozenset({1})) == 0.0


def test_load_gold_set_resolves_relative_paths() -> None:
    items = load_gold_set(GOLD_PATH)
    # All questions point at the committed Attention Is All You Need PDF.
    assert len(items) >= 3
    assert all(item.pdf_path == PAPER_FIXTURE.resolve() for item in items)
    # must_cite_page must be part of relevant_pages — guard against gold
    # drift where the two fields get out of sync.
    for item in items:
        assert item.must_cite_page in item.relevant_pages


def test_run_report_on_inline_gold_with_keyword_embedder() -> None:
    # WHY inline gold: keeps this test hermetic and decoupled from the
    # committed gold.json, so evolving the real gold set does not break
    # the harness's unit tests.
    embedder = KeywordEmbedder(["introduction", "method", "conclusion"])
    qa = PaperQA(embedder=embedder, answerer=StubAnswerer(), top_k=3)

    inline_gold = [
        GoldItem(
            pdf_path=SIMPLE_FIXTURE,
            question="Tell me about the introduction.",
            relevant_pages=frozenset({1}),
            must_cite_page=1,
        ),
        GoldItem(
            pdf_path=SIMPLE_FIXTURE,
            question="Describe the method section.",
            relevant_pages=frozenset({2}),
            must_cite_page=2,
        ),
        GoldItem(
            pdf_path=SIMPLE_FIXTURE,
            question="What does the conclusion say?",
            relevant_pages=frozenset({3}),
            must_cite_page=3,
        ),
    ]

    report = run_report(qa, inline_gold)

    assert report.n_questions == 3
    assert report.mean_recall_at_1 == 1.0
    assert report.mean_recall_at_3 == 1.0
    assert report.mean_citation_faithfulness == 1.0
    assert report.must_cite_rate == 1.0


def test_run_report_handles_empty_gold() -> None:
    qa = PaperQA(embedder=KeywordEmbedder(["anything"]))
    report = run_report(qa, gold=[])

    assert report.n_questions == 0
    assert report.per_question == ()
    assert report.mean_recall_at_1 == 0.0


def test_run_report_penalises_hallucinated_citations() -> None:
    # Force-cite a page that is retrieved but NOT in the gold relevant set.
    gold = [
        GoldItem(
            pdf_path=SIMPLE_FIXTURE,
            question="Describe the method section.",
            relevant_pages=frozenset({2}),
            must_cite_page=2,
        )
    ]
    # KeywordEmbedder with only "introduction" puts page 1 first — so the
    # stub will cite page 1, which is NOT in the gold set for this question.
    embedder = KeywordEmbedder(["introduction"])
    qa = PaperQA(embedder=embedder, answerer=StubAnswerer(), top_k=1)

    report = run_report(qa, gold)

    assert report.mean_citation_faithfulness == 0.0
    assert report.must_cite_rate == 0.0
