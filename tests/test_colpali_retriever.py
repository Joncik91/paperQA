"""Tests for the ColPali visual retriever.

The real model is 3B parameters and GPU-only; we never load it here.
Tests mock the model + processor + page-image rendering and assert the
retriever produces correctly ranked `RetrievedPassage`s from a known
score matrix. One integration test runs the real model when
PAPERQA_GPU_AVAILABLE=1 is set; otherwise it is skipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paperqa import Passage
from paperqa.retrievers.colpali import ColPaliRetriever

FAKE_PDF = Path("fake.pdf")


def _passage(page: int, text: str) -> Passage:
    return Passage(source_path=FAKE_PDF, page_number=page, text=text)


class _FakeScores:
    """Minimal stand-in for the (1, n_pages) tensor returned by score_multi_vector."""

    def __init__(self, row: list[float]) -> None:
        self._row = row

    def __getitem__(self, idx: int) -> _FakeScores:
        # Only [0] is ever indexed in the production code.
        assert idx == 0
        return self

    def tolist(self) -> list[float]:
        return list(self._row)


def _patch_internals(
    retriever: ColPaliRetriever,
    score_row: list[float],
    rendered_pages: int,
) -> None:
    """Stitch fake model/processor state into the retriever.

    WHY this helper: the only thing we want to vary between tests is the
    score row. Letting each test redo all of model/processor/embedding
    setup would be copy-paste; extracting it here is the DRY rule.
    """
    retriever._render_pages = MagicMock(return_value=[object()] * rendered_pages)  # type: ignore[method-assign]
    retriever._ensure_indexed = MagicMock()  # type: ignore[method-assign]
    # _score returns a (1, n_pages) tensor-like; we only need [0].tolist().
    retriever._score = MagicMock(return_value=_FakeScores(score_row))  # type: ignore[method-assign]


def test_returns_top_k_pages_in_score_order() -> None:
    passages = [
        _passage(1, "intro"),
        _passage(2, "method"),
        _passage(3, "conclusion"),
    ]
    retriever = ColPaliRetriever(pdf_path=FAKE_PDF, passages=passages, device="cpu")
    _patch_internals(retriever, score_row=[0.1, 0.9, 0.5], rendered_pages=3)

    hits = retriever.retrieve("anything", top_k=2)

    assert [h.passage.page_number for h in hits] == [2, 3]
    assert hits[0].score == pytest.approx(0.9)
    assert hits[1].score == pytest.approx(0.5)


def test_top_k_is_clamped_to_passage_count() -> None:
    passages = [_passage(1, "a"), _passage(2, "b")]
    retriever = ColPaliRetriever(pdf_path=FAKE_PDF, passages=passages, device="cpu")
    _patch_internals(retriever, score_row=[0.3, 0.7], rendered_pages=2)

    hits = retriever.retrieve("q", top_k=99)
    assert len(hits) == 2


def test_empty_passages_short_circuits_with_no_model_load() -> None:
    # WHY: zero pages must not trigger the lazy model load. That would
    # download 6 GB of weights for nothing.
    retriever = ColPaliRetriever(pdf_path=FAKE_PDF, passages=[], device="cpu")
    # Spy on _ensure_indexed; if the short-circuit works it is never called.
    retriever._ensure_indexed = MagicMock()  # type: ignore[method-assign]
    hits = retriever.retrieve("q", top_k=4)
    assert hits == []
    retriever._ensure_indexed.assert_not_called()


def test_constructor_does_not_load_model_or_render_pdf() -> None:
    # WHY: construction must be cheap; loading is deferred to first retrieve().
    # If this test fails, the lazy-init contract is broken.
    retriever = ColPaliRetriever(pdf_path=FAKE_PDF, passages=[], device="cpu")
    assert retriever._model is None
    assert retriever._processor is None
    assert retriever._page_embeddings is None


def test_device_is_explicit_when_passed() -> None:
    retriever = ColPaliRetriever(pdf_path=FAKE_PDF, passages=[], device="cuda")
    assert retriever._device == "cuda"


@pytest.mark.integration
def test_real_colpali_retrieval_when_gpu_available() -> None:
    if os.environ.get("PAPERQA_GPU_AVAILABLE") != "1":
        pytest.skip("Set PAPERQA_GPU_AVAILABLE=1 to run the real ColPali integration test.")
    pytest.importorskip("colpali_engine")
    pytest.importorskip("pypdfium2")
    pytest.importorskip("torch")

    from paperqa import chunk_by_page

    paper = Path(__file__).parent / "fixtures" / "attention_is_all_you_need.pdf"
    passages = chunk_by_page(paper)
    retriever = ColPaliRetriever(pdf_path=paper, passages=passages)

    hits = retriever.retrieve(
        "What does the ablation in Table 3 tell us about attention heads?",
        top_k=3,
    )

    assert len(hits) == 3
    pages = [h.passage.page_number for h in hits]
    # The Table 3 ablation lives on page 9. ColPali should surface it in
    # the top 3 — that is the entire reason this retriever exists. If this
    # assertion fails, ADR-0006's premise is wrong and we should rethink.
    assert 9 in pages, f"expected page 9 (Table 3) in top-3 hits; got {pages}"
