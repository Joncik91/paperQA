"""Tests for the end-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paperqa import PaperQA, StubAnswerer

FIXTURE = Path(__file__).parent / "fixtures" / "three_pages.pdf"
SECOND_FIXTURE = Path(__file__).parent / "fixtures" / "two_pages.pdf"


class KeywordEmbedder:
    """Same pattern as test_indexing: one dim per keyword, L2-normalised."""

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


# ---------- single-doc path ----------


def test_ask_returns_answer_and_retrieved_passages() -> None:
    embedder = KeywordEmbedder(["introduction", "method", "conclusion"])
    qa = PaperQA.with_embedder(embedder, answerer=StubAnswerer(), top_k=2)

    result = qa.ask(FIXTURE, "Tell me about the method.")

    assert len(result.retrieved) == 2
    top_page = result.retrieved[0].passage.page_number
    top_name = result.retrieved[0].passage.source_path.name
    # Stub cites the top retrieved passage in `[file, page]` form (ADR-0008).
    assert f"[{top_name}, page {top_page}]" in result.answer.text
    assert result.answer.citations[0].page_number == top_page
    assert result.answer.citations[0].source_path.name == top_name


def test_index_is_cached_per_pdf(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # WHY: repeat questions must not re-embed. Verified by counting chunker
    # calls — if the cache works, chunk_by_page runs exactly once per PDF.
    calls = {"n": 0}
    from paperqa import pipeline as pipeline_mod

    original = pipeline_mod.chunk_by_page

    def counting(path: str | Path) -> list:  # type: ignore[type-arg]
        calls["n"] += 1
        return original(path)

    monkeypatch.setattr(pipeline_mod, "chunk_by_page", counting)

    qa = PaperQA.with_embedder(KeywordEmbedder(["anything"]))
    qa.ask(FIXTURE, "q1")
    qa.ask(FIXTURE, "q2")
    qa.ask(FIXTURE, "q3")

    assert calls["n"] == 1


def test_forget_drops_cache() -> None:
    qa = PaperQA.with_embedder(KeywordEmbedder(["a"]))
    qa.ask(FIXTURE, "first")
    assert len(qa._retriever_cache) == 1
    qa.forget(FIXTURE)
    assert qa._retriever_cache == {}


# ---------- multi-doc path (ADR-0008) ----------


@pytest.mark.skipif(
    not SECOND_FIXTURE.exists(),
    reason="second fixture not generated yet — run scripts/make_test_fixtures.py",
)
def test_multi_doc_pools_passages_from_all_pdfs() -> None:
    embedder = KeywordEmbedder(["introduction", "method", "conclusion", "alpha"])
    qa = PaperQA.with_embedder(embedder, answerer=StubAnswerer(), top_k=5)

    result = qa.ask([FIXTURE, SECOND_FIXTURE], "Tell me about the introduction.")

    # The pool contains pages from both PDFs (3 + 2 = 5 total).
    sources = {hit.passage.source_path.name for hit in result.retrieved}
    assert sources == {FIXTURE.name, SECOND_FIXTURE.name}


@pytest.mark.skipif(
    not SECOND_FIXTURE.exists(),
    reason="second fixture not generated yet — run scripts/make_test_fixtures.py",
)
def test_multi_doc_cache_is_invariant_under_input_order(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    # WHY: ask([a, b]) and ask([b, a]) must hit the same cache entry.
    # Sorted-tuple cache key (ADR-0008) makes that invariant.
    calls = {"n": 0}
    from paperqa import pipeline as pipeline_mod

    original = pipeline_mod.chunk_by_page

    def counting(path: str | Path) -> list:  # type: ignore[type-arg]
        calls["n"] += 1
        return original(path)

    monkeypatch.setattr(pipeline_mod, "chunk_by_page", counting)

    qa = PaperQA.with_embedder(KeywordEmbedder(["anything"]))
    qa.ask([FIXTURE, SECOND_FIXTURE], "q1")
    qa.ask([SECOND_FIXTURE, FIXTURE], "q2")  # reversed order

    # Each fixture chunked exactly once, even though the call order flipped.
    assert calls["n"] == 2
    assert len(qa._retriever_cache) == 1


def test_ask_with_empty_list_raises() -> None:
    qa = PaperQA.with_embedder(KeywordEmbedder(["x"]))
    with pytest.raises(ValueError, match="at least one PDF"):
        qa.ask([], "q?")


def test_single_doc_and_list_of_one_share_a_cache_entry() -> None:
    # ask(FIXTURE) and ask([FIXTURE]) normalise to the same key.
    qa = PaperQA.with_embedder(KeywordEmbedder(["x"]))
    qa.ask(FIXTURE, "q1")
    qa.ask([FIXTURE], "q2")
    assert len(qa._retriever_cache) == 1
