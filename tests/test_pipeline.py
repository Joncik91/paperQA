"""Tests for the end-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paperqa import PaperQA, StubAnswerer

FIXTURE = Path(__file__).parent / "fixtures" / "three_pages.pdf"


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


def test_ask_returns_answer_and_retrieved_passages() -> None:
    embedder = KeywordEmbedder(["introduction", "method", "conclusion"])
    qa = PaperQA(embedder=embedder, answerer=StubAnswerer(), top_k=2)

    result = qa.ask(FIXTURE, "Tell me about the method.")

    assert len(result.retrieved) == 2
    # Stub cites the top retrieved page.
    top_page = result.retrieved[0].passage.page_number
    assert f"[page {top_page}]" in result.answer.text
    assert result.answer.citations[0].page_number == top_page


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

    qa = PaperQA(embedder=KeywordEmbedder(["anything"]))
    qa.ask(FIXTURE, "q1")
    qa.ask(FIXTURE, "q2")
    qa.ask(FIXTURE, "q3")

    assert calls["n"] == 1


def test_forget_drops_cache() -> None:
    qa = PaperQA(embedder=KeywordEmbedder(["a"]))
    qa.ask(FIXTURE, "first")
    assert len(qa._index_cache) == 1
    qa.forget(FIXTURE)
    assert qa._index_cache == {}
