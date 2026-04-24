"""Tests for the embedding-index layer.

These use a handcrafted fake embedder so the test suite never downloads a
model. The real MiniLM embedder (`paperqa.embedders`) is covered by
integration tests that run outside CI by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paperqa import Passage, PassageIndex


class KeywordEmbedder:
    """Deterministic embedder: one dimension per keyword, L2-normalised.

    WHY this design: every test's expected ranking is obvious from the
    keyword overlap — no need to reason about cosine geometry to predict
    which passage will win.
    """

    def __init__(self, keywords: list[str]) -> None:
        self._keywords = keywords
        self._dim = len(keywords)

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


FAKE_PDF = Path("fake.pdf")


def _passage(page: int, text: str) -> Passage:
    return Passage(source_path=FAKE_PDF, page_number=page, text=text)


def test_index_is_empty_for_empty_passages() -> None:
    index = PassageIndex.build(passages=[], embedder=KeywordEmbedder(["x"]))
    assert len(index) == 0
    assert index.query("anything", KeywordEmbedder(["x"]), top_k=4) == []


def test_query_returns_most_similar_passage_first() -> None:
    passages = [
        _passage(1, "The transformer architecture uses attention."),
        _passage(2, "Training on ImageNet requires a GPU cluster."),
        _passage(3, "Graph neural networks model relational data."),
    ]
    embedder = KeywordEmbedder(["transformer", "imagenet", "graph"])
    index = PassageIndex.build(passages, embedder)

    results = index.query("Tell me about the transformer.", embedder, top_k=3)

    assert [r.passage.page_number for r in results[:1]] == [1]
    assert results[0].score > results[1].score


def test_query_top_k_is_clamped_to_index_size() -> None:
    passages = [_passage(1, "alpha"), _passage(2, "beta")]
    embedder = KeywordEmbedder(["alpha", "beta"])
    index = PassageIndex.build(passages, embedder)

    results = index.query("alpha", embedder, top_k=10)

    assert len(results) == 2


def test_scores_are_sorted_descending() -> None:
    passages = [
        _passage(1, "alpha"),
        _passage(2, "alpha beta"),
        _passage(3, "beta gamma"),
    ]
    embedder = KeywordEmbedder(["alpha", "beta", "gamma"])
    index = PassageIndex.build(passages, embedder)

    results = index.query("alpha beta", embedder, top_k=3)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_mismatched_lengths_raise() -> None:
    # WHY: the invariant guards against silently returning scores for the
    # wrong passage if a caller ever constructs the index directly.
    with pytest.raises(ValueError, match="length mismatch"):
        PassageIndex(
            passages=[_passage(1, "a"), _passage(2, "b")],
            embeddings=np.zeros((1, 3), dtype=np.float32),
        )
