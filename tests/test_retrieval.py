"""Tests for the Retriever protocol and the default DenseRetriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paperqa import DenseRetriever, Passage

FAKE_PDF = Path("fake.pdf")


class KeywordEmbedder:
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


def _passage(page: int, text: str) -> Passage:
    return Passage(source_path=FAKE_PDF, page_number=page, text=text)


def test_dense_retriever_returns_top_k_passages_in_order() -> None:
    passages = [
        _passage(1, "Transformer attention mechanism."),
        _passage(2, "ImageNet training pipeline."),
        _passage(3, "Graph neural networks."),
    ]
    retriever = DenseRetriever(passages, KeywordEmbedder(["transformer", "imagenet", "graph"]))

    hits = retriever.retrieve("Tell me about the transformer.", top_k=2)

    assert [h.passage.page_number for h in hits[:1]] == [1]
    assert hits[0].score >= hits[1].score


def test_dense_retriever_clamps_top_k_to_size() -> None:
    passages = [_passage(1, "alpha"), _passage(2, "beta")]
    retriever = DenseRetriever(passages, KeywordEmbedder(["alpha", "beta"]))

    hits = retriever.retrieve("alpha", top_k=10)
    assert len(hits) == 2
