"""Retriever protocol and the default dense (text-embedding) implementation.

WHAT: `Retriever` is the abstraction every retrieval backend satisfies.
      `DenseRetriever` is the v1 default — wraps a `PassageIndex` + an
      `Embedder` so the existing MiniLM path keeps working with no
      behaviour change.
WHY:  Visual retrievers (e.g. ColPali, ADR-0006) score pages with
      multi-vector late interaction, not single-vector cosine. Forcing
      them through `Embedder.embed -> PassageIndex.query` would lose the
      patch-level signal entirely. Promoting `Retriever` to the protocol
      `PaperQA` depends on means future backends slot in without
      touching the orchestrator or the answering layer.
"""

from __future__ import annotations

from typing import Protocol

from paperqa.chunking import Passage
from paperqa.indexing import Embedder, PassageIndex, RetrievedPassage

__all__ = ["DenseRetriever", "Retriever"]


class Retriever(Protocol):
    """Anything that ranks passages of a single document against a question."""

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]: ...


class DenseRetriever:
    """Default retriever — single-vector cosine over MiniLM-style embeddings.

    Thin wrapper over `PassageIndex` so the v1 retrieval semantics
    (ADR-0003) are reachable through the new `Retriever` protocol with
    zero behaviour change.
    """

    def __init__(self, passages: list[Passage], embedder: Embedder) -> None:
        self._embedder = embedder
        self._index = PassageIndex.build(passages, embedder)

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        return self._index.query(question, self._embedder, top_k=top_k)

    def __len__(self) -> int:
        return len(self._index)
