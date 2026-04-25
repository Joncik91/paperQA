"""Embedding + in-memory cosine retrieval over page-level passages.

WHAT: `Embedder` is the minimal protocol every embedding backend must satisfy.
      `PassageIndex.build(passages, embedder)` embeds every passage once.
      `PassageIndex.query(question, embedder, top_k)` returns the top_k
      passages with their cosine similarity scores.
WHY:  Single-paper QA never needs a vector database — per ADR-0003, a NumPy
      matrix + dot product is auditable, fast, and has zero infra. The
      `Embedder` protocol keeps the retrieval layer swappable (MiniLM today,
      ColPali or a hosted API later) without touching `PassageIndex`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from paperqa.chunking import Passage

__all__ = ["Embedder", "PassageIndex", "RetrievedPassage"]


class Embedder(Protocol):
    """Minimal contract for anything that turns strings into unit vectors.

    WHY unit vectors: normalising at the source lets retrieval be a single
    dot product — no per-query renormalisation, no numerical surprises.
    Implementations must normalise before returning.
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), L2-normalised."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """A passage paired with its retrieval score.

    WHY score is exposed: downstream prompting and UI both want to show or
    threshold on confidence. Keeping it on the retrieval record (not on
    `Passage`) preserves `Passage` as a pure data contract (ADR-0002).
    """

    passage: Passage
    score: float


class PassageIndex:
    """In-memory cosine-similarity index over a fixed passage set.

    WHY fixed set: v1 scope is single-paper QA (ADR-0003). Incremental
    updates and multi-document corpora are out of scope — attempting them
    here would add state the v1 product does not need.
    """

    def __init__(self, passages: list[Passage], embeddings: np.ndarray) -> None:
        # WHY this invariant: guards against shape drift between the passage
        # list and the embedding matrix, which would silently misalign scores
        # with the wrong passages at query time.
        if len(passages) != embeddings.shape[0]:
            raise ValueError(
                f"passages ({len(passages)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )
        self._passages = passages
        self._matrix = embeddings.astype(np.float32, copy=False)

    @classmethod
    def build(cls, passages: list[Passage], embedder: Embedder) -> PassageIndex:
        """Embed `passages` once and return a ready-to-query index.

        Empty passage lists are accepted and produce an empty index.
        """
        if not passages:
            return cls(passages=[], embeddings=np.empty((0, 0), dtype=np.float32))
        matrix = embedder.embed([p.text for p in passages])
        return cls(passages=passages, embeddings=matrix)

    def query(
        self,
        question: str,
        embedder: Embedder,
        top_k: int = 4,
    ) -> list[RetrievedPassage]:
        """Return up to `top_k` passages ranked by cosine similarity.

        WHY clamp top_k: callers (UI, evals) often pass a default like 4 even
        for tiny papers; silently returning fewer beats raising.
        """
        if not self._passages:
            return []
        query_vec = embedder.embed([question])[0]
        scores = self._matrix @ query_vec
        k = min(top_k, len(self._passages))
        # argpartition is O(n); argsort only the k candidates. Order matters
        # here because retrieval results feed directly into the prompt.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            RetrievedPassage(passage=self._passages[i], score=float(scores[i])) for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._passages)
