"""Concrete `Embedder` implementations.

WHAT: `SentenceTransformerEmbedder` wraps a sentence-transformers model and
      satisfies the `Embedder` protocol from `paperqa.indexing`. Default
      model is all-MiniLM-L6-v2 (ADR-0003).
WHY:  Kept in its own module so `paperqa.indexing` can be imported (and
      unit-tested) without paying the sentence-transformers import cost.
      CI uses a fake embedder; this module is exercised only in integration
      tests and at runtime.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

__all__ = ["SentenceTransformerEmbedder"]

# ADR-0003 locks this as the v1 default. If it changes, bump the ADR in the
# same commit — the string appears in user-visible logs and is part of the
# reproducibility contract.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Embeds text using a sentence-transformers model.

    The underlying model is loaded lazily on first `.embed()` call so that
    importing this module (e.g. during CLI startup or tests that don't touch
    embeddings) does not pay the model-load cost.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        # WHY lazy import: sentence-transformers pulls torch + transformers,
        # adding ~1s cold-start and ~500MB of deps. Only pay it when
        # embedding is actually requested.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (N, dim)."""
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
