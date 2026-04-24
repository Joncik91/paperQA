"""End-to-end orchestration: PDF -> passages -> index -> answer.

WHAT: `PaperQA` ties the pipeline stages together with one public entry
      point, `ask(question)`. Indexes are cached per source PDF so that
      repeat questions against the same paper do not re-embed every page.
WHY:  The UI layer (Gradio app) and any future CLI must share exactly the
      same orchestration — a single wiring module prevents UI and CLI
      drift (DRY).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from paperqa.answering import Answer, Answerer, StubAnswerer
from paperqa.chunking import chunk_by_page
from paperqa.indexing import Embedder, PassageIndex, RetrievedPassage

__all__ = ["AskResult", "PaperQA"]


@dataclass(frozen=True, slots=True)
class AskResult:
    """Full payload of a single `ask()` call.

    Exposing the retrieved hits (not just the final `Answer`) lets the UI
    render source excerpts next to the answer and lets evals reason about
    retrieval quality independently of generation quality.
    """

    answer: Answer
    retrieved: tuple[RetrievedPassage, ...]


@dataclass
class PaperQA:
    """Bound to an embedder + answerer; caches one index per source path.

    v1 scope (ADR-0003) is single-paper QA, but the session-level cache
    means the UI can hold one index across multiple questions without
    re-embedding on every turn.
    """

    embedder: Embedder
    answerer: Answerer = field(default_factory=StubAnswerer)
    top_k: int = 4
    _index_cache: dict[Path, PassageIndex] = field(default_factory=dict, init=False)

    def ask(self, pdf_path: str | Path, question: str) -> AskResult:
        key = Path(pdf_path).resolve()
        index = self._index_cache.get(key)
        if index is None:
            index = PassageIndex.build(chunk_by_page(key), self.embedder)
            self._index_cache[key] = index
        hits = index.query(question, self.embedder, top_k=self.top_k)
        answer = self.answerer.answer(question, hits)
        return AskResult(answer=answer, retrieved=tuple(hits))

    def forget(self, pdf_path: str | Path | None = None) -> None:
        """Drop cached index for one PDF (or all of them)."""
        if pdf_path is None:
            self._index_cache.clear()
            return
        self._index_cache.pop(Path(pdf_path).resolve(), None)
