"""End-to-end orchestration: PDF -> retriever -> answer.

WHAT: `PaperQA` ties the pipeline stages together with one public entry
      point, `ask(pdf, question)`. Retrievers are cached per source PDF so
      that repeat questions against the same paper do not rebuild the index.
WHY:  The UI layer (Gradio app) and the measurement harness must share
      exactly the same orchestration — a single wiring module prevents UI
      and harness drift (DRY). `PaperQA` depends on the `Retriever`
      protocol (ADR-0006) rather than a concrete embedder so the visual
      retriever (ColPali) and any future hybrid retriever slot in
      without touching `PaperQA` or the answering layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from paperqa.answering import Answer, Answerer, StubAnswerer
from paperqa.chunking import Passage, chunk_by_page
from paperqa.indexing import Embedder, RetrievedPassage
from paperqa.retrieval import DenseRetriever, Retriever

__all__ = ["AskResult", "PaperQA", "RetrieverFactory"]


# A factory that, given a PDF path and its already-extracted passages,
# returns a `Retriever` for that document. Text-only retrievers ignore the
# PDF path; visual retrievers ignore the passages and re-render the PDF.
RetrieverFactory = Callable[[Path, list[Passage]], Retriever]


@dataclass(frozen=True, slots=True)
class AskResult:
    """Full payload of a single `ask()` call.

    Exposing the retrieved hits (not just the final `Answer`) lets the UI
    render source excerpts next to the answer and lets the measurement
    harness reason about retrieval quality independently of generation.
    """

    answer: Answer
    retrieved: tuple[RetrievedPassage, ...]


@dataclass
class PaperQA:
    """Bound to a retriever-factory + answerer; caches one retriever per PDF.

    v1 scope (ADR-0003) is single-paper QA. The session-level cache lets
    the UI hold one retriever across multiple questions without rebuilding
    the index on every turn.
    """

    retriever_factory: RetrieverFactory
    answerer: Answerer = field(default_factory=StubAnswerer)
    top_k: int = 4
    _retriever_cache: dict[Path, Retriever] = field(default_factory=dict, init=False)

    @classmethod
    def with_embedder(
        cls,
        embedder: Embedder,
        answerer: Answerer | None = None,
        top_k: int = 4,
    ) -> PaperQA:
        """Convenience constructor for the default text-embedding path.

        WHY: most callers (the Gradio app, the measurement harness, almost
        every test) use `DenseRetriever` over MiniLM. Spelling that out as
        a `RetrieverFactory` lambda at every call site is noise; this
        keeps the common case one line.
        """

        def factory(_pdf: Path, passages: list[Passage]) -> Retriever:
            return DenseRetriever(passages, embedder)

        return cls(
            retriever_factory=factory,
            answerer=answerer or StubAnswerer(),
            top_k=top_k,
        )

    def ask(self, pdf_path: str | Path, question: str) -> AskResult:
        key = Path(pdf_path).resolve()
        retriever = self._retriever_cache.get(key)
        if retriever is None:
            passages = chunk_by_page(key)
            retriever = self.retriever_factory(key, passages)
            self._retriever_cache[key] = retriever
        hits = retriever.retrieve(question, top_k=self.top_k)
        answer = self.answerer.answer(question, hits)
        return AskResult(answer=answer, retrieved=tuple(hits))

    def forget(self, pdf_path: str | Path | None = None) -> None:
        """Drop cached retriever for one PDF (or all of them)."""
        if pdf_path is None:
            self._retriever_cache.clear()
            return
        self._retriever_cache.pop(Path(pdf_path).resolve(), None)
