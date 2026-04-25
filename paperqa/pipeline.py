"""End-to-end orchestration: PDF(s) -> retriever -> answer.

WHAT: `PaperQA` ties the pipeline stages together with one public entry
      point, `ask(pdf_or_pdfs, question)`. Retrievers are cached per
      *upload set* (sorted tuple of resolved PDF paths) so repeat
      questions against the same set do not rebuild the index.
WHY:  The UI layer (Gradio app) and the measurement harness must share
      exactly the same orchestration — a single wiring module prevents UI
      and harness drift (DRY). `PaperQA` depends on the `Retriever`
      protocol (ADR-0006) rather than a concrete embedder so the visual
      retriever (ColPali) and any future hybrid retriever slot in
      without touching `PaperQA` or the answering layer. ADR-0008 widens
      the entry point to accept multiple PDFs that are pooled into a
      single index — cosine scores stay comparable because everything
      lives in the same normalized embedding space.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from paperqa.answering import Answer, Answerer, StubAnswerer
from paperqa.chunking import Passage, chunk_by_page
from paperqa.indexing import Embedder, RetrievedPassage
from paperqa.retrieval import DenseRetriever, Retriever

__all__ = ["AskResult", "PaperQA", "RetrieverFactory"]


# A factory that, given a "primary" PDF path and a (possibly pooled) list of
# passages, returns a `Retriever`. Text-only retrievers (DenseRetriever) only
# use the passages list and ignore the path — pooling N PDFs into one
# retriever Just Works. Visual retrievers (ColPali) read the path; for them
# multi-doc would need to be revisited (ADR-0006 follow-up).
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
    """Bound to a retriever-factory + answerer; caches one retriever per PDF set.

    The cache key is a sorted tuple of resolved paths — `ask([a, b])` and
    `ask([b, a])` hit the same cached retriever. Adding or removing a PDF
    rebuilds.
    """

    retriever_factory: RetrieverFactory
    answerer: Answerer = field(default_factory=StubAnswerer)
    top_k: int = 4
    _retriever_cache: dict[tuple[Path, ...], Retriever] = field(default_factory=dict, init=False)

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

    def ask(
        self,
        pdf_paths: str | Path | Sequence[str | Path],
        question: str,
    ) -> AskResult:
        """Ask `question` against one or more PDFs.

        `pdf_paths` accepts a single path-like or any sequence of them.
        Multiple PDFs are pooled into one `Retriever` so cross-document
        cosine scores are directly comparable (ADR-0008).
        """
        key = _cache_key(pdf_paths)
        retriever = self._retriever_cache.get(key)
        if retriever is None:
            pooled: list[Passage] = []
            for path in key:
                pooled.extend(chunk_by_page(path))
            # The factory's path arg only matters to retrievers that read
            # the PDF directly (e.g. ColPali). For text retrievers the
            # pooled passages already carry their own source_path each.
            # We pass the first path as a representative (callers needing
            # multi-doc visual retrieval will need a separate factory).
            retriever = self.retriever_factory(key[0], pooled)
            self._retriever_cache[key] = retriever
        hits = retriever.retrieve(question, top_k=self.top_k)
        answer = self.answerer.answer(question, hits)
        return AskResult(answer=answer, retrieved=tuple(hits))

    def forget(
        self,
        pdf_paths: str | Path | Sequence[str | Path] | None = None,
    ) -> None:
        """Drop the cached retriever for one PDF set (or all of them)."""
        if pdf_paths is None:
            self._retriever_cache.clear()
            return
        self._retriever_cache.pop(_cache_key(pdf_paths), None)


def _cache_key(pdf_paths: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    """Normalise input into a sorted tuple of resolved paths.

    Sorting makes `ask([a, b])` and `ask([b, a])` cache-equivalent. A
    bare `str | Path` is treated as a single-element sequence so the
    single-doc API still feels natural.
    """
    if isinstance(pdf_paths, str | Path):
        return (Path(pdf_paths).resolve(),)
    resolved = sorted(Path(p).resolve() for p in pdf_paths)
    if not resolved:
        raise ValueError("ask() requires at least one PDF path")
    return tuple(resolved)
