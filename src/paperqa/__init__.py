"""paperQA — document QA for scientific papers with page-level citations."""

from paperqa.answering import (
    Answer,
    Answerer,
    Citation,
    StubAnswerer,
    build_prompt,
    parse_citations,
)
from paperqa.chunking import Passage, chunk_by_page
from paperqa.indexing import Embedder, PassageIndex, RetrievedPassage
from paperqa.pipeline import AskResult, PaperQA

__version__ = "0.0.1"
__all__ = [
    "Answer",
    "Answerer",
    "AskResult",
    "Citation",
    "Embedder",
    "PaperQA",
    "Passage",
    "PassageIndex",
    "RetrievedPassage",
    "StubAnswerer",
    "__version__",
    "build_prompt",
    "chunk_by_page",
    "parse_citations",
]
