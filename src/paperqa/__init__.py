"""paperQA — document QA for scientific papers with page-level citations."""

from paperqa.chunking import Passage, chunk_by_page
from paperqa.indexing import Embedder, PassageIndex, RetrievedPassage

__version__ = "0.0.1"
__all__ = [
    "Embedder",
    "Passage",
    "PassageIndex",
    "RetrievedPassage",
    "__version__",
    "chunk_by_page",
]
