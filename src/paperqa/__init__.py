"""paperQA — document QA for scientific papers with page-level citations."""

from paperqa.chunking import Passage, chunk_by_page

__version__ = "0.0.1"
__all__ = ["Passage", "__version__", "chunk_by_page"]
