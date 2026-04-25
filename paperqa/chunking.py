"""Page-level chunking of PDFs into retrievable passages.

WHAT: Turns a PDF into one `Passage` per page. Each passage carries the
      1-indexed page number and the raw extracted text.
WHY:  Page number is paperQA's citation grain (see ADR-0002). Making the
      chunking unit equal to the citation unit means citations are faithful
      by construction — no mapping layer, no drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

__all__ = ["Passage", "chunk_by_page"]


@dataclass(frozen=True, slots=True)
class Passage:
    """A chunk of a source document addressable by page number.

    WHAT: The canonical data contract passed between ingest, index, retrieve,
          and answer stages.
    WHY:  A single shared type keeps the pipeline stages decoupled; any stage
          can be swapped without touching the others (see ADR-0002).
    """

    source_path: Path
    page_number: int
    text: str


def chunk_by_page(pdf_path: str | Path) -> list[Passage]:
    """Extract one `Passage` per page from `pdf_path`.

    Pages are 1-indexed to match how humans and citation strings refer to them.

    WHY 1-indexed: `pypdf` gives us 0-indexed pages, but every citation in the
    wild ("see p. 3", "Figure 2 on page 4") is 1-indexed. Translating once here
    prevents off-by-one bugs leaking into the UI and prompts.
    """
    path = Path(pdf_path)
    reader = PdfReader(path)
    return [
        Passage(
            source_path=path,
            page_number=index + 1,
            text=page.extract_text() or "",
        )
        for index, page in enumerate(reader.pages)
    ]
