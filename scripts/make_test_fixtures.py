"""Regenerate deterministic test-fixture PDFs used by the test suite.

WHY: The committed fixtures (three_pages.pdf, two_pages.pdf) live next to
the tests so the suite has no runtime dependency on reportlab. This script
exists so the fixtures can be reproduced or evolved when chunking semantics
change, and so the generation process is auditable rather than opaque.

Run:
    pip install reportlab
    python scripts/make_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _write_pdf(filename: str, pages: list[tuple[str, str]]) -> Path:
    """Write a PDF with one page per (title, body) tuple.

    WHY a shared helper: both fixtures share the same page layout
    (title at y=720, body at y=700). Inlining the canvas calls in two
    separate functions would be the second occurrence of the pattern,
    which CONTRIBUTING §1 says to extract.
    """
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES / filename
    c = canvas.Canvas(str(out_path), pagesize=letter)
    for index, (title, body) in enumerate(pages, start=1):
        c.setFont("Helvetica", 14)
        c.drawString(72, 720, f"Page {index}: {title}")
        c.drawString(72, 700, body)
        c.showPage()
    c.save()
    return out_path


def make_three_pages() -> Path:
    """Write a 3-page PDF with predictable page content.

    WHY predictable content: tests assert on substrings present in each page.
    Any change here means updating the asserting tests in the same commit.
    """
    return _write_pdf(
        "three_pages.pdf",
        [
            ("Introduction", "Body content for Introduction - paperQA fixture."),
            ("Method", "Body content for Method - paperQA fixture."),
            ("Conclusion", "Body content for Conclusion - paperQA fixture."),
        ],
    )


def make_two_pages() -> Path:
    """Write a 2-page companion fixture for multi-doc tests (ADR-0008).

    Distinct vocabulary from three_pages.pdf so tests can assert
    cross-document retrieval routes pages back to the right source.
    """
    return _write_pdf(
        "two_pages.pdf",
        [
            ("Alpha", "Body content for Alpha - second paperQA fixture."),
            ("Beta", "Body content for Beta - second paperQA fixture."),
        ],
    )


if __name__ == "__main__":
    for path in (make_three_pages(), make_two_pages()):
        print(f"wrote {path}")
