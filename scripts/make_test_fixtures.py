"""Regenerate deterministic test-fixture PDFs used by the test suite.

WHY: The three-page fixture at tests/fixtures/three_pages.pdf is committed so
the test suite has no runtime dependency on reportlab. This script exists so
the fixture can be reproduced or evolved when chunking semantics change, and
so the generation process is auditable rather than opaque.

Run:
    pip install reportlab
    python scripts/make_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def make_three_pages() -> Path:
    """Write a 3-page PDF with predictable page content.

    WHY predictable content: tests assert on substrings present in each page.
    Any change here means updating the asserting tests in the same commit.
    """
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES / "three_pages.pdf"
    c = canvas.Canvas(str(out_path), pagesize=letter)
    for index, title in enumerate(["Introduction", "Method", "Conclusion"], start=1):
        c.setFont("Helvetica", 14)
        c.drawString(72, 720, f"Page {index}: {title}")
        c.drawString(72, 700, f"Body content for {title} - paperQA fixture.")
        c.showPage()
    c.save()
    return out_path


if __name__ == "__main__":
    path = make_three_pages()
    print(f"wrote {path}")
