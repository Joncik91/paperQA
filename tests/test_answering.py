"""Tests for the answering layer — prompt shape, citation parsing, stub."""

from __future__ import annotations

from pathlib import Path

from paperqa import (
    Answer,
    Passage,
    RetrievedPassage,
    StubAnswerer,
    build_prompt,
    parse_citations,
)
from paperqa.answering import SYSTEM_INSTRUCTION

FAKE_PDF = Path("fake.pdf")
SECOND_PDF = Path("second.pdf")


def _hit(
    page: int,
    text: str,
    score: float = 1.0,
    source: Path = FAKE_PDF,
) -> RetrievedPassage:
    return RetrievedPassage(
        passage=Passage(source_path=source, page_number=page, text=text),
        score=score,
    )


# ---------- build_prompt ----------


def test_build_prompt_contains_system_instruction_and_passages() -> None:
    prompt = build_prompt(
        "What is the method?",
        [_hit(1, "Transformer architecture."), _hit(2, "Attention is key.")],
    )
    assert SYSTEM_INSTRUCTION in prompt
    # New multi-doc citation header shape: [<filename>, page N]
    assert "[fake.pdf, page 1]" in prompt
    assert "[fake.pdf, page 2]" in prompt
    assert "Transformer architecture." in prompt
    assert prompt.rstrip().endswith("ANSWER:")


def test_build_prompt_uses_retrieval_order() -> None:
    # The ranking carried in the list IS the signal; the prompt must
    # preserve it so the model attends to the highest-scored passage first.
    prompt = build_prompt(
        "q?",
        [_hit(5, "alpha"), _hit(2, "beta"), _hit(9, "gamma")],
    )
    alpha = prompt.index("alpha")
    beta = prompt.index("beta")
    gamma = prompt.index("gamma")
    assert alpha < beta < gamma


def test_build_prompt_renders_each_pdf_filename() -> None:
    prompt = build_prompt(
        "q?",
        [_hit(1, "alpha", source=FAKE_PDF), _hit(3, "gamma", source=SECOND_PDF)],
    )
    assert "[fake.pdf, page 1]" in prompt
    assert "[second.pdf, page 3]" in prompt


# ---------- parse_citations: multi-doc preferred form ----------


def test_parse_citations_extracts_filename_and_page() -> None:
    passages = [
        _hit(1, "a", score=0.9, source=FAKE_PDF),
        _hit(3, "b", score=0.5, source=SECOND_PDF),
    ]
    text = "From [fake.pdf, page 1] and from [second.pdf, page 3]."

    citations = parse_citations(text, passages)

    assert [(c.source_path.name, c.page_number) for c in citations] == [
        ("fake.pdf", 1),
        ("second.pdf", 3),
    ]
    assert citations[0].score == 0.9
    assert citations[1].score == 0.5


def test_parse_citations_drops_unretrieved_filename_or_page() -> None:
    # Both the filename and the (filename, page) pair must have been retrieved.
    passages = [_hit(1, "a", source=FAKE_PDF)]
    text = (
        "Real [fake.pdf, page 1]. Fake-page [fake.pdf, page 99]. Fake-file [imaginary.pdf, page 1]."
    )

    citations = parse_citations(text, passages)

    assert [(c.source_path.name, c.page_number) for c in citations] == [("fake.pdf", 1)]


def test_parse_citations_preserves_order_and_duplicates() -> None:
    passages = [_hit(3, "x", source=FAKE_PDF)]
    citations = parse_citations("see [fake.pdf, page 3], also [fake.pdf, page 3] again", passages)
    assert [c.page_number for c in citations] == [3, 3]


def test_parse_citations_is_case_insensitive() -> None:
    passages = [_hit(7, "y", source=FAKE_PDF)]
    citations = parse_citations("see [fake.pdf, Page 7] and [fake.pdf, PAGE 7]", passages)
    assert len(citations) == 2


# ---------- parse_citations: back-compat [page N] without filename ----------


def test_parse_citations_back_compat_single_doc() -> None:
    # `[page N]` with no filename is accepted when unambiguous (single doc).
    passages = [_hit(1, "a", source=FAKE_PDF), _hit(2, "b", source=FAKE_PDF)]
    text = "From [page 1] and from [page 2]."

    citations = parse_citations(text, passages)

    assert [(c.source_path.name, c.page_number) for c in citations] == [
        ("fake.pdf", 1),
        ("fake.pdf", 2),
    ]


def test_parse_citations_back_compat_refuses_to_guess_in_multi_doc() -> None:
    # `[page 1]` is ambiguous when two retrieved PDFs both have a page 1.
    # The parser refuses rather than guessing.
    passages = [
        _hit(1, "a", source=FAKE_PDF),
        _hit(1, "b", source=SECOND_PDF),
    ]
    text = "Ambiguous [page 1]."

    citations = parse_citations(text, passages)

    assert citations == ()


# ---------- StubAnswerer ----------


def test_stub_answerer_cites_top_passage() -> None:
    stub = StubAnswerer()
    answer = stub.answer(
        "ignored",
        [
            _hit(4, "Nougat parses academic PDFs.", score=0.8, source=FAKE_PDF),
            _hit(1, "unrelated", source=FAKE_PDF),
        ],
    )
    assert isinstance(answer, Answer)
    assert "[fake.pdf, page 4]" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 4
    assert answer.citations[0].source_path.name == "fake.pdf"
    assert answer.citations[0].score == 0.8


def test_stub_answerer_handles_empty_retrieval() -> None:
    stub = StubAnswerer()
    answer = stub.answer("q?", passages=[])
    assert answer.citations == ()
    assert "cannot answer" in answer.text.lower()
