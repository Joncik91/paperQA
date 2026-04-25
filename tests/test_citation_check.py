"""Tests for the post-hoc citation grounding check (ADR-0007).

Two headline cases:
  1. Live-demo bug: Llama emits the right BLEU numbers (28.4, 41.8) and
     cites a page that does not contain them. The grounding check must
     strip the marker.
  2. Multi-doc disambiguation (ADR-0008): two PDFs both have a page 9
     containing different numbers. A `[a.pdf, page 9]` citation must be
     verified against a.pdf's page 9 only.
"""

from __future__ import annotations

from pathlib import Path

from paperqa import Passage, RetrievedPassage, parse_citations, verify_citations

FAKE_PDF = Path("fake.pdf")
SECOND_PDF = Path("second.pdf")


def _hit(page: int, text: str, source: Path = FAKE_PDF) -> RetrievedPassage:
    return RetrievedPassage(
        passage=Passage(source_path=source, page_number=page, text=text),
        score=1.0,
    )


# ---------- the headline failure mode ----------


def test_strips_citation_when_cited_page_lacks_the_number() -> None:
    # Reproduces the live-demo bug.
    passages = [
        _hit(
            1,
            "Attention Is All You Need. Ashish Vaswani, Noam Shazeer. "
            "Google Brain. Provided proper attribution is provided ...",
        ),
        _hit(
            8,
            "Table 2. Model BLEU Training Cost EN-DE EN-FR. "
            "Transformer (big) 28.4 41.8 newstest2014 ...",
        ),
    ]
    answer_text = (
        "The Transformer achieves the following BLEU scores: 28.4 on the WMT "
        "2014 English-to-German translation task [fake.pdf, page 1] and 41.8 "
        "on the WMT 2014 English-to-French translation task [fake.pdf, page 1]."
    )

    cleaned = verify_citations(answer_text, passages)
    citations = parse_citations(cleaned, passages)

    assert "[fake.pdf, page 1]" not in cleaned
    assert "citation removed" in cleaned
    assert citations == ()


def test_keeps_citation_when_cited_page_contains_the_number() -> None:
    passages = [
        _hit(
            8,
            "Table 2. Model BLEU EN-DE EN-FR. Transformer (big) 28.4 41.8 newstest2014 ...",
        ),
    ]
    answer_text = (
        "The Transformer achieves 28.4 BLEU on English-to-German "
        "[fake.pdf, page 8] and 41.8 BLEU on English-to-French "
        "[fake.pdf, page 8]."
    )

    cleaned = verify_citations(answer_text, passages)
    citations = parse_citations(cleaned, passages)

    assert cleaned.count("[fake.pdf, page 8]") == 2
    assert "citation removed" not in cleaned
    assert [c.page_number for c in citations] == [8, 8]


# ---------- multi-doc disambiguation (ADR-0008) ----------


def test_multidoc_routes_grounding_check_to_correct_pdf() -> None:
    # Both PDFs have a page 9; the number "28.4" lives only on second.pdf's
    # page 9. A citation to fake.pdf's page 9 must be stripped; a citation
    # to second.pdf's page 9 must survive.
    passages = [
        _hit(9, "Adam optimizer learning rate decay schedule.", source=FAKE_PDF),
        _hit(9, "Reported BLEU 28.4 on newstest2014.", source=SECOND_PDF),
    ]
    answer_text_wrong = "Score was 28.4 [fake.pdf, page 9]."
    answer_text_right = "Score was 28.4 [second.pdf, page 9]."

    cleaned_wrong = verify_citations(answer_text_wrong, passages)
    cleaned_right = verify_citations(answer_text_right, passages)

    assert "[fake.pdf, page 9]" not in cleaned_wrong
    assert "citation removed" in cleaned_wrong
    assert "[second.pdf, page 9]" in cleaned_right
    assert "citation removed" not in cleaned_right


def test_back_compat_page_only_marker_skipped_when_ambiguous() -> None:
    # Two PDFs both have a page 1; `[page 1]` cannot be resolved to one
    # passage. The grounding check leaves the marker for parse_citations
    # to drop. No annotation either way.
    passages = [
        _hit(1, "Adam optimizer.", source=FAKE_PDF),
        _hit(1, "Reported BLEU 28.4.", source=SECOND_PDF),
    ]
    answer_text = "Score was 28.4 [page 1]."

    cleaned = verify_citations(answer_text, passages)

    assert "[page 1]" in cleaned
    assert "citation removed" not in cleaned
    # parse_citations refuses ambiguous back-compat markers too.
    assert parse_citations(cleaned, passages) == ()


# ---------- numerical-anchor rule ----------


def test_partial_number_match_still_strips() -> None:
    passages = [_hit(5, "Model achieves 91.3 F1 on the dev set.")]
    answer_text = "Scores were 91.3 and 88.4 [fake.pdf, page 5]."

    cleaned = verify_citations(answer_text, passages)

    assert "[fake.pdf, page 5]" not in cleaned
    assert "citation removed" in cleaned


def test_decimal_format_must_match_exactly() -> None:
    passages = [_hit(1, "Reported 28.4 BLEU score.")]
    answer_text = "The score was 28 [fake.pdf, page 1]."

    cleaned = verify_citations(answer_text, passages)
    assert "[fake.pdf, page 1]" not in cleaned


# ---------- prose sentences without numbers always pass ----------


def test_prose_sentence_without_numbers_passes() -> None:
    passages = [_hit(3, "Adam optimizer learning rate batch size hyperparameter")]
    answer_text = "The paper proposes a new visualization technique [fake.pdf, page 3]."

    cleaned = verify_citations(answer_text, passages)
    assert "[fake.pdf, page 3]" in cleaned


# ---------- edge cases ----------


def test_empty_answer_returns_empty_string() -> None:
    assert verify_citations("", [_hit(1, "x")]) == ""


def test_text_without_citations_passes_through_unchanged() -> None:
    text = "The model is interesting. It does many things."
    assert verify_citations(text, [_hit(1, "irrelevant")]) == text


def test_citation_to_unretrieved_page_is_left_for_parse_citations_to_drop() -> None:
    passages = [_hit(1, "alpha beta gamma delta")]
    answer_text = "Some claim [fake.pdf, page 99]."

    cleaned = verify_citations(answer_text, passages)
    assert "[fake.pdf, page 99]" in cleaned
    assert parse_citations(cleaned, passages) == ()


def test_multiple_sentences_only_numeric_one_is_stripped() -> None:
    passages = [
        _hit(
            1,
            "Attention Is All You Need. Vaswani et al. Google Brain. Provided proper attribution.",
        ),
        _hit(8, "Transformer 28.4 41.8 BLEU EN-DE EN-FR newstest2014"),
    ]
    answer_text = (
        "The paper introduces the Transformer [fake.pdf, page 1]. "
        "It achieves 28.4 BLEU on English-to-German [fake.pdf, page 1]."
    )

    cleaned = verify_citations(answer_text, passages)

    assert cleaned.count("[fake.pdf, page 1]") == 1
    assert "citation removed" in cleaned
