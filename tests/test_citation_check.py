"""Tests for the post-hoc citation grounding check (ADR-0007).

The headline test is the BLEU-on-page-1 case observed on the live demo:
Llama emits the right numbers (28.4, 41.8) and cites page 1, but those
numbers actually live on page 8. The grounding check must strip the
[page 1] marker and leave the numbers in place.
"""

from __future__ import annotations

from pathlib import Path

from paperqa import Passage, RetrievedPassage, parse_citations, verify_citations

FAKE_PDF = Path("fake.pdf")


def _hit(page: int, text: str) -> RetrievedPassage:
    return RetrievedPassage(
        passage=Passage(source_path=FAKE_PDF, page_number=page, text=text),
        score=1.0,
    )


# ---------- the headline failure mode ----------


def test_strips_citation_when_cited_page_lacks_the_number() -> None:
    # Reproduces the live-demo bug: Llama cites page 1 for BLEU numbers
    # that actually live on page 8.
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
        "2014 English-to-German translation task [page 1] and 41.8 on the "
        "WMT 2014 English-to-French translation task [page 1]."
    )

    cleaned = verify_citations(answer_text, passages)
    citations = parse_citations(cleaned, passages)

    assert "[page 1]" not in cleaned
    assert "citation removed" in cleaned
    assert citations == ()


def test_keeps_citation_when_cited_page_contains_the_number() -> None:
    # The same answer cited correctly: page 8 has both numbers, so both
    # markers must survive.
    passages = [
        _hit(
            8,
            "Table 2. Model BLEU EN-DE EN-FR. Transformer (big) 28.4 41.8 newstest2014 ...",
        ),
    ]
    answer_text = (
        "The Transformer achieves 28.4 BLEU on English-to-German [page 8] "
        "and 41.8 BLEU on English-to-French [page 8]."
    )

    cleaned = verify_citations(answer_text, passages)
    citations = parse_citations(cleaned, passages)

    assert cleaned.count("[page 8]") == 2
    assert "citation removed" not in cleaned
    assert [c.page_number for c in citations] == [8, 8]


# ---------- numerical-anchor rule ----------


def test_partial_number_match_still_strips() -> None:
    # Sentence claims TWO numbers; cited page has only one of them.
    # The numerical-anchor rule requires every number, so strip.
    passages = [_hit(5, "Model achieves 91.3 F1 on the dev set.")]
    answer_text = "Scores were 91.3 and 88.4 [page 5]."

    cleaned = verify_citations(answer_text, passages)

    assert "[page 5]" not in cleaned
    assert "citation removed" in cleaned


def test_decimal_format_must_match_exactly() -> None:
    # "28.4" on the cited page must match a sentence's "28.4", not "28".
    passages = [_hit(1, "Reported 28.4 BLEU score.")]
    answer_text = "The score was 28 [page 1]."

    cleaned = verify_citations(answer_text, passages)
    assert "[page 1]" not in cleaned


# ---------- prose sentences without numbers always pass ----------


def test_prose_sentence_without_numbers_passes() -> None:
    # No numbers in the sentence -> the numerical-anchor rule does not
    # fire and the citation is kept regardless of lexical overlap. The
    # lexical rule was tried in an earlier iteration but stripped real
    # citations on Llama's paraphrased prose; the headline trade is
    # documented in baseline 2026-04-25-grounding-check.
    passages = [_hit(3, "Adam optimizer learning rate batch size hyperparameter")]
    answer_text = "The paper proposes a new visualization technique [page 3]."

    cleaned = verify_citations(answer_text, passages)
    assert "[page 3]" in cleaned


# ---------- edge cases ----------


def test_empty_answer_returns_empty_string() -> None:
    assert verify_citations("", [_hit(1, "x")]) == ""


def test_text_without_citations_passes_through_unchanged() -> None:
    text = "The model is interesting. It does many things."
    assert verify_citations(text, [_hit(1, "irrelevant")]) == text


def test_citation_to_unretrieved_page_is_left_for_parse_citations_to_drop() -> None:
    # We do not annotate "removed" here because parse_citations already
    # filters citations to retrieved pages — annotating twice would be
    # noise.
    passages = [_hit(1, "alpha beta gamma delta")]
    answer_text = "Some claim [page 99]."

    cleaned = verify_citations(answer_text, passages)
    # Marker preserved by the grounding check; parse_citations drops it.
    assert "[page 99]" in cleaned
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
        "The paper introduces the Transformer [page 1]. "
        "It achieves 28.4 BLEU on English-to-German [page 1]."
    )

    cleaned = verify_citations(answer_text, passages)

    # Sentence 1: no numbers -> citation kept (numerical rule does not fire).
    # Sentence 2: "28.4" not on page 1 -> citation stripped.
    assert cleaned.count("[page 1]") == 1
    assert "citation removed" in cleaned
