"""Post-hoc citation grounding check.

WHAT: `verify_citations(answer_text, retrieved_passages)` reads each
      `[page N]` marker, checks that the surrounding sentence's
      distinctive content tokens (numbers + multi-character words) actually
      appear in page N's passage text, and strips the marker if not.
WHY:  Sharpening the LLM's system prompt (ADR-0004) caps out at "the
      model tries"; for tabular questions the model reliably emits the
      correct *number* but anchors the *citation* on a topically-similar
      page that does not contain the number. ADR-0007 makes the
      grounding check deterministic and post-hoc so a fact's distinctive
      tokens — especially numbers — must show up on the cited page or
      the citation is dropped.
"""

from __future__ import annotations

import re

from paperqa.indexing import RetrievedPassage

__all__ = ["verify_citations"]


# Citation marker pattern. Same shape as paperqa.answering.PAGE_MARKER_RE
# but referenced separately so this module does not depend on the
# answering package's internals.
_MARKER_RE = re.compile(r"\[page\s+(\d+)\]", flags=re.IGNORECASE)

# Splits an answer into sentences on `.`, `!`, `?` followed by whitespace
# or end-of-string. Crude on purpose — the citation marker is the unit we
# care about, not exact NLP-grade sentencing. Markers that span sentences
# are rare; if they happen, the surrounding sentence dominates the check.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Number tokens: integers, decimals, percentages. Preserves formatting so
# "28.4" matches "28.4" but not "28".
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def _numbers(text: str) -> set[str]:
    """Set of number tokens appearing in `text`."""
    return set(_NUMBER_RE.findall(text))


def _is_supported(sentence: str, page_text: str) -> bool:
    """True if `sentence` is grounded in `page_text` per ADR-0007.

    Only the **numerical-anchor** rule fires: if the sentence contains
    any number, every such number must appear on the cited page.
    Sentences without numbers always pass — the lexical-overlap rule
    was tried in an earlier iteration but stripped real citations on
    Llama's paraphrased prose (see baseline 2026-04-25-grounding-check).
    """
    sentence_no_markers = _MARKER_RE.sub(" ", sentence)
    sent_nums = _numbers(sentence_no_markers)
    if not sent_nums:
        return True
    return sent_nums.issubset(_numbers(page_text))


def verify_citations(
    answer_text: str,
    retrieved_passages: list[RetrievedPassage],
) -> str:
    """Strip unsupported `[page N]` markers from `answer_text`.

    For every removed marker, the sentence is annotated with
    `_(citation removed: page N does not contain this fact)_` so the
    user sees the failure rather than silently losing the source.
    Returns the cleaned text; pass it through `parse_citations` to
    rebuild the structured `Citation` list.
    """
    text_by_page = {hit.passage.page_number: hit.passage.text for hit in retrieved_passages}
    sentences = _SENTENCE_SPLIT_RE.split(answer_text)
    cleaned: list[str] = []
    for sentence in sentences:
        if not _MARKER_RE.search(sentence):
            cleaned.append(sentence)
            continue
        cleaned.append(_clean_sentence(sentence, text_by_page))
    return " ".join(s for s in cleaned if s)


def _clean_sentence(sentence: str, text_by_page: dict[int, str]) -> str:
    """Drop unsupported `[page N]` markers from a single sentence."""
    removed_pages: list[int] = []

    def replace(match: re.Match[str]) -> str:
        page = int(match.group(1))
        page_text = text_by_page.get(page)
        if page_text is None:
            # Cited a page that was not retrieved — let parse_citations
            # filter it later. We do not annotate here because the
            # earlier filter already explains it.
            return match.group(0)
        if _is_supported(sentence, page_text):
            return match.group(0)
        removed_pages.append(page)
        # Strip the marker (and any leading whitespace just before it,
        # to avoid double spaces).
        return ""

    cleaned = _MARKER_RE.sub(replace, sentence)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if removed_pages:
        notes = ", ".join(f"page {p} does not contain this fact" for p in removed_pages)
        cleaned = f"{cleaned} _(citation removed: {notes})_"
    return cleaned
