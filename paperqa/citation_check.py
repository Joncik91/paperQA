"""Post-hoc citation grounding check.

WHAT: `verify_citations(answer_text, retrieved_passages)` reads each
      `[file.pdf, page N]` (or back-compat `[page N]`) marker, checks
      that the surrounding sentence's distinctive content tokens
      (numbers) actually appear in the cited passage's text, and strips
      the marker if not.
WHY:  Sharpening the LLM's system prompt (ADR-0004) caps out at "the
      model tries"; for tabular questions the model reliably emits the
      correct *number* but anchors the *citation* on a topically-similar
      passage that does not contain the number. ADR-0007 makes the
      grounding check deterministic and post-hoc so a fact's distinctive
      tokens — especially numbers — must show up on the cited passage
      or the citation is dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

from paperqa.indexing import RetrievedPassage

__all__ = ["verify_citations"]


# Citation marker pattern. MUST stay in sync with paperqa.answering.PAGE_MARKER_RE
# — this module redeclares it (rather than importing) so the grounding-check
# layer has no reverse dependency on the answering layer's internals.
# Accepts:
#   `[file.pdf, page 3]`  — multi-doc preferred form (filename + page)
#   `[page 3]`            — back-compat single-doc shape, still recognised
_MARKER_RE = re.compile(
    r"\[(?:(?P<file>[^,\[\]]+?)\s*,\s*)?page\s+(?P<page>\d+)\]",
    flags=re.IGNORECASE,
)

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
    """Strip unsupported citation markers from `answer_text`.

    For every removed marker, the sentence is annotated with
    `_(citation removed: <file>, page N does not contain this fact)_`
    so the user sees the failure rather than silently losing the source.
    Returns the cleaned text; pass it through `parse_citations` to
    rebuild the structured `Citation` list.
    """
    # Lookup keyed by (filename, page). Filename is the source path's basename
    # so the lookup matches what the LLM emits in `[file.pdf, page N]`.
    text_by_key: dict[tuple[str, int], str] = {}
    text_by_page_only: dict[int, list[tuple[str, str]]] = {}
    for hit in retrieved_passages:
        path = hit.passage.source_path
        page = hit.passage.page_number
        text = hit.passage.text
        text_by_key[(path.name, page)] = text
        text_by_page_only.setdefault(page, []).append((path.name, text))

    sentences = _SENTENCE_SPLIT_RE.split(answer_text)
    cleaned: list[str] = []
    for sentence in sentences:
        if not _MARKER_RE.search(sentence):
            cleaned.append(sentence)
            continue
        cleaned.append(_clean_sentence(sentence, text_by_key, text_by_page_only))
    return " ".join(s for s in cleaned if s)


def _clean_sentence(
    sentence: str,
    text_by_key: dict[tuple[str, int], str],
    text_by_page_only: dict[int, list[tuple[str, str]]],
) -> str:
    """Drop unsupported citation markers from a single sentence.

    Resolves the cited passage's text by `(filename, page)` for the
    multi-doc form and by page-only fallback for the back-compat form
    (back-compat is accepted only when unambiguous, mirroring
    parse_citations).
    """
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        page = int(match.group("page"))
        file_token = match.group("file")
        page_text: str | None
        label: str
        if file_token:
            filename = Path(file_token).name
            label = f"{filename}, page {page}"
            page_text = text_by_key.get((filename, page))
        else:
            # Back-compat `[page N]` — only resolvable when exactly one
            # retrieved passage has that page. parse_citations applies the
            # same "refuse to guess" rule, so this stays consistent.
            candidates = text_by_page_only.get(page, [])
            if len(candidates) == 1:
                filename, page_text = candidates[0]
                label = f"{filename}, page {page}"
            else:
                # Ambiguous or unretrieved — leave the marker for
                # parse_citations to drop later. No annotation here.
                return match.group(0)

        if page_text is None:
            # Cited a (file, page) pair that was not retrieved — let
            # parse_citations drop it. Annotating here would be noise.
            return match.group(0)
        if _is_supported(sentence, page_text):
            return match.group(0)
        removed.append(label)
        return ""

    cleaned = _MARKER_RE.sub(replace, sentence)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if removed:
        notes = ", ".join(f"{r} does not contain this fact" for r in removed)
        cleaned = f"{cleaned} _(citation removed: {notes})_"
    return cleaned
