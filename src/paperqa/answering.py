"""Answering — turn a question + retrieved passages into a cited answer.

WHAT: Defines the `Answerer` protocol, the `Answer` / `Citation` value
      objects, the shared `build_prompt(question, passages)` function, and
      an offline `StubAnswerer` used in tests and for demos without an
      inference token.
WHY:  The prompt template is the grounding contract (ADR-0004) — every
      backend must build the prompt the same way, so it lives here and is
      shared. Parsing `[page N]` markers out of a generated answer is a
      pure function and is also shared. Only the network call differs
      between backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from paperqa.indexing import RetrievedPassage

__all__ = [
    "Answer",
    "Answerer",
    "Citation",
    "StubAnswerer",
    "build_prompt",
    "parse_citations",
]


# WHY `[page N]`: terse, unambiguous, trivially regex-parseable, and matches
# the grain of ADR-0002 (page-level passages). Anything more elaborate
# (e.g. JSON blocks) regresses the model's instruction-following.
PAGE_MARKER_RE = re.compile(r"\[page\s+(\d+)\]", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Citation:
    """A page reference extracted from a generated answer."""

    page_number: int
    # Retrieval score of the source passage at query time. Lets the UI show
    # confidence and lets evals flag low-score citations as likely junk.
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """Final answer payload returned by any `Answerer`."""

    text: str
    citations: tuple[Citation, ...]


class Answerer(Protocol):
    """Anything that can turn a question + retrieved passages into an Answer."""

    def answer(self, question: str, passages: list[RetrievedPassage]) -> Answer: ...


# WHY externalised as a constant: the system instruction is part of the
# grounding contract. Any change to it should be visible in diffs and show
# up in eval deltas (future ADR-0005).
SYSTEM_INSTRUCTION = (
    "You are a careful research assistant. Answer the user's question using "
    "ONLY the passages provided. Every factual claim in your answer must be "
    "followed by a citation in the form [page N] referring to the page of "
    "the passage you used. If the passages do not contain the answer, say "
    "so plainly and do not guess."
)


def build_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    """Assemble the single-string prompt handed to any backend.

    The structure is: system instruction, then the passages each prefixed
    with a `[page N]` header, then the question, then an ANSWER cue.

    WHY a plain string, not a message list: not every backend exposes a
    chat interface, but every backend accepts a prompt string. Keeping the
    contract stringly-typed is the LCD.
    """
    passage_block = "\n\n".join(
        f"[page {hit.passage.page_number}]\n{hit.passage.text.strip()}" for hit in passages
    )
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"PASSAGES:\n{passage_block}\n\n"
        f"QUESTION: {question.strip()}\n\n"
        f"ANSWER:"
    )


def parse_citations(
    answer_text: str,
    passages: list[RetrievedPassage],
) -> tuple[Citation, ...]:
    """Extract `[page N]` citations from `answer_text`, in order of appearance.

    Only pages that were actually in the retrieved set become `Citation`s —
    this is the first line of defence against hallucinated page numbers.
    Duplicates are preserved because the order and frequency can be useful
    for UI highlighting.
    """
    score_by_page = {hit.passage.page_number: hit.score for hit in passages}
    citations = []
    for match in PAGE_MARKER_RE.finditer(answer_text):
        page = int(match.group(1))
        if page in score_by_page:
            citations.append(Citation(page_number=page, score=score_by_page[page]))
    return tuple(citations)


class StubAnswerer:
    """Deterministic, offline `Answerer` used in tests and as a demo fallback.

    WHY this exists in-package: CI must not call the network, and the public
    demo must still render something sensible when no HF token is configured.
    The stub returns a canned answer that cites the top-ranked page and
    echoes a short excerpt of that page's text.
    """

    def answer(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        if not passages:
            return Answer(
                text="No passages were retrieved, so I cannot answer.",
                citations=(),
            )
        top = passages[0]
        excerpt = top.passage.text.strip().replace("\n", " ")[:200]
        text = f"(stub) Based on the top-ranked passage: {excerpt} [page {top.passage.page_number}]"
        return Answer(text=text, citations=parse_citations(text, passages))
