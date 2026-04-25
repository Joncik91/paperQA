"""Answering — turn a question + retrieved passages into a cited answer.

WHAT: Defines the `Answerer` protocol, the `Answer` / `Citation` value
      objects, the shared `build_prompt(question, passages)` function, and
      an offline `StubAnswerer` used in tests and for demos without an
      inference token.
WHY:  The prompt template is the grounding contract (ADR-0004) — every
      backend must build the prompt the same way, so it lives here and is
      shared. Parsing `[file, page N]` markers out of a generated answer
      is a pure function and is also shared. Only the network call differs
      between backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
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


# Citation marker shapes accepted in answers (ADR-0008):
#   `[file.pdf, page 3]`  — multi-doc preferred form (filename + page)
#   `[page 3]`            — back-compat single-doc shape, still recognised
# Both are emitted by `build_prompt` (multi-doc) and may appear in legacy
# stub answers. The named groups make the parser readable.
PAGE_MARKER_RE = re.compile(
    r"\[(?:(?P<file>[^,\[\]]+?)\s*,\s*)?page\s+(?P<page>\d+)\]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Citation:
    """A page reference extracted from a generated answer."""

    source_path: Path
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
# up in measurement deltas.
SYSTEM_INSTRUCTION = (
    "You are a careful research assistant. Answer the user's question using "
    "ONLY the passages provided. Multiple source documents may appear; each "
    "passage's header tells you which file and page it came from.\n\n"
    "Citation rules — these are strict:\n"
    "1. After every factual claim, add a citation in the form "
    "[<filename>, page N] using the exact filename and page number from "
    "the passage's header.\n"
    "2. Before citing, silently verify the fact appears in that passage's "
    "text. If it does not, do NOT cite that passage — even if it was "
    "retrieved.\n"
    "3. Never cite a passage you did not use. Passages appearing in the "
    "retrieved set but not used for any claim must not appear in the "
    "answer at all.\n"
    "4. If two facts come from different passages, cite each one separately.\n"
    "5. If the passages do not contain the answer, say so plainly and emit "
    "no citations. Do not guess."
)


def build_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    """Assemble the single-string prompt handed to any backend.

    The structure is: system instruction, then the passages each prefixed
    with a `[<filename>, page N]` header, then the question, then an
    ANSWER cue.

    WHY a plain string, not a message list: not every backend exposes a
    chat interface, but every backend accepts a prompt string. Keeping the
    contract stringly-typed is the LCD.
    """
    passage_block = "\n\n".join(
        f"[{hit.passage.source_path.name}, page {hit.passage.page_number}]\n"
        f"{hit.passage.text.strip()}"
        for hit in passages
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
    """Extract citations from `answer_text`, in order of appearance.

    Recognises both `[file.pdf, page N]` (multi-doc) and `[page N]`
    (single-doc back-compat). Only citations whose `(source_path,
    page_number)` pair was in the retrieved set become `Citation`s — the
    first line of defence against hallucinated references. Duplicates are
    preserved because order and frequency can be useful for UI highlighting.

    For back-compat `[page N]` markers (no filename), the citation is
    accepted only if the page number is unambiguous across the retrieved
    set — i.e. exactly one retrieved passage has that page number. This
    keeps single-doc behaviour identical and refuses to guess in multi-doc.
    """
    by_filename: dict[tuple[str, int], tuple[Path, float]] = {}
    by_page_only: dict[int, list[tuple[Path, float]]] = {}
    for hit in passages:
        path = hit.passage.source_path
        page = hit.passage.page_number
        by_filename[(path.name, page)] = (path, hit.score)
        by_page_only.setdefault(page, []).append((path, hit.score))

    citations: list[Citation] = []
    for match in PAGE_MARKER_RE.finditer(answer_text):
        page = int(match.group("page"))
        file_token = match.group("file")
        if file_token:
            key = (Path(file_token).name, page)
            if key in by_filename:
                src, score = by_filename[key]
                citations.append(Citation(source_path=src, page_number=page, score=score))
            continue
        # Filename-less `[page N]`: accept only when unambiguous.
        candidates = by_page_only.get(page, [])
        if len(candidates) == 1:
            src, score = candidates[0]
            citations.append(Citation(source_path=src, page_number=page, score=score))
    return tuple(citations)


class StubAnswerer:
    """Deterministic, offline `Answerer` used in tests and as a demo fallback.

    WHY this exists in-package: CI must not call the network, and the public
    demo must still render something sensible when no HF token is configured.
    The stub returns a canned answer that cites the top-ranked passage and
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
        marker = f"[{top.passage.source_path.name}, page {top.passage.page_number}]"
        text = f"(stub) Based on the top-ranked passage: {excerpt} {marker}"
        return Answer(text=text, citations=parse_citations(text, passages))
