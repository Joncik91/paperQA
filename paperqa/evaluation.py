"""Offline measurement: retrieval recall@k + citation faithfulness.

WHAT: Loads a JSON gold set, runs each question through a `PaperQA`
      instance, and scores two metrics per question:
        - recall_at_k: fraction of gold-relevant pages present in top-k
          retrieved passages.
        - citation_faithfulness: fraction of emitted citations whose page
          is in the gold-relevant set.
      A `run_report` function aggregates per-question scores into an
      `EvalReport`.
WHY:  ADR-0005 locks these two metrics as v1's promises-to-reader. Keeping
      the harness a pure function over a `PaperQA` instance means it runs
      on exactly the same code path as the live app — no bespoke test
      mode that could drift from production.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from paperqa.pipeline import PaperQA

__all__ = [
    "EvalReport",
    "GoldItem",
    "QuestionScore",
    "load_gold_set",
    "recall_at_k",
    "run_report",
    "score_citations",
]


@dataclass(frozen=True, slots=True)
class GoldItem:
    """One labelled question against a single source PDF."""

    pdf_path: Path
    question: str
    relevant_pages: frozenset[int]
    must_cite_page: int


@dataclass(frozen=True, slots=True)
class QuestionScore:
    """Per-question metrics emitted by the harness."""

    question: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    citation_faithfulness: float
    must_cite_hit: bool


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate summary across a gold set."""

    n_questions: int
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_citation_faithfulness: float
    must_cite_rate: float
    per_question: tuple[QuestionScore, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        # per-question rows are dataclasses; asdict already recursed.
        return json.dumps(payload, indent=2)


def load_gold_set(path: str | Path) -> list[GoldItem]:
    """Parse a gold-set JSON file into `GoldItem`s.

    JSON schema (list of objects):
        [
          {
            "pdf_path": "tests/fixtures/three_pages.pdf",
            "question": "...",
            "relevant_pages": [1, 2],
            "must_cite_page": 1
          },
          ...
        ]

    Relative `pdf_path` values are resolved against the gold file's parent
    directory so the set is portable with its fixtures.
    """
    gold_path = Path(path).resolve()
    raw = json.loads(gold_path.read_text())
    base_dir = gold_path.parent
    items = []
    for row in raw:
        pdf_path = Path(row["pdf_path"])
        if not pdf_path.is_absolute():
            pdf_path = (base_dir / pdf_path).resolve()
        items.append(
            GoldItem(
                pdf_path=pdf_path,
                question=row["question"],
                relevant_pages=frozenset(row["relevant_pages"]),
                must_cite_page=row["must_cite_page"],
            )
        )
    return items


def recall_at_k(retrieved_pages: list[int], relevant: frozenset[int], k: int) -> float:
    """Share of `relevant` pages that appear in the first `k` of `retrieved_pages`.

    Returns 0.0 when `relevant` is empty — defensive: a malformed gold entry
    should not divide-by-zero the whole run.
    """
    if not relevant:
        return 0.0
    top = retrieved_pages[:k]
    hit = sum(1 for page in relevant if page in top)
    return hit / len(relevant)


def score_citations(cited_pages: list[int], relevant: frozenset[int]) -> float:
    """Fraction of emitted citations whose page is in the gold-relevant set.

    A stricter filter than `parse_citations`, which only drops pages absent
    from the *retrieved* set. Here we drop pages absent from the *gold*
    set — catching the "cited a retrieved page that wasn't actually on-topic"
    failure mode.

    Returns 1.0 for an empty citation list because there is nothing unfaithful.
    An answer with no citations will still be flagged separately via the
    `must_cite_hit` field.
    """
    if not cited_pages:
        return 1.0
    hit = sum(1 for page in cited_pages if page in relevant)
    return hit / len(cited_pages)


def run_report(qa: PaperQA, gold: list[GoldItem]) -> EvalReport:
    """Run every gold item through `qa` and aggregate the metrics."""
    scores: list[QuestionScore] = []
    for item in gold:
        result = qa.ask(item.pdf_path, item.question)
        retrieved_pages = [h.passage.page_number for h in result.retrieved]
        cited_pages = [c.page_number for c in result.answer.citations]
        scores.append(
            QuestionScore(
                question=item.question,
                recall_at_1=recall_at_k(retrieved_pages, item.relevant_pages, 1),
                recall_at_3=recall_at_k(retrieved_pages, item.relevant_pages, 3),
                recall_at_5=recall_at_k(retrieved_pages, item.relevant_pages, 5),
                citation_faithfulness=score_citations(cited_pages, item.relevant_pages),
                must_cite_hit=item.must_cite_page in cited_pages,
            )
        )
    n = len(scores)
    if n == 0:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, ())
    return EvalReport(
        n_questions=n,
        mean_recall_at_1=sum(s.recall_at_1 for s in scores) / n,
        mean_recall_at_3=sum(s.recall_at_3 for s in scores) / n,
        mean_recall_at_5=sum(s.recall_at_5 for s in scores) / n,
        mean_citation_faithfulness=sum(s.citation_faithfulness for s in scores) / n,
        must_cite_rate=sum(1 for s in scores if s.must_cite_hit) / n,
        per_question=tuple(scores),
    )
