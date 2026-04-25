"""Offline measurement CLI.

WHY: ADR-0005 promises a one-command report. This script is that command.
It builds a PaperQA instance with the real MiniLM embedder and whichever
Answerer is available (HFInferenceAnswerer if HF_TOKEN is set, else the
offline StubAnswerer), runs the default gold set, and prints a table plus
the full report JSON to stdout.

Usage:
    python scripts/run_eval.py                          # default gold set
    python scripts/run_eval.py path/to/gold.json        # custom set
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from paperqa import PaperQA, StubAnswerer
from paperqa.answering import Answerer
from paperqa.embedders import SentenceTransformerEmbedder
from paperqa.evaluation import EvalReport, load_gold_set, run_report

DEFAULT_GOLD = Path(__file__).resolve().parent.parent / "tests" / "eval" / "gold.json"


def _pick_answerer() -> Answerer:
    """Same rule as the Gradio app: real backend if a token is available."""
    if os.environ.get("HF_TOKEN"):
        from paperqa.backends.hf_inference import HFInferenceAnswerer

        return HFInferenceAnswerer()
    return StubAnswerer()


def _print_table(report: EvalReport) -> None:
    print(f"Questions: {report.n_questions}")
    print(f"mean recall@1            : {report.mean_recall_at_1:.3f}")
    print(f"mean recall@3            : {report.mean_recall_at_3:.3f}")
    print(f"mean recall@5            : {report.mean_recall_at_5:.3f}")
    print(f"mean citation faithfulness: {report.mean_citation_faithfulness:.3f}")
    print(f"must-cite rate           : {report.must_cite_rate:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paperQA offline metrics.")
    parser.add_argument(
        "gold",
        nargs="?",
        default=str(DEFAULT_GOLD),
        help=f"Path to a gold-set JSON file (default: {DEFAULT_GOLD})",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the JSON report; omit the human-readable table.",
    )
    args = parser.parse_args(argv)

    qa = PaperQA.with_embedder(SentenceTransformerEmbedder(), answerer=_pick_answerer())
    report = run_report(qa, load_gold_set(args.gold))

    if not args.json_only:
        _print_table(report)
        print()
    print(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
