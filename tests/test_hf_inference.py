"""Tests for the HF Inference API answerer.

Two layers:
  1. Offline unit tests using monkeypatched network — fast, always run.
  2. Integration test that hits the real Inference API — skipped unless
     HF_TOKEN is set. Marked `integration` and excluded from default CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from paperqa import Passage, RetrievedPassage
from paperqa.backends.hf_inference import HFInferenceAnswerer

FAKE_PDF = Path("fake.pdf")


def _hit(page: int, text: str) -> RetrievedPassage:
    return RetrievedPassage(
        passage=Passage(source_path=FAKE_PDF, page_number=page, text=text),
        score=1.0,
    )


def test_empty_passages_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    # No network call should happen when there are no passages.
    called = {"n": 0}

    def fake_generate(self: HFInferenceAnswerer, prompt: str) -> str:
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr(HFInferenceAnswerer, "_generate", fake_generate)

    answer = HFInferenceAnswerer(token="fake").answer("q?", passages=[])

    assert called["n"] == 0
    assert answer.citations == ()
    assert "cannot answer" in answer.text.lower()


def test_parses_citations_from_generated_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(self: HFInferenceAnswerer, prompt: str) -> str:
        # Real prompt is passed in; return a stock answer that cites a valid
        # page and a fake page to prove the hallucination filter works.
        return "The method is attention [page 1]. Ignore [page 99]."

    monkeypatch.setattr(HFInferenceAnswerer, "_generate", fake_generate)

    passages = [_hit(1, "Transformer architecture."), _hit(2, "Training details.")]
    answerer = HFInferenceAnswerer(token="fake")

    answer = answerer.answer("What is the method?", passages)

    assert "[page 1]" in answer.text
    assert [c.page_number for c in answer.citations] == [1]


def test_token_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # WHY: matches the HF SDK convention; a Space typically exports HF_TOKEN.
    monkeypatch.setenv("HF_TOKEN", "from-env")
    answerer = HFInferenceAnswerer()
    assert answerer._token == "from-env"


def test_explicit_token_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    answerer = HFInferenceAnswerer(token="explicit")
    assert answerer._token == "explicit"


@pytest.mark.integration
def test_real_inference_call_if_token_available() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not set; integration test skipped")
    pytest.importorskip("huggingface_hub")

    passages = [
        _hit(1, "The paper proposes a new attention mechanism called flash attention."),
        _hit(2, "Training used 8 A100 GPUs for two days."),
    ]
    answerer = HFInferenceAnswerer(token=token, max_new_tokens=64)
    answer = answerer.answer("What does the paper propose?", passages)

    # Minimal smoke assertions — the real API is not deterministic.
    assert isinstance(answer.text, str)
    assert len(answer.text) > 0
    # Citation page numbers must be within the retrieved set.
    retrieved_pages = {h.passage.page_number for h in passages}
    assert all(c.page_number in retrieved_pages for c in answer.citations)
