"""Answerer backed by the Hugging Face Inference API.

WHAT: `HFInferenceAnswerer` satisfies the `Answerer` protocol (ADR-0004)
      by calling `huggingface_hub.InferenceClient.chat_completion` with
      the prompt assembled by `paperqa.answering.build_prompt`. Citations
      are parsed back out of the generated text with `parse_citations`.
WHY:  Kept out of the core package so that importing paperqa does not pull
      the Inference SDK, and so the HF token requirement is scoped to
      users who actually want a real LLM answer. Tests for this module are
      integration-marked and skipped unless HF_TOKEN is set.

History note: an earlier draft used `client.text_generation`. That code
path was removed by HF's serverless providers in 2026 — instruct models
are reachable only through `chat_completion`. See ADR-0004.
"""

from __future__ import annotations

import os

from paperqa.answering import Answer, build_prompt, parse_citations
from paperqa.indexing import RetrievedPassage

__all__ = ["DEFAULT_MODEL", "HFInferenceAnswerer"]


# v1 default model. Swapping it is a runtime parameter; if the default
# itself changes, bump ADR-0004 in the same commit because the choice is
# part of the reproducibility contract.
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Generation budget per answer. Keeps the UX snappy on free-tier inference
# and caps runaway outputs. Tune only with a concrete reason.
DEFAULT_MAX_TOKENS = 512


class HFInferenceAnswerer:
    """Call the HF Inference API to produce an answer grounded in passages."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        # WHY env fallback: the HF convention; lets a Space or a shell
        # export the token once and forget about it.
        self._model = model
        self._token = token or os.environ.get("HF_TOKEN")
        self._max_tokens = max_tokens

    def answer(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        if not passages:
            return Answer(
                text="No passages were retrieved, so I cannot answer.",
                citations=(),
            )
        prompt = build_prompt(question, passages)
        text = self._generate(prompt)
        return Answer(text=text, citations=parse_citations(text, passages))

    def _generate(self, prompt: str) -> str:
        # Lazy import: keeps `huggingface_hub` an optional extra. See
        # pyproject.toml `[project.optional-dependencies].llm`.
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self._model, token=self._token)
        # WHY a single user message: build_prompt already assembles the
        # system instruction + passages + question. Splitting it into
        # role messages here would duplicate that contract and risk drift
        # from the StubAnswerer / measurement-harness path.
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content
        return str(content) if content is not None else ""
