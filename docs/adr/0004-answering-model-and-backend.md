# ADR-0004: Answering model + inference backend

- **Status:** accepted
- **Date:** 2026-04-25
- **Deciders:** Jounes

## Context

Chunking (ADR-0002) and retrieval (ADR-0003) hand off a ranked list of page-level passages. The answering stage turns those passages plus the user's question into a short, grounded answer with page citations.

Forces:
- Target surface is Hugging Face Spaces on free/basic hardware — running a 7B+ LLM locally is infeasible in that footprint.
- A portfolio project should not pay an API bill per demo visitor. Any paid backend must be optional and gated.
- Tests and CI must not depend on a network call or a downloaded model.
- The prompt must make page-level citations trivial to produce and trivial to verify.

## Decision

1. **Default backend:** Hugging Face Inference API. Model choice is a runtime parameter; `Qwen/Qwen2.5-7B-Instruct` is the v1 default. The caller provides a token via `HF_TOKEN` env var.

   **Model-default amendment (2026-04-25):** the original choice was `meta-llama/Llama-3.1-8B-Instruct`. In practice the HF Inference API routes Llama-3.1 through the `novita` provider, which 429s and 504s heavily on free-tier quotas — verified live on the deployed Space. Qwen2.5-7B routes through `together` (a different rate-limit pool) and consistently responds in <1 s. Quality on the project's gold set is at parity for grounded QA — the citation-grounding check from ADR-0007 is doing the heavy lifting on output quality, not the choice between two strong 7-8B instruct models. The model name is still a runtime parameter, so callers who prefer Llama can pass `model="meta-llama/Llama-3.1-8B-Instruct"` to `HFInferenceAnswerer`.
2. **Abstraction:** an `Answerer` protocol with a single method, `answer(question: str, passages: list[RetrievedPassage]) -> Answer`. Concrete implementations live in `paperqa.answering` (or submodules under it).
3. **Fallback for tests and offline runs:** a `StubAnswerer` that returns a deterministic echo of the highest-ranked passage with its page citation. Lives in-package and is the only answerer exercised in CI.
4. **Prompt contract:** the prompt template is built by a pure function `build_prompt(question, passages)` that interleaves passages with `[page N]` markers. The model is instructed to answer only from the provided passages and to cite `[page N]` inline. Same function is reused by every backend.
5. **Answer shape:** `Answer(text, citations)` where `citations: list[Citation]` and `Citation(page_number, score)`. Parsing `[page N]` markers out of the model's output is the responsibility of each `Answerer`, not the caller.
6. **Out of scope for v1:** streaming responses, multi-turn conversation, tool-use / agentic backends, long-context models. All deferred to future ADRs if the product warrants them.

## Alternatives considered

- **Local LLM via `llama.cpp` / transformers.** Rejected as default: too heavy for a free HF Space; model weights bloat the repo or force a lazy download on every cold start. Remains a valid pluggable backend someone can add later without touching the protocol.
- **OpenAI / Anthropic / other paid APIs.** Rejected as default: an API key every visitor pays for is not a portfolio demo, it's a liability. Can be added as an optional backend.
- **Directly calling `transformers.pipeline("text-generation")`.** Rejected: couples the answering layer to a specific runtime and is indistinguishable at the user level from the Inference API path — the HF Inference API wins because it offloads hosting.
- **Baking the prompt template into each backend.** Rejected: the template is the grounding contract. Drifting templates across backends would silently change behaviour. One `build_prompt` shared by all backends keeps the contract auditable.

## Consequences

- **Positive:**
  - CI stays offline and fast — only `StubAnswerer` is exercised.
  - Swapping to a local LLM or a hosted proprietary model is a ~50 LoC change behind the `Answerer` protocol.
  - The `[page N]` citation convention is trivially verifiable: parse the answer, look up each page in the index, render the source passage side-by-side.
- **Negative:**
  - Real answer quality depends on HF Inference API availability and rate limits. Demo UX needs to surface a friendly error when the token is missing or the upstream rate-limits.
  - Citation fidelity is only as good as the model's instruction-following. Evals (future ADR-0005) must check that every `[page N]` marker corresponds to a retrieved passage, not a hallucination.
- **Follow-ups:**
  - ADR-0005: evaluation plan (citation faithfulness, retrieval recall, end-to-end win rate on a handcrafted set).
  - Future ADR: optional local-LLM backend once there is a reason.
  - Future ADR: streaming / multi-turn if the UI needs them.
