# ADR-0005: Evaluation plan

- **Status:** accepted
- **Date:** 2026-04-25
- **Deciders:** Jounes

## Context

Without measurement, "does paperQA work?" reduces to vibes. The project promises page-level citations and grounded answers — neither claim is believable unless something in the repo actually checks them.

Forces:
- Measurement must run offline and deterministically — no dependency on an LLM to grade answers (that collapses the metric back onto vibes).
- A portfolio project does not need a 10k-question benchmark. A small, handcrafted set whose answers a human can verify is more credible than a large set whose gold labels are mysterious.
- Metrics must pin down what we actually commit to: "answers cite their sources" and "retrieval finds the right page".
- The harness must run on the same code path as the live app — no bespoke "test-mode" detours that drift from production.

## Decision

1. **Gold set.** A JSON file at `tests/eval/gold.json` listing questions keyed to a source PDF (one of the committed fixtures), each with:
   - `question`: the user-visible query,
   - `relevant_pages`: the set of pages that contain the answer (1-indexed),
   - `must_cite_page`: a single page that any faithful answer must reference.
   The set starts small (3–5 questions) and grows as the codebase grows.
2. **Two primary metrics (v1):**
   - **Retrieval recall@k** — fraction of `relevant_pages` that appear in the top-k retrieved passages. Computed at `k=1`, `k=3`, `k=5` for each question; aggregated as mean across the set.
   - **Citation faithfulness** — for a generated `Answer`, the fraction of emitted `Citation`s whose `page_number` lies in the gold set's `relevant_pages` for that question. Hallucinated page numbers (already filtered by `parse_citations` to "exists in retrieved set") are caught at this layer by a *stricter* filter: "exists in the *gold* answer set".
3. **Answerer under measurement.** The `StubAnswerer` is the default — it lets us check the *retrieval* side of faithfulness offline. The `HFInferenceAnswerer` path is exercised out-of-band with `pytest -m integration` (or a later manual run) once a token is available.
4. **Harness.** A pure function `run_report(qa, gold_set)` that returns a typed `EvalReport`. A `scripts/run_eval.py` CLI wraps it and prints a table. No hidden state; report JSON is dumped to stdout for scripting.
5. **No regression gate in CI v1.** The first few weeks will be noisy as the set grows. Once the set stabilises, we can add a minimum recall@3 gate — that is a future ADR.

## Alternatives considered

- **Use a public benchmark (QASPER, SciQA).** Rejected for v1: licensing clarity, extra deps, and none of them are page-indexed — translating their gold to page numbers is itself a project. Revisit once the handcrafted set feels too small.
- **LLM-as-judge grading.** Rejected: re-introduces network/cost into measurement, and the judge's bias becomes the floor. Can be added *in addition* later, never as the sole signal.
- **Exact-match answer scoring.** Rejected: generated text is free-form; any exact-match criterion either punishes paraphrase or requires maintaining multiple reference answers per question. Faithfulness + recall capture the contract paperQA actually sells.
- **F1 over retrieved-page sets.** Tracked as a nice-to-have; recall@k is the more actionable number for a top-k retriever.

## Consequences

- **Positive:**
  - The harness is fully offline — runs in under a second, no token required.
  - Metrics map directly to the two promises in the README (page citations + grounded answers), so report deltas are meaningful to a reader.
  - Growing the gold set is a one-JSON-edit operation.
- **Negative:**
  - Handcrafted gold sets are small and reflect the author's assumptions. Bias is mitigated by making the set easy to audit (one JSON file, readable diff).
  - Stub-based runs only exercise the retrieval + citation-plumbing path. Real generator quality is measured separately (integration-mode).
- **Follow-ups:**
  - Add a public-benchmark slice once the handcrafted set saturates.
  - Add a CI gate (`min recall@3 ≥ 0.8`) after the set settles.
  - Add LLM-as-judge as a *secondary* signal, never a primary one.
