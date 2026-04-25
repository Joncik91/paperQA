# ADR-0007: Per-citation grounding check (post-hoc)

- **Status:** accepted
- **Date:** 2026-04-25
- **Deciders:** Jounes

## Context

The 2026-04-25 sharpened-prompt baseline ([2026-04-25-sharpened-prompt.md](../baselines/2026-04-25-sharpened-prompt.md)) and a follow-up live-demo test exposed a stubborn failure mode: Llama-3.1 cites the page with the strongest *topical* match for a question's vocabulary, even when the cited page does not contain the *fact* it just stated.

The clearest example is Q5 (BLEU question on Attention Is All You Need): Llama answers with `28.4` and `41.8` — the correct numbers — and cites `[page 1]`. The numbers themselves appear in Table 2 on page 8; page 1 contains the paper title, the author list, and the abstract. Page 1 was retrieved at the bottom of the top-4 because of keyword overlap on "BLEU" / "English-to-German", and Llama anchored on it.

Sharpening the system prompt (ADR-0004) helped on prose questions (must-cite rate 0.333 → 0.667) but did not move the tabular case. Llama's instruction-following is real but bounded; for facts that live in tables, the model emits the right numbers and the wrong citation.

Forces:
- Citation correctness is the project's headline claim. A user who clicks the cited page and finds nothing matching the answer loses trust immediately.
- The fix should run on the live CPU Space — no extra LLM calls (cost, latency, rate limits).
- It should run on the same offline path as the measurement harness so its impact is measurable.
- It must not strip *correct* citations on table-heavy pages where the page text is mostly numerical noise.

## Decision

Add a deterministic, post-hoc grounding check that runs after the LLM emits an answer:

1. For each sentence in the answer that contains a `[page N]` marker, extract the **numbers** (digits, decimals, percentages) appearing in that sentence (excluding the marker itself).
2. **Numerical-anchor rule:** if the sentence contains any number, every such number must appear in the cited page's passage. Numbers are the most reliable grounding signal — a sentence claiming "28.4" cannot honestly cite a page where "28.4" does not appear.
3. Citations failing the rule are removed from the answer text. The sentence is rewritten to drop the marker and append `_(citation removed: page N does not contain this fact)_` so the user sees what happened.
4. Sentences without numbers always pass — see "Iteration history" below.
5. Re-run `parse_citations` over the cleaned text so the returned `Answer.citations` reflects only the surviving page references.

The implementation lives in `paperqa/citation_check.py` and is called from `HFInferenceAnswerer.answer()` (and any future answerer) immediately after generation, before `parse_citations`.

### Iteration history (kept honest)

The first draft of this ADR added a **lexical-overlap rule** for non-numeric sentences (require ≥ 2 distinct content-token overlaps between the sentence and the cited page's passage). Implemented and re-baselined: it stripped legitimate citations on Llama's paraphrased prose answers, dragging `must_cite_rate` from 0.667 back down to 0.333. The lexical rule was an over-eager guess against a failure mode (prose-cites-wrong-page) we have not actually observed — only the numerical case is real today. Removed in the same iteration; only the numerical-anchor rule shipped.

## Alternatives considered

- **Stricter prompting alone.** Already attempted (sharpened SYSTEM_INSTRUCTION); helped on prose, did not move tabular. LLM cooperation is bounded.
- **LLM-as-verifier (second call).** Reliable but pays an extra round-trip and another rate-limit slot per question. Rejected for v1 — keep this option for a future ADR if deterministic checks plateau.
- **Embedding similarity per cited claim.** More forgiving than token overlap but requires another embedding call per sentence and introduces a similarity threshold that has its own tuning surface. Rejected as over-engineered for the failure mode we actually see.
- **Strip citations entirely; force the user to read the retrieved pages.** Trades the headline feature for hands-off honesty. Rejected — the project's pitch is "answer with cited pages", not "answer with a list of pages to read."

## Consequences

- **Positive:**
  - Catches the exact failure mode the live demo exhibited (Llama citing page 1 for numbers that live on page 8).
  - Deterministic and offline — runs in <1 ms per answer; no extra LLM calls; no new dependencies.
  - The "citation removed" annotation is itself a UX feature: it tells the user *why* the citation went away and which page Llama hallucinated.
  - Stacks cleanly with future retrieval improvements (BM25, ColPali) — they reduce *how often* the check has to fire.
- **Negative:**
  - Token-overlap rules will sometimes strip *valid* citations where the user's question paraphrases heavily and the cited page uses different wording. We mitigate by (a) the numerical-anchor rule firing on the most discriminative tokens, (b) keeping the lexical threshold low (2 tokens). Edge cases will be visible in the next baseline.
  - PDFs whose text extraction is very poor (image-based scans) will fail the lexical rule on every citation. Acceptable — those PDFs already fail at retrieval, the grounding check just makes the failure honest.
- **Follow-ups:**
  - Re-baseline immediately and capture as `2026-04-25-grounding-check.md`.
  - Future ADR: LLM-as-verifier as an opt-in second pass once we are willing to pay the extra call.
  - Future ADR: per-claim grounding (vs per-sentence) using structured-output JSON mode.
