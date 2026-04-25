# Baseline 2026-04-25 (sharpened prompt) — Attention Is All You Need

Third baseline, captured immediately after tightening `SYSTEM_INSTRUCTION` in `paperqa/answering.py` to forbid citing pages whose passage does not contain the cited fact. Same gold set as the previous two baselines so deltas are clean.

The previous baseline ([2026-04-25-real-llm-baseline.md](2026-04-25-real-llm-baseline.md)) flagged a real failure mode caught on the live demo: Llama-3.1 cited the cover page (p.1) for an En-Fr BLEU score that actually lives in a table on p.8. The cover page was retrieved (lowest-ranked of top-4) because of keyword overlap, and Llama cited it anyway.

## Setup

Identical to the previous baseline except for the prompt change.

- PDF: `tests/fixtures/attention_is_all_you_need.pdf`.
- Gold: `tests/eval/gold.json`.
- Embedder: MiniLM. Retriever: `DenseRetriever`. `top_k=4`.
- Answerer: `HFInferenceAnswerer` → `meta-llama/Llama-3.1-8B-Instruct`.
- Command: `HF_TOKEN=... python scripts/run_eval.py`.

## Results

| Metric | Original LLM | Sharpened prompt | Δ |
| --- | --- | --- | --- |
| mean recall@1 | 0.583 | 0.583 | — |
| mean recall@3 | 0.833 | 0.833 | — |
| mean recall@5 | 0.833 | 0.833 | — |
| **mean citation faithfulness** | 1.000 | **0.833** | **−0.167** |
| **must-cite rate** | 0.333 | **0.667** | **+0.333** |

Recall is unchanged (retriever untouched), confirming the change isolates the answering layer.

### Per-question

| # | Question | R@3 | Faith (was) | Must-cite (was) |
| - | --- | --- | --- | --- |
| 1 | RNN problems motivating the Transformer | 1.00 | 1.00 (1.00) | ✓ (✗) |
| 2 | Overall Transformer encoder/decoder architecture | 1.00 | 1.00 (1.00) | ✓ (✓) |
| 3 | Scaled Dot-Product Attention & why scaled | 1.00 | 1.00 (1.00) | ✓ (✗) |
| 4 | Multi-Head Attention & motivation | 1.00 | 1.00 (1.00) | ✓ (✓) |
| 5 | BLEU scores on En-De / En-Fr | 1.00 | **0.00** (1.00) | ✗ (✗) |
| 6 | Table 3 ablation on number of attention heads | 0.00 | 1.00 (1.00) | ✗ (✗) |

## Honest read

This is a **partial win, not a clean win.**

- **must_cite_rate doubled (0.333 → 0.667).** Llama is now hitting the exact source page much more reliably on prose questions. Q1 (intro) and Q3 (attention math) flipped from ✗ → ✓. That is the user-visible promise: "the cited page is the page my answer comes from."
- **Faithfulness regressed on Q5 (BLEU) only.** Llama, told to be stricter about citing the page that contains the fact, refused to cite p.8 — the actual location of the BLEU table — and cited a different page instead. Best read: the strict rule pushed the model to look for prose-style assertions of the number ("the model achieves 41.8 on En-Fr") and that prose lives on a *different* page (the discussion section), not Table 2 itself. Tabular data + prose-strict citation rules don't compose well.
- **Q6 still 0.00 recall.** The retriever still misses page 9. The faithfulness=1.00 there is mechanical — Llama emits no citations on Q6, so there is nothing unfaithful. Confirms again that the right next fix for Q6 is retrieval-side (ColPali, ADR-0006), not answering-side.

## Why ship it anyway

- `must_cite_rate` is the more user-facing of the two metrics. A reader checking citations will look at "did this page actually say this?" — that is what `must_cite_rate` measures.
- The faithfulness regression is concentrated in Q5 (tables), and table-handling has a *planned downstream fix* in ADR-0006 (visual retrieval). It is a known-failure-with-known-cure, not a mystery.
- Reverting the prompt change would restore faithfulness but lose the must_cite gain on prose questions Q1 and Q3 — a strictly worse trade for everything that is not a tabular question.

## Targets for next iteration

- **Fix Q5 properly** — two complementary directions:
  1. **Better retrieval** for tables (ColPali / hybrid). Solves it at the source — if p.8 is retrieved with a higher score and the prose-discussion page is retrieved too, Llama has both signals.
  2. **Per-citation grounding check** in code (post-hoc). Re-validate each cited page by checking the cited fact actually appears in that page's text; strip citations that fail. Catches the failure deterministically, no LLM cooperation needed.
- **Loosen `must_cite_page` to `must_cite_pages: frozenset[int]`** in the gold schema. The current "single allowed page" is unrealistically strict — several questions defensibly accept multiple cited pages.
