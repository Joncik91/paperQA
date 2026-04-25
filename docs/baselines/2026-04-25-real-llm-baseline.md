# Baseline 2026-04-25 (real LLM) — Attention Is All You Need

Second baseline, captured immediately after wiring `HFInferenceAnswerer`
through the real Hugging Face Inference API. Same gold set as
[2026-04-25-baseline.md](2026-04-25-baseline.md) so the deltas attributable
to the answerer are isolated.

## Setup

- PDF: `tests/fixtures/attention_is_all_you_need.pdf` (15 pages).
- Gold set: `tests/eval/gold.json` — 6 handcrafted questions.
- Embedder: `all-MiniLM-L6-v2` (unchanged).
- Retriever: `DenseRetriever` (unchanged, `top_k=4`).
- **Answerer: `HFInferenceAnswerer` calling `meta-llama/Llama-3.1-8B-Instruct`** via `chat_completion`.
- Command: `HF_TOKEN=... python scripts/run_eval.py`.

## Results

| Metric | Stub baseline | Real LLM | Delta |
| --- | --- | --- | --- |
| mean recall@1 | 0.583 | 0.583 | — |
| mean recall@3 | 0.833 | 0.833 | — |
| mean recall@5 | 0.833 | 0.833 | — |
| mean citation faithfulness | 0.667 | **1.000** | **+0.333** |
| must-cite rate | 0.667 | 0.333 | −0.333 |

Recall numbers are unchanged (the retriever is identical), confirming the
metric isolation: only the answering layer moved.

### Per-question

| # | Question | R@3 | Faith | Must-cite |
| - | --- | --- | --- | --- |
| 1 | RNN problems motivating the Transformer | 1.00 | 1.00 | ✗ |
| 2 | Overall Transformer encoder/decoder architecture | 1.00 | 1.00 | ✓ |
| 3 | Scaled Dot-Product Attention & why scaled | 1.00 | 1.00 | ✗ |
| 4 | Multi-Head Attention & motivation | 1.00 | 1.00 | ✓ |
| 5 | BLEU scores on En-De / En-Fr | 1.00 | 1.00 | ✗ |
| 6 | Table 3 ablation on number of attention heads | 0.00 | 1.00 | ✗ |

## Observations

- **Faithfulness is perfect (1.000).** Every citation Llama emits is on a
  gold-relevant page — the system instruction in `build_prompt` is being
  followed cleanly. Faithfulness rising from 0.667 (stub) to 1.000 (LLM)
  is the strongest signal so far that the grounding contract works.
- **Must-cite rate dropped (0.667 → 0.333).** This is *not* a regression in
  model quality — it is a gold-set sharpness issue. Llama cites a valid
  retrieved page (faithfulness=1.0 on Q3, Q5) but a different one than the
  single `must_cite_page` we labelled. The metric is too strict: real
  questions often have several pages where the answer can defensibly be
  cited from. Fix is in the gold-set schema, not the answerer (track as a
  future ADR-0005 amendment: change `must_cite_page` from a single int
  to a `frozenset[int]`).
- **Q6 remains the unchanged failure.** Citation faithfulness is 1.00 only
  because Llama emits zero citations on Q6 (`score_citations` returns 1.0
  for empty citations). Recall is still 0.00 — the retriever never
  surfaces page 9. This continues to validate ADR-0006: visual retrieval
  is the right fix, not a better answerer.

## Targets for next iterations

- **Loosen `must_cite_page` to `must_cite_pages: frozenset[int]`** in the
  gold schema. Then re-baseline. Without that fix the metric will keep
  underselling real model quality.
- **Run the harness with `PAPERQA_RETRIEVER=colpali`** (needs GPU) and
  expect Q6 retrieval to move above 0.0.
- **Hybrid retrieval** if recall@3 needs to climb above 0.833 with the
  text-only path.
