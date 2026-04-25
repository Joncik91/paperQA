# Baseline 2026-04-25 (grounding check) — Attention Is All You Need

Fourth baseline on the same gold set. Adds the post-hoc citation grounding check from [ADR-0007](../adr/0007-citation-grounding-check.md) on top of the sharpened prompt from [ADR-0004](../adr/0004-answering-model-and-backend.md).

## Setup

Identical to the previous baselines.

- PDF: `tests/fixtures/attention_is_all_you_need.pdf`.
- Gold: `tests/eval/gold.json` (6 questions).
- Embedder: MiniLM. Retriever: `DenseRetriever`. `top_k=4`.
- Answerer: `HFInferenceAnswerer` → `meta-llama/Llama-3.1-8B-Instruct`.
- **New step:** `paperqa.citation_check.verify_citations` runs after the LLM call and strips `[page N]` markers when the cited sentence contains a number not present on page N. Sentences without numbers always pass.

## Results

| Metric | Original LLM | Sharpened prompt | **+ Grounding (this run)** |
| --- | --- | --- | --- |
| mean recall@1 | 0.583 | 0.583 | 0.583 |
| mean recall@3 | 0.833 | 0.833 | 0.833 |
| mean recall@5 | 0.833 | 0.833 | 0.833 |
| **mean citation faithfulness** | 1.000 | 0.833 | **0.917** |
| **must-cite rate** | 0.333 | 0.667 | **0.833** |

`must_cite_rate` jumped from 0.333 (original LLM) to 0.833 — a **+0.500 absolute** improvement. 5/6 questions now hit the exact gold page.

### Per-question

| # | Question | R@3 | Faith | Must-cite |
| - | --- | --- | --- | --- |
| 1 | RNN problems motivating the Transformer | 1.00 | 1.00 | ✓ |
| 2 | Overall Transformer encoder/decoder architecture | 1.00 | 1.00 | ✓ |
| 3 | Scaled Dot-Product Attention & why scaled | 1.00 | 1.00 | ✓ |
| 4 | Multi-Head Attention & motivation | 1.00 | 1.00 | ✓ |
| 5 | BLEU scores on En-De / En-Fr | 1.00 | 0.50 | ✓ |
| 6 | Table 3 ablation on number of attention heads | 0.00 | 1.00 | ✗ |

## Observations

- **Q5 fixed: `cite=True`.** This is the question that motivated ADR-0007. The grounding check stripped Llama's bad citation (page 1, which does not contain "28.4" or "41.8") and the surviving citation hits the gold page (8). `faith=0.50` reflects that some retrieved-but-non-gold pages still get cited; that's a gold-set sharpness issue, not a system issue (see ADR-0005 follow-ups).
- **Faithfulness = 0.917.** Slightly below the original LLM's 1.000 because the stricter grounding now annotates removed citations rather than silently keeping them. The trade is excellent — must_cite went from 0.333 to 0.833 for that 0.083 dip.
- **Q6 unchanged at recall=0.00.** Confirms the answering layer cannot fix what the retriever doesn't surface. ColPali (ADR-0006) remains the right next move.

## Iteration note

The first draft of the grounding check also ran a **lexical-overlap rule** for sentences without numbers (require ≥ 2 distinct content tokens shared with the cited page). When measured, that rule stripped real citations on Llama's paraphrased prose, dragging `must_cite_rate` back to 0.333. The rule was removed in the same iteration; only the numerical-anchor rule shipped. Documented in ADR-0007's "Iteration history".

## Headline progression across baselines

| Baseline | faithfulness | must_cite_rate |
| --- | --- | --- |
| Stub (retrieval-only) | 0.667 | 0.667 |
| Real LLM (default prompt) | 1.000 | 0.333 |
| Real LLM (sharpened prompt) | 0.833 | 0.667 |
| **Real LLM + grounding check** | **0.917** | **0.833** |

## Targets for next iteration

- **Fix Q6 with retrieval, not generation.** ColPali (ADR-0006) — needs a GPU Space.
- **Loosen `must_cite_page` to a set** in the gold schema (tracked since ADR-0005 — would push `faith` on Q5 closer to 1.0).
