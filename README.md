---
title: paperQA
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.13.0"
app_file: app.py
pinned: false
license: mit
short_description: Document QA for scientific papers with page-level citations.
---

# paperQA

[![ci](https://github.com/Joncik91/paperQA/actions/workflows/ci.yml/badge.svg)](https://github.com/Joncik91/paperQA/actions/workflows/ci.yml)
[![demo](https://img.shields.io/badge/🤗-Live%20demo-FFD21F)](https://huggingface.co/spaces/Joncik/paperqa)

**Ask questions of a scientific PDF. Get answers grounded in the source, with the exact page they came from.**

> 👉 **[Try the live demo](https://huggingface.co/spaces/Joncik/paperqa)** — upload any arXiv-style paper, ask a question, see the cited pages side-by-side.

## What it does

Upload a paper. Type a question. Get back an answer that:

- cites the **exact page** it came from (`[page 8]`),
- shows the **source passage** beside the answer so you can verify it,
- never invents page numbers — citations are filtered against retrieved pages first, then against the gold set during measurement.

The system is specialised for arXiv-style scientific PDFs. The same pipeline works on any PDF, but the prompts and the included gold set are tuned for academic papers.

## Why it's different from a generic RAG demo

Most "chat with your PDF" demos stop at "model returns text." This project's portfolio claim is the engineering around the model:

- **Page-level citation grain** is a design decision, not an afterthought (see [ADR-0002](docs/adr/0002-chunking-strategy.md)). Every passage **is** a page; faithful citations follow by construction.
- **Pluggable backends.** `Answerer` and `Retriever` are protocols, not classes — the offline `StubAnswerer` (CI), `HFInferenceAnswerer` (live demo), and the visual `ColPaliRetriever` (planned GPU Space) all satisfy the same contract.
- **Measurement, not vibes.** [`docs/baselines/`](docs/baselines/) ships real numbers — every architecture change ships next to a recall@k / faithfulness delta. See [`docs/adr/0005-evaluation-plan.md`](docs/adr/0005-evaluation-plan.md).
- **Mature CI.** Lint + format + strict mypy + 41 unit tests on every push, across Python 3.11 and 3.12. Integration tests are gated behind `pytest -m integration` so the network never enters CI.

## Headline numbers (Attention Is All You Need, 6 questions)

Measured on the included [`tests/eval/gold.json`](tests/eval/gold.json) gold set, with `Llama-3.1-8B-Instruct` as the answerer:

| Metric | Value | What it means |
| --- | --- | --- |
| **mean recall@3** | **0.833** | The retriever puts the right page in the top 3 on 5/6 questions |
| **mean recall@1** | 0.583 | Top-1 is less reliable — that's why default `top_k = 4` |
| **mean citation faithfulness** | **1.000** | Every citation Llama emits is on a gold-relevant page |
| **must-cite rate** | 0.333 | Strict gold-set metric (single allowed page per question; relaxing this is tracked) |

Full detail and the per-question breakdown are at [`docs/baselines/2026-04-25-real-llm-baseline.md`](docs/baselines/2026-04-25-real-llm-baseline.md).

## How it works

1. **Ingest** — `pypdf` extracts one `Passage` per PDF page.
2. **Index** — pages are embedded with `all-MiniLM-L6-v2`; the index is an in-memory NumPy matrix (per [ADR-0003](docs/adr/0003-embeddings-and-retrieval.md), no vector DB needed for single-paper QA).
3. **Retrieve** — the question is embedded and scored against the page matrix; top-k pages are returned.
4. **Answer** — the question + retrieved passages go to `Llama-3.1-8B-Instruct` via the HF Inference API. The system prompt forces the model to cite `[page N]` and refuse out-of-document questions.
5. **Cite** — emitted `[page N]` markers are parsed back, filtered against the retrieved set (no hallucinated page numbers), and rendered next to the source passages.

The visual-retrieval path (ColPali, [ADR-0006](docs/adr/0006-visual-retrieval-colpali.md)) targets the one measured failure case: questions about **table-heavy pages** where text extraction loses the signal. It's gated behind a `[visual]` extra and a `PAPERQA_RETRIEVER=colpali` env switch — only relevant on a GPU Space.

## Architectural decisions

The interesting calls are recorded as ADRs, not buried in commits:

- [ADR-0001 — Scope, niche, and initial model choice](docs/adr/0001-scope-and-model-choice.md)
- [ADR-0002 — Chunking strategy: page-level passages](docs/adr/0002-chunking-strategy.md)
- [ADR-0003 — Embeddings and retrieval: MiniLM + in-memory cosine](docs/adr/0003-embeddings-and-retrieval.md)
- [ADR-0004 — Answering model and inference backend](docs/adr/0004-answering-model-and-backend.md)
- [ADR-0005 — Evaluation plan](docs/adr/0005-evaluation-plan.md)
- [ADR-0006 — Visual retrieval: ColPali behind a Retriever abstraction](docs/adr/0006-visual-retrieval-colpali.md)

## Running locally

```bash
git clone https://github.com/Joncik91/paperQA.git
cd paperQA
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,embed,llm,app]"
python app.py                 # opens Gradio on localhost:7860
```

Set `HF_TOKEN` to use the real Inference API; without it the app falls back to the offline `StubAnswerer` so the demo never hard-errors.

Full guide: [`docs/running-locally.md`](docs/running-locally.md). Deploying to a Space: [`docs/deploying.md`](docs/deploying.md).

## Engineering rules

This is also a portfolio piece, so the discipline is part of the product. The hard rules are codified in [`CONTRIBUTING.md`](CONTRIBUTING.md):

- **DRY** on the second occurrence — no copy-paste tolerated.
- **Code comments** explain WHAT and WHY, never HOW. The code is the HOW.
- **Commit messages** are WHAT changed, WHY it changed, WHERE it landed. One logical change per commit.
- **Documentation lands in the same commit as the code it describes.** No "I'll write the docs later."

## License

MIT — see [LICENSE](LICENSE).
