# ADR-0001: Scope, niche, and initial model choice

- **Status:** accepted
- **Date:** 2026-04-24
- **Deciders:** Jounes

## Context

paperQA is a portfolio project. It needs to be narrow enough to finish and polish, and differentiated enough not to look like another generic RAG demo.

Forces:
- Recruiters and peers skim READMEs and demos; a crisp niche communicates faster than a broad toolkit.
- The author has no dedicated GPU training budget for this project.
- Hugging Face Spaces is the target deployment surface — constrains model size, runtime memory, and cold-start cost.
- The codebase should demonstrate production engineering habits (DRY, tests, docs, CI), not research novelty.

## Decision

1. **Niche:** scientific papers (arXiv-style PDFs). The product is a Document-QA tool that answers natural-language questions against a single paper with page-level citations.
2. **Modelling approach:** use pretrained models only for v1. No from-scratch training; no fine-tuning until a specific, named deficiency motivates it.
3. **Model stack (initial):**
   - **Parsing/OCR:** [Nougat](https://huggingface.co/facebook/nougat-base) for academic-paper layout (formulas, tables).
   - **Retrieval:** start text-first with sentence-transformers embeddings over page-level passages. Re-evaluate whether to switch to [ColPali](https://huggingface.co/vidore/colpali) (vision-based retrieval) after a measurable baseline exists.
   - **Answering:** a small instruction-tuned LLM accessible via Hugging Face Inference API or local CPU-friendly weights (decision deferred to ADR-0003).
4. **Deployment:** Hugging Face Spaces, Gradio UI.
5. **Language:** Python 3.11+.
6. **Engineering constraints (hard):** DRY on second occurrence; code comments explain WHAT and WHY, never HOW; commit messages follow WHAT/WHY/WHERE; documentation is updated in the same commit as the code change.

## Alternatives considered

- **Generic document QA (any PDF type).** Rejected: indistinguishable from dozens of existing demos; loses the "I actually use this" narrative.
- **Invoices / receipts niche (Donut + CORD/SROIE).** Rejected as initial pick: strong business signal but uglier data, less demo-friendly, and less personally useful to the author. Remains a plausible ADR-0001-successor niche if scope changes.
- **Fine-tune from day one.** Rejected: training adds weeks before a demo exists; pretrained baselines are strong enough to ship. Fine-tuning can be added later as an explicit chapter in the project story.
- **Train from scratch.** Rejected: cost/benefit is indefensible for a portfolio piece; recruiters cannot distinguish "trained from scratch" from "well-fine-tuned" from the README.
- **Python + separate JS frontend.** Rejected for v1: Gradio on HF Spaces is the fastest path to a working demo and matches the HF ecosystem the project is built on.

## Consequences

- **Positive:**
  - Narrow niche gives the README and demo a clear elevator pitch.
  - Pretrained-only keeps the critical path short and lets engineering polish be the differentiator.
  - HF Spaces deployment is free and shareable; no infra story required.
  - Scientific-paper data is free and unrestricted (arXiv).
- **Negative:**
  - We inherit Nougat's failure modes (slow on CPU, struggles on non-English layouts); may force a model swap later.
  - HF Spaces cold starts and quotas become a visible UX factor; documentation must set expectations.
  - No fine-tuning means no "look, I trained a model" bullet — mitigated by doing the engineering chapter well.
- **Follow-ups:**
  - ADR-0002: chunking strategy (page-level vs. semantic; how citations are produced).
  - ADR-0003: answering model and inference backend (Inference API vs. local; which model).
  - ADR-0004: evaluation plan (QASPER? SciQA? small handcrafted set?).
  - ADR-0005: when and how to introduce ColPali / visual retrieval as an option.
