# ADR-0002: Chunking strategy — page-level passages

- **Status:** accepted
- **Date:** 2026-04-24
- **Deciders:** Jounes

## Context

paperQA's core value proposition is **page-level citations**: every answer points back to a specific page of the source PDF. That commitment constrains how we split a document for retrieval.

Forces:
- Citations must be verifiable at a glance — "page 4, paragraph 2" is actionable; "passage 137" is not.
- Scientific papers mix prose, formulas, figures, tables, and footnotes. Naive sentence splitters shred equations and caption/body relationships.
- We want the chunker to be fast, deterministic, and testable without running a GPU model.
- v1 is pretrained-only and CPU-friendly (ADR-0001) — no heavy layout parsing on the chunker's critical path.

## Decision

1. **Unit of chunking = one PDF page.** Each page becomes exactly one `Passage` with `page_number` (1-indexed), `text`, and `source_path`.
2. **Text extraction via `pypdf`** for v1 — pure Python, no system deps, adequate for most arXiv papers. Nougat (ADR-0001) enters the pipeline later, at parse-quality time, not at chunking time.
3. **No sub-page splitting in v1.** Embedding models tolerate 512-to-2048-token passages; full pages typically fit. If a page overflows the embedder's context, we truncate at encode time and log it — splitting is deferred until we see it matter.
4. **`Passage` is the canonical data contract** between ingest → index → retrieve → answer. It lives in `src/paperqa/chunking.py` and is re-exported from the package root.

## Alternatives considered

- **Sentence-level chunking** (spaCy / nltk). Rejected: breaks equation and figure-caption locality; makes citation grain finer than the user needs; adds a tokenizer dependency.
- **Fixed-token windowing with overlap** (classic RAG). Rejected for v1: obscures page boundaries, which *is* our citation grain. Reconsider if retrieval quality demands it (future ADR).
- **Layout-aware chunking with Nougat on ingest.** Rejected for v1: slow on CPU, adds a heavy dependency to the critical path, and is not needed to deliver the first demo. Revisit once a baseline exists.
- **Semantic chunking** (embed-and-cluster). Rejected: destroys the page→citation mapping; adds cost without clear benefit for single-paper QA.

## Consequences

- **Positive:**
  - Citations are trivially faithful: `passage.page_number` *is* the citation.
  - Chunking is a pure function of the PDF — fully deterministic, unit-testable offline.
  - No model dependency in the chunking layer; fast CI.
- **Negative:**
  - Very long pages may hit embedding-model context limits and will be truncated at encode time. Acceptable for v1; tracked for later.
  - Pages with mostly figures and little extractable text will produce near-empty passages. The retriever will correctly down-rank them; no action needed now.
- **Follow-ups:**
  - ADR-0003: answering model and inference backend.
  - Revisit sub-page splitting if retrieval recall on long pages becomes a measured problem (track with the eval harness from the planned ADR-0004).
