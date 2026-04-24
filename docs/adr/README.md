# Architecture Decision Records

Every non-obvious decision that shapes paperQA's design lands here as an ADR.

## Index

- [0001 — Scope, niche, and initial model choice](0001-scope-and-model-choice.md) — accepted
- [0002 — Chunking strategy — page-level passages](0002-chunking-strategy.md) — accepted
- [0003 — Embeddings and retrieval — MiniLM + in-memory cosine](0003-embeddings-and-retrieval.md) — accepted
- [0004 — Answering model + inference backend](0004-answering-model-and-backend.md) — accepted

## How to add one

1. Copy `TEMPLATE.md` to `NNNN-short-title.md` (next integer, zero-padded to 4).
2. Fill in Context, Decision, Alternatives, Consequences.
3. Add a one-line entry to the index above in the same commit.
4. Reference the ADR from the commit body (`Implements ADR-NNNN.`) and from code comments where the decision is load-bearing.
