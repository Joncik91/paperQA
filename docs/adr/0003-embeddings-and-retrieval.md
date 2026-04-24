# ADR-0003: Embeddings and retrieval — MiniLM + in-memory cosine

- **Status:** accepted
- **Date:** 2026-04-25
- **Deciders:** Jounes

## Context

Chunking (ADR-0002) produces one `Passage` per page. The next stage turns a user question into a ranked list of passages so the answering model sees only the relevant pages.

Forces:
- Target surface is Hugging Face Spaces — RAM and cold-start budget are tight.
- A single paper produces tens to low hundreds of passages. Full-scan cosine over that many vectors is microseconds; a vector DB is overkill.
- The project is pretrained-only for v1 (ADR-0001); we want an embedder with no license friction and a small footprint.
- The index layer should be testable without downloading model weights in CI.

## Decision

1. **Embedder (default):** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). 384-dim, ~80 MB, CPU-friendly, strong on short-to-medium passages, Apache-2.0.
2. **Retrieval:** in-memory cosine similarity over a NumPy matrix of passage embeddings. No FAISS, no Chroma, no sqlite-vss in v1.
3. **Abstraction:** an `Embedder` protocol (`embed(texts: list[str]) -> np.ndarray`) and a `PassageIndex` class (`build(passages, embedder)` → object, `query(question, embedder, top_k)` → ranked passages + scores). The protocol lets tests inject a deterministic fake embedder and lets us swap to ColPali or a hosted embedding API later without touching the index.
4. **Scope boundary:** the index is per-document (one paper per session). Multi-document corpora, persistence, and incremental updates are explicitly out of scope for v1.

## Alternatives considered

- **OpenAI / Cohere / Voyage hosted embeddings.** Rejected for v1: adds an API key, a cost line, and a network dependency to a project that can run fully local. Reconsider if quality becomes the limiting factor.
- **FAISS / Chroma / lancedb.** Rejected: unnecessary at this corpus size, adds a heavy dependency, and obscures the retrieval logic we want to be auditable.
- **BM25 / keyword retrieval.** Rejected as the primary path — scientific questions often paraphrase the paper's wording. BM25 remains a viable *addition* (hybrid retrieval) in a later ADR if recall becomes a measured problem.
- **ColPali / visual retrieval.** Deferred — still on the roadmap per ADR-0001, gated on a measurable baseline first (future ADR-0005).

## Consequences

- **Positive:**
  - Zero infra: an index is a NumPy array and a list of passages.
  - Tests are fully deterministic with a fake embedder — no model download in CI.
  - Swapping embedders later is a one-line change: pass a different `Embedder` impl to `PassageIndex.build`.
- **Negative:**
  - MiniLM is good, not state-of-the-art. Complex multi-hop scientific questions may miss. Tracked as a candidate trigger for ADR-0005 (visual retrieval) or a fine-tune ADR.
  - In-memory only means every session re-embeds the uploaded PDF. Caching per-file is a small, later optimisation — noted, not scheduled.
- **Follow-ups:**
  - ADR-0004: answering model + inference backend.
  - Future ADR: persistence / caching of embeddings by PDF content hash.
  - Future ADR: hybrid BM25+dense retrieval if recall needs it.
