# ADR-0008: Multi-document support — pooled index, file-prefixed citations

- **Status:** accepted
- **Date:** 2026-04-25
- **Deciders:** Jounes

## Context

The five baselines captured on 2026-04-25 push the single-paper QA story to a strong place: faithfulness 1.000, must-cite 0.833, citations grounded in the page they actually came from. The natural next functional gap is **corpus-level QA** — letting a user upload several PDFs and ask one question across all of them. Use cases the single-doc surface cannot serve: literature review ("does any of these three papers report a BLEU above 40?"), cross-paper comparison, "summarise the related-work section across these references".

The codebase already anticipated this: `Passage.source_path` (paperqa/chunking.py) carries the source file from day one. Most of the work is plumbing that field through the prompt, the parser, the UI, and the cache key.

Forces:
- The default deploy (HF Space, CPU basic) must keep working. A multi-doc design that breaks the single-doc UX is unacceptable.
- Cosine similarity scores are only directly comparable when the vectors live in a single normalized space. Mixing scores from N independently normalized indexes is statistically dishonest unless explicitly re-scored.
- The citation grain is *the* product promise (ADR-0002). Multi-doc citations must remain page-anchored *and* source-anchored — `[paper.pdf, page 8]` not just `[page 8]`.
- The grounding check (ADR-0007) must keep firing per-citation; a multi-doc retrieved set may have two PDFs that both have a page 9 with different content.

## Decision

1. **One pooled index across all uploaded PDFs.** `PaperQA.ask(pdf_paths, question)` calls `chunk_by_page` per PDF, concatenates all `Passage`s, and hands them to a single `Retriever`. Cosine scores stay comparable because every passage is embedded in the same normalized space.

2. **Citation marker:** `[<filename>, page N]`. `build_prompt` emits each passage header as `[<source_path.name>, page N]`. `parse_citations` and `verify_citations` recognise the new shape. The legacy `[page N]` form is still accepted as a back-compat path, but only when *unambiguous* — exactly one retrieved passage has that page number. In a multi-doc retrieved set where two PDFs both have a page 1, a bare `[page 1]` citation is refused (parse) or left for `parse_citations` to drop (verify). Refusing to guess > silently picking one.

3. **Cache key:** sorted tuple of resolved PDF paths. `ask([a, b])` and `ask([b, a])` hit the same cached retriever. Adding or removing a PDF from the set rebuilds. Single-doc calls — `ask(path)` and `ask([path])` — normalise to the same single-element tuple, so single-doc behaviour is byte-identical to before.

4. **`Citation` dataclass** gains a `source_path: Path` field. Re-exported. `parse_citations` returns `Citation(source_path, page_number, score)`. Existing callers that only read `page_number` keep working; new callers (UI, future evaluation harness) can read both.

5. **`RetrieverFactory` signature unchanged.** Still `(Path, list[Passage]) -> Retriever`. The pipeline passes the *first* path of the cache-key tuple as a representative; text retrievers ignore it (they only need the pooled passages, each of which carries its own `source_path`). Visual retrievers (ColPali) read the path — multi-doc visual retrieval is therefore *not* supported in v1; ADR-0006 follow-up to revisit when needed.

## Alternatives considered

- **N per-PDF retrievers, merge top-k post-hoc.** Cleaner cache (per-file, no rebuild on set change), but cross-document score merging is statistically dodgy. A top-1 hit in a small irrelevant doc can outrank a top-3 hit in a large relevant doc because their score distributions are shaped differently. Rejected for v1; reachable as a future hybrid if recall demands it.
- **Numeric source IDs `[src 1, page 8]`.** Lower hallucination risk (the model can't fabricate filenames) but worse readability — the UI has to translate IDs back to filenames, and the citation marker stops being human-readable in the raw answer text. Rejected; the LLM's instruction-following plus the `parse_citations` filter handle the hallucination case adequately.
- **Per-source filtering at retrieval time** (`pdf=specific.pdf` query parameter). Useful but not necessary for v1's pooled-index design. Adding it later is a one-line scan over the pooled passages. Out of scope.
- **Multi-doc evaluation gold set** (`tests/eval/gold.json` extended with cross-document questions). Out of scope for this slice — the gold set is single-paper today, building a multi-doc gold is its own project. Tracked as an ADR-0005 follow-up.

## Consequences

- **Positive:**
  - Demo unlocks corpus-level use cases (literature review, cross-paper comparison) without changing the basic UX (just upload N files instead of 1).
  - Pooled-index scoring is the most defensible multi-doc retrieval semantics short of a learned cross-doc reranker.
  - Cache invariance under input order means the UI can re-render selections without invalidating embeddings.
  - Single-doc behaviour is unchanged — same code path, same observed outputs (other than the citation format gaining a filename).

- **Negative:**
  - Adding a PDF to a working set rebuilds the whole pooled index. For a 50-page paper that's <2 s on the CPU Space, but for a 30-paper corpus the rebuild becomes noticeable. Persist-by-content-hash (planned but not specced) is the eventual fix.
  - The pooled-index design assumes all PDFs share the same embedder. Mixing text retriever (MiniLM) and visual retriever (ColPali) over different PDFs is not supported. Realistic for v1; a heterogeneous deploy is a future ADR.
  - Filename collisions (two uploaded files both named `paper.pdf`) make the citation `[paper.pdf, page 3]` ambiguous. v1 silently keys on the basename only; the second upload's `paper.pdf` shadows the first in the cited-pages render. Documented limitation; real fix would be filename disambiguation (`paper.pdf` and `paper-2.pdf`) at upload time.
  - Visual retrieval (ColPali, ADR-0006) is single-doc-only under this design. Multi-doc visual retrieval needs a separate factory shape that takes `list[Path]`. Out of scope here.

- **Follow-ups:**
  - Per-PDF embedding cache by content hash — would let re-uploading the same PDF skip the embed pass entirely.
  - Multi-doc gold set extension to `tests/eval/`. Until then, the multi-doc baseline is a smoke check, not a measurement.
  - Filename disambiguation at upload time (handle two `paper.pdf` files cleanly).
  - Multi-doc visual retrieval (ADR-0006 follow-up).
