# ADR-0006: Visual retrieval — ColPali behind a `Retriever` abstraction

- **Status:** accepted
- **Date:** 2026-04-25
- **Deciders:** Jounes

## Context

The 2026-04-25 baseline (`docs/baselines/2026-04-25-baseline.md`) made one failure mode concrete: the dense MiniLM retriever scores 0.0 on Q6 ("What does the Table 3 ablation tell us about the number of attention heads?") at every k. The page is dominated by a numerical table; pypdf-extracted text is a noisy ribbon of column values that has almost no lexical or semantic overlap with the question's wording. ADR-0003 explicitly flagged this as the trigger condition for visual retrieval.

[ColPali](https://huggingface.co/vidore/colpali-v1.3) is the strongest open answer to that failure mode in 2026: a 3B-parameter vision-language model based on PaliGemma that produces ColBERT-style multi-vector embeddings *of the rendered page image*. The retrieval signal is therefore patch-level visual content, not extracted text — exactly the modality our text retriever is blind to.

Forces:
- The current `Embedder` protocol returns one vector per text. ColPali returns *many* vectors per page (one per image patch) and scores via late interaction (`processor.score_multi_vector`). Squeezing it into `Embedder` would be the wrong abstraction.
- ColPali at 3B parameters does not fit a free-tier CPU HF Space (~2 vCPU, 16 GB RAM). It needs a GPU Space (T4 small or better, paid) or off-Space inference.
- The ColPali paper and `colpali-engine` library are pre-1.0 and the API has shifted between v0.2 and v0.3. We must pin tightly.
- We must not regress the default install path. CPU-only users running `pip install -e .` should keep working.

## Decision

1. **Introduce a `Retriever` protocol** at `paperqa.retrieval.Retriever`:

   ```python
   class Retriever(Protocol):
       def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]: ...
   ```

   The existing dense path becomes `DenseRetriever(passages, embedder)`; `PassageIndex` is renamed/wrapped to fit this shape. `PaperQA` is rewired to accept a `Retriever`, not an `Embedder`.

2. **Add `ColPaliRetriever`** at `paperqa.retrievers.colpali` behind a new optional extra `[visual]`:
   - Renders PDF pages to PIL images (via `pypdfium2` — no system libs required).
   - Embeds page images with `vidore/colpali-v1.3`.
   - At query time embeds the question and runs `processor.score_multi_vector`.
   - Returns the same `RetrievedPassage` shape so the answering layer is unchanged.

3. **Default unchanged.** `paperqa.embedders.SentenceTransformerEmbedder` + `DenseRetriever` remain the default in `app.py` and `scripts/run_eval.py`. `ColPaliRetriever` is opt-in via a runtime flag (`PAPERQA_RETRIEVER=colpali`) or by constructing it directly. The free-tier Space stays runnable.

4. **CI behaviour.** No CI job loads ColPali. A single `integration`-marked test imports `paperqa.retrievers.colpali` and asserts the constructor surface; it is skipped without `PAPERQA_GPU_AVAILABLE=1`. The model weights download (~6 GB) and the GPU requirement keep this out of GitHub Actions.

5. **Pinning.** `colpali-engine>=0.3,<0.4` and `pypdfium2>=4` go in the `[visual]` extra. Pin tightly because the engine API moves.

## Alternatives considered

- **Squeeze ColPali into `Embedder`** (return a flattened multi-vector). Rejected: the late-interaction scoring is the entire point. Flattening loses the patch-level signal and reduces ColPali to a weak single-vector embedder.
- **Replace MiniLM as the default with ColPali.** Rejected: kills the free-tier deployment story. Default must run on CPU.
- **Hybrid retrieval (BM25 + dense + visual) at first.** Rejected for now: too many moving parts to introduce at once. Hybridisation can be a later ADR once each retriever is measured independently against the same gold set.
- **Use `pdf2image` + Poppler instead of `pypdfium2`.** Rejected: Poppler is a system dep that bloats Dockerfiles and HF Space build logs. `pypdfium2` is pip-installable and self-contained.
- **Use a hosted multimodal API (Voyage `voyage-multimodal-3`, etc.).** Rejected as default: same "no API bill per visitor" reasoning as ADR-0004. May be added as a third backend later.

## Consequences

- **Positive:**
  - The `Retriever` abstraction makes future retrievers (BM25, hybrid, hosted multimodal) drop-in.
  - ColPali targets a specific *measured* failure (Q6 in the baseline). The success criterion for the next baseline is unambiguous: recall@3 on Q6 must move above 0.0.
  - CPU-only users are unaffected; the demo Space stays free-tier.
- **Negative:**
  - Visual retrieval needs a GPU at runtime — there is no realistic CPU path. Anyone wanting to run ColPali locally needs CUDA or an Apple-Silicon MPS box.
  - Page rendering (`pypdfium2`) adds ~30 ms/page on top of pypdf text extraction. Negligible per-paper but real.
  - The pinned `colpali-engine<0.4` will need a deliberate bump when v0.4 lands. Track as a recurring maintenance ADR if the API moves substantially.
- **Follow-ups:**
  - Once `ColPaliRetriever` lands, capture a new baseline run with `PAPERQA_RETRIEVER=colpali` and compare against `2026-04-25-baseline.md`.
  - If recall@3 on Q6 moves to ≥ 0.66, consider hybrid retrieval (text + visual) as ADR-0007.
  - Add a "Visual retrieval (paid GPU Space)" deploy path to `docs/deploying.md` once the GPU Space is actually provisioned.
