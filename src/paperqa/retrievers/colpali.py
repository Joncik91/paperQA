"""ColPali-based visual retriever.

WHAT: `ColPaliRetriever` renders a PDF's pages to images, embeds each one
      with a ColPali multi-vector model (default: vidore/colpali-v1.3),
      and at query time scores the question against the page embeddings
      with `processor.score_multi_vector`. Returns the same
      `RetrievedPassage` shape as the dense retriever, so the answering
      layer is unchanged.
WHY:  Targets the failure mode the 2026-04-25 baseline made concrete: the
      dense MiniLM retriever scores 0.0 on table-heavy pages where
      pypdf-extracted text is a noisy ribbon of column values. ColPali
      reads the page image directly, so tables and figures are first-class
      retrieval signals (ADR-0006).

Heavy / GPU-only. Lives in its own module + optional `[visual]` extra so
the core install path stays CPU-friendly and CI never loads the model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paperqa.chunking import Passage
from paperqa.indexing import RetrievedPassage

if TYPE_CHECKING:
    # Type-only imports keep these optional deps off the import path.
    from PIL.Image import Image

__all__ = ["DEFAULT_MODEL", "ColPaliRetriever"]

# vidore/colpali-v1.3 is the current recommended checkpoint as of 2026-04.
# When we bump this, bump ADR-0006 in the same commit — the choice is part
# of the reproducibility contract.
DEFAULT_MODEL = "vidore/colpali-v1.3"

# Render DPI for pypdfium2. 144 is a reasonable balance between visual
# detail (tables stay legible) and memory (~1 MB per page on a typical
# arXiv PDF). Tune only with a measurement.
DEFAULT_DPI = 144

logger = logging.getLogger(__name__)


class ColPaliRetriever:
    """Visual retriever — embeds rendered page images with ColPali.

    The model and processor are loaded lazily on first `retrieve()` call;
    page images are rendered and embedded at that point too so per-query
    latency afterwards is just a question encode plus a score call.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        passages: list[Passage],
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        dpi: int = DEFAULT_DPI,
    ) -> None:
        # Keep the passage list aligned to the page-image list by index so
        # we can map a ranked image back to its `Passage` (and therefore
        # its page number) without an extra lookup table.
        self._pdf_path = Path(pdf_path)
        self._passages = passages
        self._model_name = model_name
        self._device = device or _autodetect_device()
        self._dpi = dpi
        self._model: Any | None = None
        self._processor: Any | None = None
        self._page_embeddings: Any | None = None

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        if not self._passages:
            return []
        self._ensure_indexed()
        scores = self._score(question)
        # `scores` is shape (1, n_pages); flatten to a list of floats so
        # the rest of the function does not depend on torch.
        scored = [(float(s), i) for i, s in enumerate(scores[0].tolist())]
        scored.sort(reverse=True)
        k = min(top_k, len(self._passages))
        return [RetrievedPassage(passage=self._passages[i], score=score) for score, i in scored[:k]]

    def _ensure_indexed(self) -> None:
        if self._page_embeddings is not None:
            return
        # Lazy import keeps colpali-engine + torch + transformers off the
        # paperqa import path. Anything imported here belongs to [visual].
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        logger.info("Loading ColPali model %s on %s", self._model_name, self._device)
        model = ColPali.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16 if self._device != "cpu" else torch.float32,
            device_map=self._device,
        )
        # WHY .eval(): turns off dropout and batch-norm running-stat updates
        # for inference. ColPali's checkpoint is published with training-mode
        # defaults. (This is torch idiom, unrelated to Python's eval().)
        self._model = model.eval()
        self._processor = ColPaliProcessor.from_pretrained(self._model_name)

        images = self._render_pages()
        # Process and embed in one batch — page counts here are tens, not
        # thousands, so a single batch is fine on a small GPU.
        with torch.no_grad():
            batch = self._processor.process_images(images).to(self._device)
            self._page_embeddings = self._model(**batch)

    def _score(self, question: str) -> Any:
        import torch

        assert self._model is not None
        assert self._processor is not None
        assert self._page_embeddings is not None
        with torch.no_grad():
            batch = self._processor.process_queries([question]).to(self._device)
            query_embeddings = self._model(**batch)
        return self._processor.score_multi_vector(query_embeddings, self._page_embeddings)

    def _render_pages(self) -> list[Image]:
        # pypdfium2 is the chosen rasteriser (ADR-0006): pip-installable,
        # no Poppler / system deps, ~30 ms/page on the baseline hardware.
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(self._pdf_path)
        scale = self._dpi / 72.0
        images: list[Image] = []
        try:
            for page in pdf:
                pil = page.render(scale=scale).to_pil()
                images.append(pil)
                page.close()
        finally:
            pdf.close()
        return images


def _autodetect_device() -> str:
    """Pick CUDA, then MPS, then CPU.

    WHY auto-detect: callers should not have to know what hardware their
    Space happens to be running on; the only realistic deployment surfaces
    are GPU Spaces (CUDA) or local Apple Silicon (MPS) for prototyping.
    Falling back to CPU is supported but practically unusable for ColPali
    — left in only so import-time tests do not crash.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
