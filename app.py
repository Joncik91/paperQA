"""Gradio entry point for the paperQA Hugging Face Space.

WHAT: Single-file Gradio app. Uploads a PDF, takes a question, renders an
      answer plus the cited page excerpts. Uses the real HF Inference
      backend when HF_TOKEN is set; falls back to the offline StubAnswerer
      otherwise so the demo never hard-errors on a missing token.
WHY:  HF Spaces looks for app.py at the repo root. Keeping the UI a thin
      wrapper over `PaperQA` (from paperqa.pipeline) means the UI and any
      future CLI share the same orchestration — no drift (DRY).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import gradio as gr

from paperqa import AskResult, PaperQA, Passage, StubAnswerer
from paperqa.answering import Answerer
from paperqa.embedders import SentenceTransformerEmbedder
from paperqa.pipeline import RetrieverFactory
from paperqa.retrieval import DenseRetriever, Retriever


def _build_answerer() -> tuple[Answerer, str]:
    """Pick the best available answerer and a one-line status string.

    WHY the fallback is automatic: the public Space must render something
    even without a token, and the status line tells the visitor which
    backend they are looking at.
    """
    if os.environ.get("HF_TOKEN"):
        # Lazy import keeps the HF SDK out of the import path when missing.
        from paperqa.backends.hf_inference import HFInferenceAnswerer

        return HFInferenceAnswerer(), "Answerer: HF Inference API (Llama-3.1-8B-Instruct)"
    return StubAnswerer(), "Answerer: offline stub (set HF_TOKEN for real answers)"


def _build_retriever_factory() -> tuple[RetrieverFactory, str]:
    """Pick the retriever per `PAPERQA_RETRIEVER` env (default: dense MiniLM).

    WHY env-driven (not auto-detect): ColPali requires a GPU and downloads
    a 6 GB model on first use. Visitors of the free CPU Space must opt
    into that explicitly — the default has to stay free-tier-runnable.
    """
    choice = (os.environ.get("PAPERQA_RETRIEVER") or "dense").lower()
    if choice == "colpali":
        from paperqa.retrievers.colpali import ColPaliRetriever

        def factory(pdf: Path, passages: list[Passage]) -> Retriever:
            return ColPaliRetriever(pdf_path=pdf, passages=passages)

        return factory, "Retriever: ColPali v1.3 (visual)"

    embedder = SentenceTransformerEmbedder()

    def dense_factory(_pdf: Path, passages: list[Passage]) -> Retriever:
        return DenseRetriever(passages, embedder)

    return dense_factory, "Retriever: MiniLM dense (text)"


def _format_result(result: AskResult) -> tuple[str, str]:
    """Render `AskResult` into (answer_markdown, sources_markdown)."""
    answer_md = result.answer.text
    if result.answer.citations:
        unique_pages = sorted({c.page_number for c in result.answer.citations})
        answer_md += "\n\n**Cited pages:** " + ", ".join(f"p.{p}" for p in unique_pages)

    if not result.retrieved:
        return answer_md, "_No passages retrieved._"

    blocks = []
    for hit in result.retrieved:
        excerpt = hit.passage.text.strip().replace("\n", " ")
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        blocks.append(
            f"### Page {hit.passage.page_number}  \n_score = {hit.score:.3f}_\n\n{excerpt}"
        )
    return answer_md, "\n\n---\n\n".join(blocks)


def main() -> gr.Blocks:
    retriever_factory, retriever_status = _build_retriever_factory()
    answerer, answerer_status = _build_answerer()
    qa = PaperQA(retriever_factory=retriever_factory, answerer=answerer, top_k=4)
    backend_status = f"{retriever_status} · {answerer_status}"

    def ask(pdf_file: str | None, question: str) -> tuple[str, str]:
        if not pdf_file:
            return "Upload a PDF first.", ""
        if not question.strip():
            return "Ask a question about the uploaded paper.", ""
        result = qa.ask(Path(pdf_file), question)
        return _format_result(result)

    with gr.Blocks(title="paperQA", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# paperQA\n\n"
            "Ask questions of a scientific PDF. Every answer cites the "
            "page it came from.\n\n"
            f"_{backend_status}_"
        )
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is the main contribution of this paper?",
                    lines=2,
                )
                submit = gr.Button("Ask", variant="primary")
            with gr.Column(scale=2):
                answer_output = gr.Markdown(label="Answer")
                sources_output = gr.Markdown(label="Retrieved pages")

        # WHY chain two clicks: the first call returns immediately and only
        # updates the UI to the "thinking" state (button disabled, answer
        # panel shows a spinner-style message). The second call runs the
        # actual ask() and re-enables the button when done. Without this,
        # the user clicks Ask, the LLM call takes 5-8 seconds, and the UI
        # gives no feedback — the entire reason for adding this.
        submit.click(
            lambda: (
                gr.update(value="Ask", interactive=False),
                "⏳ Retrieving passages and asking the model… "
                "(first request after a cold start can take 20+ seconds)",
                "",
            ),
            inputs=None,
            outputs=[submit, answer_output, sources_output],
        ).then(
            ask,
            inputs=[pdf_input, question_input],
            outputs=[answer_output, sources_output],
        ).then(
            lambda: gr.update(value="Ask", interactive=True),
            inputs=None,
            outputs=submit,
        )

    # WHY cast: gradio ships no type stubs so `gr.Blocks()` narrows to Any.
    # The cast preserves the strict-mypy promise without pretending to type
    # third-party internals.
    return cast("gr.Blocks", demo)


if __name__ == "__main__":
    main().launch()
