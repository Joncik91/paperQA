# Baselines

Snapshots of `scripts/run_eval.py` output captured at meaningful checkpoints.
Each file is named `YYYY-MM-DD-<tag>.md` and documents the setup (PDF, gold
set, embedder, answerer, `top_k`) alongside the metrics, so future changes
can be compared apples-to-apples.

## Index

- [2026-04-25 — baseline on Attention Is All You Need (StubAnswerer)](2026-04-25-baseline.md)
- [2026-04-25 — real-LLM baseline (Llama-3.1-8B-Instruct via HF Inference)](2026-04-25-real-llm-baseline.md)
- [2026-04-25 — sharpened prompt (must-cite +0.333, faithfulness −0.167)](2026-04-25-sharpened-prompt.md)
- [2026-04-25 — + grounding check (must-cite 0.833, faithfulness 0.917)](2026-04-25-grounding-check.md)
- [2026-04-25 — + Qwen2.5-7B (must-cite 0.833, faithfulness **1.000**) ✨ current default](2026-04-25-qwen.md)
- [2026-04-25 — multi-doc smoke (pooled-index across two PDFs)](2026-04-25-multi-doc.md)
