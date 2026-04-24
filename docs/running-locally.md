# Running paperQA locally

> **Status:** partial — the PDF chunker is runnable. No UI or end-to-end pipeline yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"              # core + dev tooling
pip install -e ".[embed]"             # add real embedder (optional)
```

The `embed` extra pulls sentence-transformers (~500 MB of transitive deps).
Skip it if you only want to run the unit tests — the index layer tests use a
deterministic fake embedder.

## Run the tests

```bash
pytest -q
```

## Use the chunker from the REPL

```python
from paperqa import chunk_by_page

passages = chunk_by_page("path/to/paper.pdf")
for p in passages[:3]:
    print(p.page_number, p.text[:80])
```

## Embed and query a paper

Requires the `embed` extra.

```python
from paperqa import PassageIndex, chunk_by_page
from paperqa.embedders import SentenceTransformerEmbedder

passages = chunk_by_page("paper.pdf")
embedder = SentenceTransformerEmbedder()
index = PassageIndex.build(passages, embedder)

for hit in index.query("What is the main contribution?", embedder, top_k=4):
    print(f"[p{hit.passage.page_number} score={hit.score:.3f}] "
          f"{hit.passage.text[:120]}")
```

## End-to-end with the offline stub answerer

No network, no inference token — useful for local smoke tests:

```python
from paperqa import PassageIndex, StubAnswerer, chunk_by_page
from paperqa.embedders import SentenceTransformerEmbedder

passages = chunk_by_page("paper.pdf")
embedder = SentenceTransformerEmbedder()
index = PassageIndex.build(passages, embedder)

hits = index.query("What is the main contribution?", embedder, top_k=4)
answer = StubAnswerer().answer("What is the main contribution?", hits)

print(answer.text)
for c in answer.citations:
    print(f"  cited page {c.page_number} (score={c.score:.3f})")
```

## End-to-end with the real HF Inference API

Requires the `llm` extra and an HF token (export `HF_TOKEN=...`):

```python
from paperqa import PassageIndex, chunk_by_page
from paperqa.embedders import SentenceTransformerEmbedder
from paperqa.backends.hf_inference import HFInferenceAnswerer

passages = chunk_by_page("paper.pdf")
embedder = SentenceTransformerEmbedder()
index = PassageIndex.build(passages, embedder)

hits = index.query("What is the main contribution?", embedder, top_k=4)
answer = HFInferenceAnswerer().answer("What is the main contribution?", hits)

print(answer.text)
for c in answer.citations:
    print(f"  cited page {c.page_number} (score={c.score:.3f})")
```

`HFInferenceAnswerer` satisfies the same `Answerer` protocol as
`StubAnswerer` — it is a drop-in replacement (see
[ADR-0004](adr/0004-answering-model-and-backend.md)).

## Running integration tests

Default `pytest` runs skip `integration`-marked tests. To exercise the real
HF Inference API:

```bash
export HF_TOKEN=hf_...
pytest -m integration
```

## Regenerate test fixtures

The committed `tests/fixtures/three_pages.pdf` is deterministic. To rebuild it:

```bash
pip install reportlab
python scripts/make_test_fixtures.py
```

## Prerequisites

- Python 3.11+
- No GPU required (v1 stages added so far are CPU-only).

This file is updated in the same commit that introduces each new setup step, per [CONTRIBUTING.md §4](../CONTRIBUTING.md).
