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
