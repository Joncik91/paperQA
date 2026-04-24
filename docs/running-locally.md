# Running paperQA locally

> **Status:** placeholder. No runnable entry point exists yet. This page will be filled in as soon as a first runnable slice lands (see ADR-0002 and subsequent).

## Planned setup

```bash
# Once the first runnable slice lands:
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# Entry-point command TBD.
```

## Prerequisites (anticipated)

- Python 3.11+
- ~2 GB free disk for cached model weights (Nougat base)
- An internet connection on first run (downloads weights from Hugging Face Hub)

This file is updated in the same commit that introduces each new setup step, per [CONTRIBUTING.md §4](../CONTRIBUTING.md).
