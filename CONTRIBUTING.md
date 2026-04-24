# Contributing to paperQA

This project has **hard rules**. They exist because the repo is also a portfolio piece: the quality of the commits and comments is as visible as the code. Follow them or the change does not land.

---

## 1. DRY — Don't Repeat Yourself

- **Second occurrence extracts.** The first time a pattern appears, inline is fine. The second time, extract into a function, class, or constant.
- **No copy-paste.** If you find yourself copying a block, stop and extract instead.
- **Shared config over hard-coded literals.** Thresholds, paths, model names live in one place.
- **One source of truth per fact.** Version strings, URLs, schema definitions, etc. are declared once and imported elsewhere.

Exceptions require a one-line comment explaining why the duplication is load-bearing.

---

## 2. Code comments — WHAT + WHY, never HOW

The code *is* the HOW. Comments add what the code cannot say.

**WHAT** — a one-line summary of intent, when the function name alone is not enough.
**WHY** — the non-obvious reason behind a choice, constraint, or workaround.

Good:
```python
# WHAT: skip embedded figures — they confuse the text chunker.
# WHY: Nougat emits alt-text that duplicates the caption downstream.
```

Bad:
```python
# Loop over pages
for page in pages:   # increment page counter
    ...
```

Rules:
- **No HOW comments.** If readers need a comment to understand the mechanics, rewrite the code instead.
- **No narration.** Do not describe what the next line does.
- **No dead comments.** Remove, don't comment out.
- **Docstrings for public API only.** One-liner minimum; add Args/Returns only when types don't already say it.

---

## 3. Commit messages — WHAT + WHY + WHERE

Every commit answers three questions. Subject line is a short WHAT. Body expands WHAT, WHY, and WHERE.

**Subject (≤72 chars, imperative):**
```
add page-level PDF chunker
```

**Body:**
```
WHAT: Introduce src/paperqa/chunking.py with chunk_by_page(pdf_path) -> list[Passage].
WHY:  Retrieval needs page-aligned passages so citations can point at a specific page.
      Sentence-level chunking loses that mapping.
WHERE: src/paperqa/chunking.py (new), src/paperqa/__init__.py (export),
       tests/test_chunking.py (new), docs/adr/0002-chunking-strategy.md (new).
```

Rules:
- One logical change per commit. Don't mix refactor + feature + doc in one.
- **Docs land in the same commit as the code they describe** (see §4).
- No "WIP", no "fix stuff", no emoji, no Claude/Copilot co-author tags on this repo.
- Reference ADRs in the body when a commit implements one: `Implements ADR-0003.`

Template lives at `.gitmessage` — configured via `git config commit.template .gitmessage`.

---

## 4. Docs updated continuously

Documentation is not a phase, it is a line in every commit.

- A change that adds/alters public behavior **must** update the affected doc in the same commit.
- Architectural decisions are recorded as ADRs in `docs/adr/NNNN-short-title.md` using the template at `docs/adr/TEMPLATE.md`. ADR number is monotonic; never reuse.
- `README.md` reflects *current* state, not aspiration. If a feature is not yet working, it does not appear in README.
- Runbooks (`docs/running-locally.md`, `docs/deploying.md`) are updated the moment their commands change.

A commit with code changes and no matching doc changes will be rejected in review — unless the change is genuinely doc-invisible (internal refactor, test-only change, type annotations).

---

## 5. Testing

- New logic ships with tests. Bug fixes ship with a regression test that fails before the fix.
- `pytest` must pass locally before commit. CI enforces the same.
- Tests live under `tests/` mirroring `src/paperqa/` structure.

---

## 6. Tooling

- `ruff check . && ruff format .` before commit.
- `mypy src/` must pass (strict mode).
- Python ≥ 3.11.

---

## 7. Pull requests

Not applicable while this is a solo repo. When opened to contributors, the PR description mirrors the commit body format: WHAT, WHY, WHERE.
