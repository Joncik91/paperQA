# paperQA

[![ci](https://github.com/Joncik91/paperQA/actions/workflows/ci.yml/badge.svg)](https://github.com/Joncik91/paperQA/actions/workflows/ci.yml)

Ask questions of scientific papers. Get answers grounded in the source, with page-level citations.

## What

A document-QA system specialised for arXiv-style scientific PDFs. Upload a paper (or paste an arXiv ID), ask a natural-language question, receive an answer with the exact page and passage it came from.

## Why

Reading papers is slow. Generic chatbots hallucinate citations. paperQA closes the gap by grounding every answer in retrievable page regions from the source PDF — no invented references, no off-document text.

## How it works

1. **Ingest** — PDF is parsed (text + layout) and split into page-level passages
2. **Index** — passages are embedded and stored for retrieval
3. **Retrieve** — user question is embedded, top-k passages fetched
4. **Answer** — a vision-language model reads the retrieved pages and answers, citing page numbers

Model and retrieval choices are documented in [docs/adr/](docs/adr/).

## Status

Pre-alpha. See [docs/adr/0001-scope-and-model-choice.md](docs/adr/0001-scope-and-model-choice.md) for current scope.

## Running locally

See [docs/running-locally.md](docs/running-locally.md). Current slice: PDF → page-level `Passage` list.

## Demo

TBD — deployed to Hugging Face Spaces (link once live).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Hard rules on comment style, commit messages, and docs-with-code apply.

## License

MIT — see [LICENSE](LICENSE).
