# Deploying to Hugging Face Spaces

paperQA's public demo runs as a Gradio Space. This runbook is the source of truth for how to deploy it — update this file in the same commit as any deploy-affecting change.

## Where it lives

- **Live demo:** https://huggingface.co/spaces/Joncik/paperqa
- **Source:** this GitHub repo. The Space is a sibling git remote (`space`) pushed from this checkout.
- The Space's `README.md` is *the same file* as this repo's `README.md` — its YAML frontmatter at the top configures the Space (title, emoji, gradio version, app entrypoint). GitHub renders the frontmatter as a quiet metadata block and otherwise ignores it; HF parses it.

## Prerequisites

- A Hugging Face account.
- A token with **write** access to your Space (https://huggingface.co/settings/tokens). Required for `git push` to the Space remote.
- A token with **read** access exported as `HF_TOKEN` in the Space's secrets if you want the real Inference API backend. Without it the Space falls back to the offline `StubAnswerer`.

## One-time Space creation

1. Create a **Gradio**, **Blank** Space at https://huggingface.co/new-space. SDK: `gradio`. Hardware: CPU basic. Storage: none. Visibility: public.
2. From this repo's checkout, add the Space as a git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/paperqa
   ```

3. First push:

   ```bash
   git push space main
   ```

   HF builds the Space, installs `requirements.txt`, and runs `app.py`. Cold builds take 3–5 minutes on a CPU basic Space.

## Configuring the Inference API backend

1. In the Space's **Settings → Variables and secrets**, add `HF_TOKEN` as a **secret** (not a public variable).
2. Restart the Space (Settings → "Factory rebuild" or `Restart Space`).
3. The app's status line changes from `Answerer: offline stub (...)` to `Answerer: HF Inference API (Llama-3.1-8B-Instruct)`.

If the token is wrong or rate-limited, the app still renders but generation fails. The UX path for that is handled in `app.py` (the Inference call throws → Gradio surfaces the traceback). Tightening that error path is tracked as a follow-up.

## Switching to ColPali (paid GPU Space)

Per [ADR-0006](adr/0006-visual-retrieval-colpali.md), the visual retriever needs a GPU. To deploy the ColPali path:

1. Create the Space on **GPU hardware** (T4 small minimum) — set this in **Settings → Hardware** before the first build.
2. In **Settings → Variables and secrets**, add `PAPERQA_RETRIEVER=colpali` as a (non-secret) variable.
3. Add `colpali-engine>=0.3,<0.4` and `pypdfium2>=4,<5` to `requirements.txt` on the Space (these are *not* in the default `requirements.txt` because the free CPU Space cannot use them).
4. Restart the Space. First load downloads ~6 GB of weights; expect a long cold start.

The CPU Space and the GPU Space can coexist as separate Spaces — the same GitHub repo backs both, with the deploy variant chosen by env var.

## Updating a live Space

Any code change merged to `main` on GitHub:

```bash
git push space main
```

That is the whole flow. There is no CI for the Space build itself — GitHub Actions run against GitHub, and the Space builds on push to the Space remote.

## Frontmatter and gradio version

The YAML block at the top of the repo's `README.md` is the Space's metadata. When the installed `gradio` major/minor version changes, bump `sdk_version` there in the same commit as the dep bump. Mismatches between `sdk_version` and what `requirements.txt` actually installs cause silent UI quirks on the Space.
