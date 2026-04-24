# Deploying to Hugging Face Spaces

paperQA's public demo runs as a Gradio Space. This runbook is the source of truth for how to deploy it — update this file in the same commit as any deploy-affecting change.

## Prerequisites

- A Hugging Face account.
- The `huggingface_hub` CLI installed and authenticated (`huggingface-cli login`).
- A read-access token exported as `HF_TOKEN` if you want the Space to use the real Inference API backend. Without the token the Space runs the offline `StubAnswerer`.

## One-time Space creation

1. Create a **Gradio** Space on https://huggingface.co/new-space. SDK: `gradio`. Hardware: CPU basic is enough for v1.
2. Clone the Space repo locally:

   ```bash
   huggingface-cli repo clone spaces/<your-username>/paperqa
   cd paperqa
   ```

3. Add this GitHub repo as an additional remote so changes can be pushed from here:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/paperqa
   ```

4. Add the Space-specific metadata header to `README.md` (HF requires it at the top of the Space's README, but the GitHub repo's README does not have it):

   ```yaml
   ---
   title: paperQA
   emoji: 📄
   colorFrom: blue
   colorTo: indigo
   sdk: gradio
   sdk_version: "4.44.0"   # bump when the installed gradio major/minor changes
   app_file: app.py
   pinned: false
   ---
   ```

   Keep a separate copy of this header in `docs/space-readme-header.md` in this repo so it survives deploys; see below.

## Deploy

From the GitHub repo's checkout:

```bash
git push space main
```

HF builds the Space, installs `requirements.txt`, and runs `app.py`. Cold builds take 3–5 minutes.

## Configuring the Inference API backend

1. In the Space's **Settings → Variables and secrets**, add `HF_TOKEN` as a **secret**.
2. Restart the Space.
3. The app status line changes from `Backend: offline stub (...)` to `Backend: HF Inference API (...)`.

If the token is wrong or rate-limited, the app still renders but generation fails. The UX path for that is handled in `app.py` (the Inference call throws → Gradio surfaces the traceback). Tightening that error path is tracked as a follow-up.

## Updating a live Space

Any code change merged to `main` on GitHub:

```bash
git push space main
```

That is the whole flow. There is no CI for the Space build itself — GitHub Actions run against GitHub, and the Space builds on push to the Space remote.

## Space-specific README header

The Space's README needs the YAML frontmatter above. This repo's README does not (it would confuse GitHub's renderer). The canonical copy lives in [`space-readme-header.md`](space-readme-header.md); prepend it to the Space's `README.md` on first deploy, then rely on `git pull space main` to keep updates in sync if they drift.
