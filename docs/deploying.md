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

3. **First push — orphan branch (HF rejects PDF binaries from regular git).**

   Hugging Face's pre-receive hook rejects any binary file (e.g. our 2.2 MB `tests/fixtures/attention_is_all_you_need.pdf`) from a regular git push, even at small sizes. Two ways forward:

   1. Migrate the PDF to LFS — adds a hard `git-lfs` clone-time dep for both remotes. Rejected.
   2. Push only the **runtime** tree (no `tests/`) on a fresh orphan branch. Adopted.

   Procedure (re-run from `main` every time the Space needs an update):

   ```bash
   git checkout --orphan space-deploy
   git rm -rf --cached .
   rm -rf tests/                                   # tests are not runtime
   git add app.py requirements.txt README.md LICENSE \
           .gitattributes .gitignore pyproject.toml src/ docs/
   git commit -m "deploy: paperQA Space runtime tree"

   # Use a write-scope HF token. Inline-in-URL keeps the token out of
   # `git config` (where a credential helper would otherwise persist it).
   git push -f "https://<user>:${HF_WRITE_TOKEN}@huggingface.co/spaces/<user>/paperqa" \
     space-deploy:main

   git checkout main
   git branch -D space-deploy                      # disposable
   ```

   Burn the token after the push if it was a one-shot.

   Cold builds on CPU basic take 3–5 minutes (downloads sentence-transformers on first run).

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

Re-run the orphan-branch procedure above. There is no CI for the Space build itself — GitHub Actions run against GitHub, and the Space builds on each force-push to its `main`. Worth scripting once we update more than once a week.

## Frontmatter and gradio version

The YAML block at the top of the repo's `README.md` is the Space's metadata. When the installed `gradio` major/minor version changes, bump `sdk_version` there in the same commit as the dep bump. Mismatches between `sdk_version` and what `requirements.txt` actually installs cause silent UI quirks on the Space.
