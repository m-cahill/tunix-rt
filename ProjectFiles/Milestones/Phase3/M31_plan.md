According to a document from **December 25, 2025**, **M31** is explicitly the **“Final Submission Package”** milestone, with acceptance criteria: **video uploaded, artifacts packaged, Kaggle notebook runs**. 

From the competition side, the key constraints to design around in M31:

* The hackathon is built around **Gemma 2 2B / Gemma 3 1B** and a **single-session Kaggle workflow**. ([Kaggle][1])
* The **submission video** is expected to be **≤ 3 minutes** and uploaded (then linked via Kaggle’s Media Gallery). ([Kaggle][1])
* The public rubric emphasizes **Notebook quality + Model quality + Video quality** (with an optional multi-session component). ([Kaggle][1])
* The competition listing also shows the **end date / deadline window** you should treat as the “real clock” for M31. ([Kaggle][1])

Your repo is already “competition-ready” in the engineering sense (Kaggle notebook path exists, dry-run verified, submission checklist exists, dataset(s) exist). M31 is now **pure packaging + narrative + a final reproducible run**.

---

## Prompt to handoff to Cursor (M31)

```text
You are Cursor acting as a senior engineer. Implement Milestone M31: “Final Submission Package”.

Context
- Repo: tunix-rt
- Current status: CI green; M30 closed; Kaggle dry-run + submission checklist already exist.
- M31 acceptance criteria (must hit all): (1) Kaggle notebook runs from scratch, (2) artifacts packaged cleanly, (3) 3-minute video delivered (script + final link placeholder in docs).
- IMPORTANT: keep changes small and safe; do not introduce new features unless directly required for submission packaging.

Branch / Baseline Gate (Phase 0)
1) Create branch: milestone/M31-final-submission-package from main.
2) Verify baseline is clean:
   - backend: uv run ruff format --check . && uv run ruff check . && uv run mypy .
   - backend: uv run pytest
   - frontend: npm test
   - e2e: run the usual e2e command used in CI
3) If anything fails, fix it immediately before proceeding (no “we’ll do it later”).

Phase 1 — “Submission Freeze” + Versioning
Goal: make a crisp, reproducible snapshot.
1) Add a “submission freeze” marker:
   - Create docs/submission_freeze.md with:
     - commit SHA placeholder
     - dataset key used (dev-reasoning-v1 vs golden-v2)
     - exact training config file used
     - exact evaluation command used
     - exact artifact bundle name produced
2) Ensure a single, stable “submission entrypoint” exists:
   - If notebooks/kaggle_submission.ipynb is the canonical entry: confirm it runs end-to-end with no manual edits.
   - If notebooks/kaggle_submission.py is more stable: keep both, but make the ipynb simply a thin wrapper around the .py.
3) Add a lightweight version stamp:
   - Create a constant SUBMISSION_VERSION = "m31_v1" in an appropriate place (docs or notebook metadata), not in core runtime code.

Phase 2 — Kaggle Notebook “From Scratch” Hardening
Goal: notebook executes cleanly in a fresh Kaggle session.
1) Audit notebooks/kaggle_submission.ipynb for:
   - deterministic seed usage (ensure the notebook sets seed in one place)
   - no hidden local filesystem assumptions
   - paths correct for Kaggle working dir
2) Add a “smoke mode” cell near the top:
   - runs only 1–2 steps / tiny batch
   - confirms imports, dataset load, checkpoint write, and a tiny eval output
3) Add a “full run” cell:
   - clearly bounded by max_steps and time budget
   - emits:
     - metrics.jsonl
     - final metrics summary
     - checkpoint path (orbax)
     - predictions output (if that’s part of your flow)
4) Ensure the notebook prints a final “Submission Summary” section:
   - model id
   - dataset key
   - steps trained
   - eval score
   - artifact paths

Phase 3 — One-Command Artifact Packaging
Goal: produce a single zip/tarball with everything a judge or teammate needs.
1) Create backend/tools/package_submission.py (or tools/package_submission.py; pick the repo’s convention) that:
   - creates ./submission/ (gitignored)
   - bundles:
     - notebooks/kaggle_submission.ipynb
     - notebooks/kaggle_submission.py
     - docs/kaggle_submission.md
     - docs/submission_checklist.md
     - docs/submission_freeze.md
     - training config yaml used
     - dataset manifest(s) used (backend/datasets/<name>/manifest.json)
     - a small README in the bundle explaining “how to run”
   - names the archive like: submission/tracehammer_m31_<YYYY-MM-DD>_<shortsha>.zip
2) Add docs/submission_artifacts.md describing:
   - what’s in the zip
   - why each artifact exists
   - how to reproduce the run

Phase 4 — Video Script + Shot List (no actual recording in CI)
Goal: give Michael a ready-to-record, <=3 minute video plan.
1) Create docs/video_script_m31.md with:
   - 150–170 seconds max spoken script (target 2:30 to be safe)
   - required beats:
     - what the project is
     - what “trace-first” means (show one trace)
     - how reproducibility works (seed + manifest + artifact bundle)
     - quick results (eval score + what it demonstrates)
     - how to run the notebook
2) Create docs/video_shotlist_m31.md with timestamps and what to show on screen.
3) In docs/submission_checklist.md:
   - add a “Video” section with a placeholder field for the final YouTube URL and Kaggle Media Gallery confirmation.

Phase 5 — Final Local Verification
1) Run the packaging tool and confirm the archive is created.
2) Do a clean-room style test locally:
   - unzip the archive into a temp folder
   - confirm the notebook and docs are present and referenced paths make sense
3) Update docs/submission_freeze.md with:
   - the archive name
   - commit SHA
   - dataset key
   - training config path

Phase 6 — PR Hygiene
1) Keep PR small and focused.
2) Ensure ruff format is run on any touched files.
3) CI must be green before merge.

Deliverables Checklist (must be true)
- Kaggle submission notebook runs from scratch (smoke + full).
- One-command packaging tool produces a single archive.
- Video script + shot list exist and are <= 3 minutes.
- Submission checklist updated with final placeholders.
- CI green.

After implementation, provide:
- List of files changed/added
- Exact commands to run packaging + smoke run
- Any assumptions you made (Kaggle paths, dataset choice, etc.)
```

---

## What the next few milestones likely look like (after M31)

Your own audit already sketches the sequence:

* **M31: Final Submission Package** (what we’re doing now) 
* **M32: Data Scale-Up** (seed/ingest a **500+ trace dataset**, re-verify training) 
* **M33: Final Training Sweep** (pick best config/model by eval; lock the best checkpoint) 

That matches the rubric reality: once the notebook + video are polished, **incremental model gains come from dataset scale + a small sweep**, not architecture churn. ([Kaggle][1])

[1]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
