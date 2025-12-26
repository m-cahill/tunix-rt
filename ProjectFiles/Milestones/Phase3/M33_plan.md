According to your M32 summary/audit, you now have (a) a scaled dataset `dev-reasoning-v2` (550 traces with a defined composition), (b) strengthened evidence-capture + packaging (`package_submission.py` now includes run manifest + eval summary + kaggle log), and (c) improved CI/coverage posture. That sets you up for the first **true competition-grade run** and a repeatable “submission rehearsal” loop.

Below is the **Cursor handoff prompt for M33** (next milestone), designed to be small, end-to-end verifiable, and aligned with the competition timeline (Final Submission: **Jan 12, 2026**) and Kaggle compute realities (e.g., TPU session limits; keep outputs <~1K tokens). ([Kaggle][1])

---

## Prompt to handoff to Cursor — Milestone M33: Kaggle “Submission Rehearsal” Run v1 + Evidence Lock

### Goal

Produce a **single, fully reproducible Kaggle execution** (train → eval → artifacts → packaged zip) using the current pipeline + datasets, with **evidence files filled in** and stored in-repo, and a “known-good” submission archive emitted.

This milestone is explicitly about **proof**, not optimization.

### Why now

M32’s “Next Steps (M33+)” explicitly calls for running on Kaggle, capturing logs, and filling evidence templates—that’s the next gating item before serious tuning/iteration.

---

## Phase 0 — Baseline gate (must be green first)

1. Create branch: `milestone/M33-kaggle-rehearsal-v1` from `main`.
2. Verify locally:

   ```bash
   make test
   # or:
   cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy tunix_rt_backend && uv run pytest
   cd ../frontend && npm test
   cd ../e2e && npx playwright test
   ```
3. **Guardrail**: before any push, run:

   ```bash
   cd backend && uv run ruff check . && uv run ruff format .
   ```

**DoD (Phase 0):** local + CI green.

---

## Phase 1 — Kaggle execution path: make it truly “one-click runnable”

### 1.1 Validate model + deps assumptions

* Ensure notebook/script uses a **competition-allowed Gemma** model (Gemma 2 2B or Gemma 3 1B). (Prefer Gemma 3 1B for iteration speed unless you already have stable 2B runs.) ([Hugging Face][2])
* Confirm Tunix positioning (library exists; optional for your repo), but for M33 the target is: “Kaggle run completes + produces artifacts.” ([Google Developers Blog][3])

### 1.2 Notebook robustness

Fix/confirm the Kaggle notebook uses **Python-native invocation** (e.g., `subprocess.run`) instead of brittle shell interpolation.

**Acceptance:** running the notebook top-to-bottom in a fresh Kaggle session succeeds without manual patching.

---

## Phase 2 — Run v1 on Kaggle with dev-reasoning-v2 + eval_v1

### 2.1 Choose the exact “rehearsal recipe”

* Dataset: `dev-reasoning-v2` (550 traces)
* Eval set: whichever your submission docs currently designate (stick to the canonical one already in repo)
* Output token discipline: keep generated trace length within reasonable limits (competition guidance suggests <1K output tokens is acceptable under constraints). ([I Programmer][4])

### 2.2 Capture evidence artifacts

Create a new run folder:

* `submission_runs/m33_v1/`

  * `run_manifest.json`
  * `eval_summary.json`
  * `kaggle_output_log.txt` (raw cell output or captured stdout/stderr)
  * any small “provenance” notes you already template

**Important:** Track these files in git (they’re small and explicitly called out as evidence in M32).

**DoD (Phase 2):** you can point to a Kaggle run log + saved evidence files that prove the run completed and evaluation executed.

---

## Phase 3 — Package the submission zip from the exact run evidence

1. Run packaging:

   ```bash
   python tools/package_submission.py --run-dir submission_runs/m33_v1
   ```
2. Output archive naming should match whatever your current convention is (repo/project aligned), and include:

   * notebook/script used
   * configs used
   * evidence files
   * “how to reproduce” doc pointer

**DoD (Phase 3):** a single zip exists under `submission/` that a third party could unzip + follow docs to reproduce.

---

## Phase 4 — Regression guardrails (small but critical)

Add a **tiny CI-safe test** that prevents “Kaggle rehearsal drift”:

* Validate `submission_runs/m33_v1/run_manifest.json` exists + schema-valid
* Validate `eval_summary.json` exists + contains required keys
* Validate packaging tool includes these files (unit test the file list)

This avoids the classic failure mode: “works once, evidence not captured, archive missing the proof.”

**DoD (Phase 4):** new tests added, CI green, and failing behavior is explicit if evidence files are missing.

---

## Deliverables checklist (M33)

* [ ] CI green
* [ ] Kaggle notebook runs top-to-bottom successfully
* [ ] `submission_runs/m33_v1/{run_manifest.json, eval_summary.json, kaggle_output_log.txt}` committed
* [ ] `submission/…m33…zip` produced locally
* [ ] One-page doc update: “M33 rehearsal run — exact command + exact artifacts” (link to evidence files)

---

# After M33: what the next milestones look like (high level)

**M34 (Optimization Loop 1):** 5–20 trial Ray Tune sweep on *dev-reasoning-v2* using your existing M19 tuning infra, selecting only a small hyperparam set (LR, batch size, steps, max_tokens).
**M35 (Quality Loop 1):** tighten eval aggregation + leaderboard scoring + run comparison UI so you can see improvements quickly.
**M36 (If needed for judges):** optional “real Tunix” training-path integration (minimal SFT via Tunix APIs) while keeping your trace-first orchestration intact. ([Google Developers Blog][3])

---

If you want, I can also generate a **ready-to-paste `M33_TODO.md`** with exact file paths + stubbed JSON schemas for `run_manifest.json` / `eval_summary.json` so Cursor can execute with minimal ambiguity.

[1]: https://www.kaggle.com/competitions/google-tunix-hackathon/overview/timeline?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[2]: https://huggingface.co/google/gemma-3-1b-it/tree/main?utm_source=chatgpt.com "google/gemma-3-1b-it at main"
[3]: https://developers.googleblog.com/introducing-tunix-a-jax-native-library-for-llm-post-training/?utm_source=chatgpt.com "Introducing Tunix: A JAX-Native Library for LLM Post-Training"
[4]: https://www.i-programmer.info/news/204-challenges/18460-google-tunix-hack-hackathon-now-open.html?utm_source=chatgpt.com "Google Tunix Hack Hackathon Now Open"
