## Where you stand vs the competition (as of M33)

You’ve now built something many teams *won’t* have: a **reproducible, evidence-locked rehearsal path** (run manifests + eval summaries + packaging + CI tests). That directly supports the “judge-scored” reality of this hackathon: you’re being evaluated on *how well you can demonstrate a trace-first training + evaluation workflow*, not just raw model score. ([Kaggle][1])

Two practical constraints to keep front-of-mind as you move from smoke → real runs:

* Kaggle sessions and quotas are constrained (e.g., session duration + weekly accelerator hours). ([I Programmer][2])
* Output length / “show your work” requirements can become a failure mode if traces balloon. Build guardrails around trace length early. ([I Programmer][2])

**Data quality:** what you’re collecting is “structurally good” (schema-valid, ingestible, exportable, versionable), which is the main risk early on. The next risk is *distribution + judge alignment*: your dev sets should increasingly resemble the kinds of prompts the hidden eval set is likely to contain, while keeping trace verbosity under control. ([I Programmer][2])

---

# Recommended next milestone: **M34 — Optimization Loop 1 (Small Tune Sweep + Scoreboard + Rehearsal v2)**

Below is a Cursor-ready prompt. It’s intentionally **small, end-to-end verifiable**, and keeps you moving toward “real Kaggle execution” without getting stuck in perfectionism.

---

## Prompt to handoff to Cursor (M34)

### Title

**M34 — Optimization Loop 1: Ray Tune Mini-Sweep + Primary Score + Rehearsal v2 Evidence**

### Goal

Improve “model quality” signal **without breaking reproducibility**, by:

1. defining a single **primary_score** aggregation used everywhere,
2. running a **small Ray Tune sweep** (5–20 trials) using existing M19 infrastructure,
3. promoting the best params into a “submission-ish” config, and
4. producing a new evidence folder `submission_runs/m34_v1/` + updated package zip.

### Non-Goals (explicit)

* No big architectural refactors.
* No “full Kaggle run” required inside CI (human-run later).
* No major dataset redesign; only *incremental* quality guardrails.

---

## Phase 0 — Baseline Gate (must pass first)

**Branch:** `milestone/M34-optimization-loop-1` off `main`.

**Must be green locally before pushing:**

* Backend: `uv run ruff check . && uv run ruff format --check . && uv run mypy tunix_rt_backend && uv run pytest`
* Frontend: `npm test && npm run build`
* E2E: run the standard Playwright command already used in repo (keep unchanged)

**Guardrail:** ensure pre-commit runs `ruff check --fix` **then** `ruff format` to avoid recurring CI format failures.

---

## Phase 1 — Define “primary_score” once (and reuse everywhere)

### Deliverables

1. Add a small backend scoring module, e.g.:

* `backend/tunix_rt_backend/scoring/primary_score.py`

2. Implement:

* `compute_primary_score(evaluation_rows) -> float | None`
* Default formula: **mean(answer_correctness)** over the eval set (ignore nulls).
* Return `None` if no valid rows.

3. Wire it into:

* evaluation responses returned by `/api/tunix/evaluations` (or wherever eval results are served)
* evidence writer used for `submission_runs/mXX_vY/eval_summary.json` so it always includes:

  * `metrics` (raw)
  * `primary_score` (computed)

### Tests (required)

* Unit tests for:

  * all-correct, mixed, empty, null correctness
  * stability of rounding/serialization (don’t over-round; store float)

**Done when:** `primary_score` appears in API + evidence and tests prove it.

---

## Phase 2 — Mini Ray Tune sweep (5–20 trials, smoke-limited)

### Deliverables

1. Add a tuning preset script/config:

* `backend/tools/run_tune_m34.py` (or similar)

2. Use existing M19 endpoints/service:

* Create a tuning job with a small search space (example):

  * learning_rate: loguniform
  * weight_decay: uniform
  * warmup_steps: choice
  * batch_size: choice (within memory constraints)
  * optionally: max_steps / smoke_steps (keep short)

3. Enforce a **hard cap** so each trial is cheap:

* use `--smoke_steps` or an equivalent short training mode
* dataset: `dev-reasoning-v2` (default)
* eval set: existing lightweight eval (whatever repo uses)

4. Persist results:

* ensure each trial stores params + resulting `primary_score`

### Tests (required)

* A mocked Ray Tune unit test that asserts:

  * job created
  * at least one “trial result” recorded with `primary_score`
* No real training required in tests.

**Done when:** one command can run a 5-trial sweep locally and the UI/API can show trial scores.

---

## Phase 3 — Promote “best trial” → config + Rehearsal v2 artifacts

### Deliverables

1. Create a config file that represents “best known params”:

* `training/configs/m34_best.yaml` (or similar)
* It should be derived from best trial params.

2. Create evidence folder:

* `submission_runs/m34_v1/`

  * `run_manifest.json` (filled with commit sha, model_id, dataset, config_path, command)
  * `eval_summary.json` (filled, includes primary_score)
  * `kaggle_output_log.txt` (local run output is acceptable for M34)

3. Update notebook default **only if** it’s clearly beneficial:

* If changing defaults, bump notebook run_version → `m34_v1`.

4. Run packaging tool:

* `python backend/tools/package_submission.py --run-dir submission_runs/m34_v1`
* Output: `submission/tunix_rt_m34_<date>_<sha>.zip`

### Tests (required)

* Add/extend evidence schema tests to include m34 folder existence + required keys.
* Packaging tool test confirms evidence files are included.

**Done when:** repo contains `submission_runs/m34_v1/*` and the zip builds locally.

---

## Phase 4 — Small UI “quick win” (if time allows, but keep it tight)

Add one improvement that helps judge narrative and internal iteration:

**Option A (preferred):** Tuning results table shows `primary_score` and can “promote params → config” (even if that promotion is a manual copy button).

**Option B:** Run history list highlights `primary_score` and links to evidence files.

**Done when:** UI change is covered by a minimal unit test and does not break E2E.

---

## Definition of Done (M34)

* CI green.
* `primary_score` is computed consistently and appears in API + evidence.
* A 5–20 trial smoke sweep can be executed locally using M19 tuning infra.
* `submission_runs/m34_v1` is committed with required evidence files.
* `submission/tunix_rt_m34_*.zip` can be produced locally with evidence included.

---

## Why this is the right “next step”

This pushes you toward the *actual judged deliverables* (credible evidence + clear scoring + iteration discipline) while building real performance signal safely, inside the constraints of the hackathon environment. ([I Programmer][2])

[1]: https://www.kaggle.com/competitions/google-tunix-hackathon/discussion/651560?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[2]: https://www.i-programmer.info/news/204-challenges/18460-google-tunix-hack-hackathon-now-open.html?utm_source=chatgpt.com "Google Tunix Hack Hackathon Now Open"
