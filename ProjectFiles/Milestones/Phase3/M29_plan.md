M28 put you in a really strong place technically: CI is mature, tuning/comparison/leaderboard are real, and the audit’s *only* “medium” structural concern is still **`app.py` size creep**, plus a handful of small hygiene items.  

For **M29**, I’d pivot from “more features” to **competition readiness + maintainability**:

* **Competition readiness:** Kaggle’s hackathon is judged and has a fixed timeline (Final Submission **Jan 12, 2026**; judging Jan 13–23). ([Kaggle][1])
* **Single-session constraint pressure:** Kaggle TPU sessions are time/weekly capped (commonly discussed as **~9 hours per session** and **~20 hours/week** in the official discussion/FAQ context). ([Kaggle][2])
* **Your current data:** great for validating pipelines (golden-v2 style), but **not yet enough** for “model quality” unless you scale diversity/coverage of reasoning formats and tasks.

So M29 should do 3 things:

1. **Shrink risk** (modularize `app.py` + clear TODOs)
2. **Scale data correctly** (schema + provenance + dataset build pipeline for “competition” datasets)
3. **Package for Kaggle** (a reproducible single-session notebook path)

---

# Cursor Handoff Prompt — M29

## Milestone: M29 — Competition-Ready Data + App Router Modularization

**Goal:** Make the project “submission-ready”: stable API structure, scalable dataset pipeline, and a Kaggle-oriented single-session training + eval path, while keeping CI green and end-to-end verified.

### Branch

`milestone/M29-competition-data-and-routers`

---

## Phase 0 — Baseline Gate (no work until green locally)

1. Pull latest `main`
2. Run full local gates:

   * Backend: ruff check/format, mypy, pytest, coverage gate
   * Frontend: unit tests + build
   * E2E: Playwright
3. Record baseline metrics in PR description:

   * backend test count / coverage
   * e2e pass count
   * “app.py line count” snapshot (we’re going to reduce it)

**Acceptance:** all gates green before changes.

---

## Phase 1 — `app.py` modularization into routers (audit-driven quick win)

**Context:** Audit flags `app.py` at ~1,563 lines and recommends router extraction. 

### Tasks

1. Create `backend/tunix_rt_backend/routers/` package:

   * `routers/health.py`
   * `routers/traces.py`
   * `routers/datasets.py`
   * `routers/tunix_runs.py`
   * `routers/tuning.py`
   * (whatever else is already grouped in `app.py`)
2. Move route handlers out of `app.py` into routers with `APIRouter`.
3. Keep `app.py` focused on:

   * app creation
   * middleware
   * router inclusion
   * exception handlers
4. Ensure OpenAPI tags remain stable (don’t break frontend client expectations).

### Guardrails

* No route path changes unless absolutely necessary.
* If any path changes: update frontend client + e2e selectors in same PR.

**Acceptance**

* `app.py` reduced substantially (target: < 900 lines in M29; further reductions later)
* Backend + frontend + e2e all pass.

---

## Phase 2 — Resolve remaining TODO/FIXME markers

Audit reports **3 TODO/FIXME** markers (model_registry/regression). 

### Tasks

* Either implement them or convert into explicit tracked issues with:

  * reason
  * owner milestone (e.g., M30)
  * link + removal from code

**Acceptance:** `rg "TODO|FIXME"` yields zero in core backend (or only intentionally-allowed patterns with justification).

---

## Phase 3 — Dataset scale-up pipeline (schema-first, not volume-first)

**Goal:** Make it easy to build “real” datasets with provenance and repeatability.

### Tasks

1. Add a **generic dataset builder/ingestor** that can:

   * take a JSONL source (or directory)
   * validate schema
   * persist traces to DB
   * emit a dataset key + manifest (versioned)
2. Add provenance metadata:

   * dataset source name
   * build timestamp
   * schema version
   * count of items/traces imported
3. Add one “bigger-than-golden” dev dataset (still small enough for CI):

   * e.g., `dev-reasoning-v1` (a few hundred items)
   * stored under a repo-safe location (or generated in CI from a script)
4. Add tests:

   * schema validation tests
   * “build dataset then dry-run train export is non-empty”
   * regression test to avoid a repeat of “Dataset is empty” e2e failures

**Acceptance**

* `POST /api/datasets/build` (or equivalent) can build a dataset deterministically
* dataset export for training produces non-empty JSONL and passes validation.

---

## Phase 4 — Kaggle submission path (single-session reproducibility)

Competition reality: time/weekly TPU caps push toward “one clean run” + tight reproducibility. ([Kaggle][2])

### Tasks

1. Add `docs/kaggle_submission.md` describing:

   * exact command sequence for a single-session run
   * expected artifacts
   * how to reproduce evaluation
2. Add a `notebooks/` folder with a “Kaggle-style” script/notebook scaffold:

   * minimal cells / clear narrative
   * pulls a dataset build step
   * runs training for a bounded time
   * runs evaluation and prints leaderboard score
3. Add “fast mode” flags:

   * `--max_steps`, `--max_examples`, `--time_budget_minutes`
   * deterministic seeds

**Acceptance**

* Running the notebook path locally produces:

  * a trained checkpoint artifact
  * eval results JSONL
  * printed scalar score
* Documented as a copy/paste “one session” recipe.

---

## Phase 5 — CI tier upgrade (optional but recommended)

Audit notes “Nightly tier not implemented.” 

### Tasks

* Add a nightly workflow that runs:

  * full backend tests
  * e2e
  * optional longer dataset validation / smoke train (very bounded)

**Acceptance:** nightly workflow exists and is green (may be skipped in PR CI, but must run on schedule).

---

## Deliverables

* Router refactor merged with no API breakage
* TODOs removed or tracked
* Generic dataset build pipeline + tests
* Kaggle single-session submission guide + scaffold
* CI green

---

## PR checklist (must include)

* Screenshots: API docs page + (if relevant) dataset builder UI flow
* “How to run M29 locally” section
* Confirmation: backend(3.11/3.12), frontend, e2e, security jobs all green

---

## Where you stand vs the competition (right now)

* You’re ahead on **engineering rigor** (CI gates, SBOM, trace artifacts, tuning infra). 
* The missing piece is **competition-shaped data + a Kaggle-native “single session” narrative**. The event is judged and has a fixed deadline window. ([Kaggle][1])
* M29 as scoped above directly targets “model quality in one session” and notebook clarity—exactly the pressure points people are discussing. ([Kaggle][2])

If you want, after M29 I’d set **M30 = “Eval semantics hardening + judge calibration + video/script outline”** (because video quality is explicitly part of judging in multiple community summaries/discussions), then **M31 = “Final submission lock”**. ([competehub.dev][3])

[1]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[2]: https://www.kaggle.com/competitions/google-tunix-hackathon/discussion/651560?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[3]: https://www.competehub.dev/en/competitions/kagglegoogle-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work - Comp..."
