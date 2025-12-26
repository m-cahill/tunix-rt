According to a document from **December 25, 2025**, M29 put you in a genuinely strong spot: `app.py` was reduced from ~1,800 lines to ~56 via router modularization, you now have a competition-oriented Kaggle submission notebook + guide, a dataset ingest endpoint with provenance, and a seeded “dev-reasoning-v1” dataset (200 traces) for smoke training/eval. The audit’s remaining gaps are mostly polish + submission hardening (not architecture).

## Competition reality check (where you stand)

* **You have the right shape of pipeline**: dataset build/ingest → bounded training → prediction generation → scoring/reporting is explicitly documented and notebooked now.
* **Data looks “good enough to iterate”**: `dev-reasoning-v1` is deterministic and reasoning-trace-heavy (70% step-by-step) which is exactly what you want for rapid loop closure. It’s not “final competitive scale” yet, but it’s a solid dev dataset for calibration and regression control.
* **Timeline (verify current)**: third-party coverage indicates a **submission deadline of Jan 12, 2026**, with judging following shortly after. ([I Programmer][1])
* **Judging emphasis**: sources describing the event consistently highlight **notebook/model quality + video quality** as major scoring components (i.e., *presentation and reproducibility matter as much as raw model quality*). ([AI Competition Hub][2])

---

## Quick confirmation on your earlier “Tunix placeholder” question

Yes — if `tunix` is a placeholder dependency in this repo, then “Real Tunix/JAX training path” should be implemented as a **direct JAX/Flax/Optax SFT loop** in your `train_jax.py` / `training/train_jax.py` (i.e., *behave like what Tunix would do*, without calling an external Tunix library). Keep the interfaces/CLI stable so swapping to a real Tunix package later is mechanical.

---

# M30 — Prompt to hand off to Cursor

Below is a Cursor-ready milestone prompt that matches the audit’s recommended “M30: Polish & Submission Prep” items (mypy ignore cleanup, dataset ingest E2E, deprecation fix, router docstrings, Kaggle dry-run, submission checklist).

---

## ✅ Cursor Milestone Prompt: M30 — Polish & Submission Prep

### Goal

Ship a **competition-grade, submission-ready** repo state by closing the remaining **low-risk polish items** identified in the M29 full audit, without introducing feature creep.

This milestone should be small, surgical, and end with **CI green**.

### Non-goals

* No new product features.
* No schema changes unless strictly required for one of the tasks below.
* No refactors not tied to acceptance criteria.

---

## Phase 0 — Baseline Gate

1. Branch from `main`:

   * `milestone/M30-polish-and-submission-prep`
2. Confirm baseline is green locally:

   * Backend: `uv run ruff check . && uv run ruff format --check . && uv run mypy tunix_rt_backend && uv run pytest`
   * Frontend: `npm test`
   * E2E: whatever the repo’s standard command is (ensure it passes once before edits)

**Acceptance:** green baseline.

---

## Phase 1 — Remove unused mypy ignores (audit item M30-1)

The M29 audit calls out **unused type-ignore comments** in integration/availability modules. Remove them so mypy output stays clean and the codebase doesn’t accumulate “ignore barnacles”.

**Tasks**

1. Run mypy in a way that surfaces `unused-ignore` (repo already does in CI).
2. Remove unused `# type: ignore` lines (or narrow them only if they’re truly needed).
3. Add short inline comments *only where ignores remain* explaining why they are necessary.

**Acceptance**

* `uv run mypy tunix_rt_backend` passes with **0 errors**.
* No new ignores added unless justified with a comment.

---

## Phase 2 — Add dataset ingest E2E coverage (audit item M30-2)

M29 added `POST /api/datasets/ingest` but the audit flags that it has **no E2E test** yet.

**Implementation strategy (robust, CI-friendly)**

1. Add a tiny JSONL fixture in a location the backend can read during E2E (recommended):

   * `backend/tools/testdata/e2e_ingest.jsonl` (or similar)
   * Keep it to 2–3 traces, valid Pydantic shape.
2. Add Playwright test:

   * `e2e/tests/datasets_ingest.spec.ts` (or append to an existing datasets spec)
   * Call `/api/datasets/ingest` with:

     * `path: "tools/testdata/e2e_ingest.jsonl"` (relative to backend working dir in CI)
     * `source_name: "e2e-test"`
   * Assert:

     * HTTP 200
     * Response includes `ingested_count > 0`
3. If CI backend working directory differs, adjust the path to match how CI starts the backend (keep the fix minimal).

**Acceptance**

* E2E job passes in CI.
* Test asserts ingest succeeded (not just “200 OK”).

---

## Phase 3 — Fix deprecated HTTP status constants (audit item M30-3)

Audit flags deprecation of `HTTP_422_UNPROCESSABLE_ENTITY` and recommends moving to `HTTP_422_UNPROCESSABLE_CONTENT`.

**Tasks**

1. Search for `HTTP_422_UNPROCESSABLE_ENTITY` usage across backend.
2. Replace with `HTTP_422_UNPROCESSABLE_CONTENT` (Starlette).
3. Run tests.

**Acceptance**

* No remaining deprecated constant usage.
* CI green.

---

## Phase 4 — Router module docstrings (audit item M30-4)

Audit calls router modules “minimally documented” and wants module-level docstrings explaining what each router owns.

**Tasks**
For each file in `backend/tunix_rt_backend/routers/*.py`:

1. Add a concise module docstring:

   * What domain it covers
   * Primary endpoints
   * Any cross-cutting concerns (auth, payload size, provenance, etc.)

Keep it short; don’t add verbose narrative.

**Acceptance**

* Every router module has a module docstring.
* No functional changes.

---

## Phase 5 — Kaggle notebook dry-run (audit item M30-5)

M29 created both `notebooks/kaggle_submission.ipynb` and `notebooks/kaggle_submission.py` and a full guide; M30 should prove it runs end-to-end in a minimal mode.

**Tasks**

1. Run a minimal dry-run locally (CPU is fine):

   * Build dataset (use `dev-reasoning-v1` to keep it fast)
   * Train with very small steps (e.g., `--max_steps 2` or equivalent)
   * Generate predictions
   * Produce score/report
2. Capture the exact commands used and record them in a short “Dry Run Verified” section in `docs/kaggle_submission.md` (or a small `docs/kaggle_dry_run.md`).

**Acceptance**

* Dry run succeeds end-to-end with a tiny step count.
* Documented commands are copy/pasteable.

---

## Phase 6 — Submission checklist (audit item M30-6)

Create a “final submission” checklist so you don’t lose points on presentation/repro steps (notebook quality + video quality matter per competition writeups). ([AI Competition Hub][2])

**Tasks**
Add `docs/submission_checklist.md` with:

* Environment setup
* Dataset chosen + provenance notes
* Training config (steps, seed, checkpoint policy)
* Eval config + scoring
* Artifacts to export (model, logs, metrics, report)
* Video requirements (outline only; script can be M31)
* Final “last-mile” sanity checks

**Acceptance**

* Checklist exists and is specific to this repo’s paths/scripts.

---

## Final Gate (required)

Before merging:

* `uv run ruff check .`
* `uv run ruff format --check .`
* `uv run mypy tunix_rt_backend`
* `uv run pytest`
* `npm test`
* E2E suite passes
* CI green

---

# What the next milestones likely look like (preview)

This matches the M29 full audit’s suggested sequencing: **M30 polish**, then **M31 submission packaging**.

* **M31 — Final Submission Package**

  * Code freeze
  * Video script + narrative
  * Final Kaggle run(s) and exported artifacts
* **M32 — Data scale-up + judge calibration**

  * Expand dataset beyond 200 traces (if quality holds)
  * Calibrate evaluation semantics so improvements are real (not noise)
* **M33 — Final training + sweep**

  * Your best bounded training recipe (and possibly a small tuning sweep if time allows)

If you want, I can also turn that preview into Cursor prompts for M31/M32 immediately—same style as above, same “small and verifiable” phases.

[1]: https://www.i-programmer.info/news/204-challenges/18460-google-tunix-hack-hackathon-now-open.html "Google Tunix Hack Hackathon Now Open"
[2]: https://www.competehub.dev/en/competitions/kagglegoogle-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work - Comp..."
