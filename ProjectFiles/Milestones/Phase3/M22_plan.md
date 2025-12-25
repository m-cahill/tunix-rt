Below is a **Cursor-ready prompt for M22** that (1) closes the remaining M21 audit gaps (coverage + frontend tests), (2) **freezes evaluation semantics** so training signal is real, and (3) adds the minimum UX/visibility needed to confidently start training afterward. It’s intentionally broken into **small, end-to-end verifiable phases**. (Grounded in the M21 audit + full audit + summary.)   

---

## Cursor Handoff Prompt — M22: Evaluation Lock + Coverage Recovery + Training Readiness

### Context

* M21 delivered security hardening + hermetic E2E + SBOM reliability + “Promote Best Trial”. 
* Audits say the **main remaining gap** is **backend coverage (~69%)** and **more frontend unit tests**, plus formalizing the evaluation loop before training.  
* Training should begin only once evaluation semantics and datasets are stable and reproducible; LLM-as-judge systems specifically need explicit rubrics + calibration practices. ([Evidently AI][1])

### M22 Goal (North Star)

**Make the system training-ready** by:

1. Restoring backend coverage to **≥75%** via targeted unit tests,
2. Implementing and freezing **Answer Correctness** evaluation semantics (ground-truth based),
3. Making evaluation results **visible and comparable** (API + minimal UI),
4. Adding guardrails so tuning/training doesn’t optimize on moving targets.

---

# Phase 0 — Baseline Gate (mandatory)

**Objective:** verify M21 head is solid before changes.

* Pull latest `main`.
* Run locally:

  * `cd backend && ruff format --check . && ruff check . && pytest -q`
  * `cd frontend && npm ci && npm run build && npm test`
  * `cd e2e && npx playwright test`
* If anything fails: fix in a tiny PR before M22 work.

**Exit criteria:** baseline green locally.

---

# Phase 1 — Coverage Recovery (quick wins, high audit ROI)

**Target:** backend coverage **≥75%** (or at least restore ≥70% gate immediately, then push to ≥75%).

## 1.1 Add unit tests for `TuningService` (error paths)

Create tests that exercise:

* invalid search space validation / schema errors
* Ray Tune job start failure handling (mocked)
* DB session isolation expectations (worker-style)
* evaluation failure propagation into trial status

**Acceptance:** measurable coverage improvement + tests deterministic (no Ray required).

## 1.2 Add unit tests for `ModelRegistryService` (error paths + idempotency)

Cover:

* promote with missing run artifacts → 4xx/meaningful message
* promote from failed run (unless force) → blocked
* idempotency: same run + same artifact hash returns existing version
* download endpoint behavior (happy path + missing artifact)

**Acceptance:** coverage rises; tests don’t touch real filesystem beyond tmpdir.

## 1.3 Frontend tests: “Promote Best” error handling

Add Vitest component tests to cover:

* promote success toast + refresh
* promote failure (API rejects) shows error banner/toast

**Guardrail:** keep E2E as integration; these tests should be fast and mocking-only.

**Exit criteria (Phase 1):**

* backend coverage ≥75% (preferred) or ≥70% minimum gate restored
* CI green

---

# Phase 2 — Freeze Evaluation Semantics (Answer Correctness)

This is the *training-readiness gate*.

## 2.1 Define the metric spec (code + doc)

Create `docs/evaluation.md` and a small constants module:

* **primary metric**: `answer_correctness`
* scale: `0/1` (MVP)
* aggregation: mean across dataset
* tie-breakers: choose next-best metric or lowest latency, etc.
* failure handling: missing prediction → 0, invalid format → 0

(Keep it simple and explicit.)

## 2.2 Implement `answer_correctness` evaluator

Implement in backend eval pipeline (service layer), using dataset items with:

* `prompt`
* `ground_truth` (string)
* `prediction` (string from run output)

MVP scoring:

* normalize whitespace/case
* exact match OR simple canonicalization rules
* return `correct: bool`, `score: 0|1`, `explanation`

**Important:** Keep it deterministic. This is your first “locked” metric.

> LLM-as-judge can be added later; when you do, use explicit rubrics and calibrate against a labeled sample to avoid judge drift. ([Evidently AI][1])

## 2.3 Persist evaluation results per run

Add fields/tables if needed:

* store metrics JSON with:

  * per-item correctness list (optional)
  * aggregate metrics
  * evaluator version (string)
  * dataset key/version

**Exit criteria (Phase 2):**

* `answer_correctness` computed for a run and stored
* metric spec documented and referenced by tuning/registry promotion

---

# Phase 3 — Dataset Canonicalization (small “golden set”)

## 3.1 Introduce a Golden Dataset contract

Add a canonical dataset key (example): `golden-v1`

* small (20–200 samples)
* versioned and reproducible
* includes ground truth

## 3.2 Dataset validator guardrails

Enforce at dataset build/export time:

* dataset must not be empty
* required fields present for evaluation type
* manifest references must resolve to DB entities
* fail fast with actionable messages

**Exit criteria (Phase 3):**

* `golden-v1` exists and exports reliably in CI/local
* evaluator works against `golden-v1`

---

# Phase 4 — Minimal UI/UX for Evaluation Visibility

## 4.1 Show evaluation metrics in Run / Tuning / Registry views

Minimal additions:

* In Run detail: show `answer_correctness` and dataset used
* In Tuning trials table: show primary metric column
* In Model Registry version detail: show promoted metric snapshot

**Exit criteria (Phase 4):**

* metrics visible end-to-end
* promotion continues to work and now carries meaningful evaluation signal

---

# Phase 5 — Training Readiness Checklist + “No premature tuning” guardrail

## 5.1 Add `docs/training_readiness.md`

Include explicit gates:

* evaluation spec locked
* golden dataset available
* primary metric stored + used for “best”
* registry promotion stable
* CI green

## 5.2 Guardrail: block “real tuning” unless evaluation locked

Implement a simple check:

* if evaluator version/spec not set → block tuning start (or require `--force`)

This prevents optimizing on undefined metrics (big research-quality win).

**Exit criteria (Phase 5):**

* docs merged
* guardrail prevents accidental meaningless tuning

---

# Definition of Done (M22)

* [ ] Backend coverage **≥75%** (or minimum ≥70% restored immediately), driven by new unit tests for Tuning + Registry services 
* [ ] `answer_correctness` metric implemented, deterministic, documented, persisted 
* [ ] Canonical `golden-v1` dataset contract + validator prevents empty/invalid exports
* [ ] Metrics visible in UI for runs/trials/registry versions
* [ ] Training readiness doc + “don’t tune before metrics are locked” guardrail
* [ ] CI green; E2E remains hermetic

---

## Notes / References (for implementation choices)

* LLM-as-judge best practice: define a rubric prompt + calibrate judge vs labeled sample to avoid drift. ([Evidently AI][1])
* Playwright trace options (retain traces on first failure) live in config; use `retain-on-failure` for debuggability. ([Playwright][2])

---

If you want M22 to be *extra* clean, make `answer_correctness` the only new metric and defer LLM-as-judge to M23 (after you’ve got stable baselines).

[1]: https://www.evidentlyai.com/llm-guide/llm-as-a-judge?utm_source=chatgpt.com "LLM-as-a-judge: a complete guide to using LLMs for ..."
[2]: https://playwright.dev/docs/test-use-options?utm_source=chatgpt.com "Configuration (use)"
