Below is a **Cursor-ready M18 handoff prompt**, written to **close the loop from M17** and proceed with **maximum leverage and minimum risk**. This prompt reflects the findings and recommendations from the **M17 Audit**, **M17 Full Codebase Audit**, and **M17 Completion Summary** .

---

## 🎯 Cursor Prompt — Milestone M18: Judge Abstraction, Real Judge Integration, Regression Gates

### Context (Read First)

M17 successfully delivered:

* A deterministic **EvaluationService** with a **Mock Judge**
* A durable **evaluation schema + DB table**
* Automatic evaluation on run completion (sync + async)
* A **Leaderboard UI** showing ranked runs
* Stable CI and strong test coverage

The codebase is **healthy, modular, and ready** for the next conceptual step.

However, the M17 audit identified that:

* Scoring logic is currently **embedded** in `EvaluationService`
* The mock judge must now be replaced by a **real LLM judge**
* Leaderboard queries will soon need pagination
* There are no **regression gates** yet to prevent silent quality degradation

M18 addresses these gaps.

---

## 🧭 M18 North Star

> Transition from *deterministic mock evaluation* to a **pluggable, production-grade judge architecture**, with the first real judge implementation and the first **quality regression gates**.

This milestone **does not** start large-scale data generation yet.
It locks **semantics, trust, and evaluation credibility**.

---

## ✅ M18 Deliverables (Strict Scope)

### 1️⃣ Extract a `Judge` Interface (Required)

**Goal:** Decouple scoring logic from `EvaluationService`.

**Actions**

* Introduce a `Judge` protocol / interface (e.g. `BaseJudge`)
* Methods should include:

  * `evaluate(run, artifacts) -> EvaluationResult`
  * `judge_id` / `version`
* Refactor `EvaluationService` so it:

  * Fetches the run
  * Delegates scoring entirely to a `Judge`
  * Persists results

**Concrete output**

* `MockJudge` becomes one implementation of `Judge`
* `EvaluationService` no longer contains scoring logic

**Acceptance**

* No behavior change for existing tests
* Mock judge tests still pass

---

### 2️⃣ Implement First Real Judge (`GemmaJudge`) — Minimal but Real

**Goal:** Prove the system supports a real LLM-based judge.

**Constraints**

* Start **simple and conservative**
* Do NOT optimize prompts yet
* Do NOT introduce hyperparameter sweeps yet

**Actions**

* Implement `GemmaJudge` (or equivalent) that:

  * Consumes run output + trace
  * Produces:

    * `score` (numeric)
    * `verdict` (pass/fail)
    * `metrics` (lightweight)
    * `raw_judge_output` (stored in `details`)
* Judge must be:

  * Async-safe
  * Timeout-bounded
  * Fail-closed (evaluation fails loudly, not silently)

**Config**

* Allow selecting judge via:

  * config flag
  * env var
  * or `judge_override` field (already present)

**Acceptance**

* Mock judge remains default in CI
* Real judge can be enabled locally or in controlled runs
* Evaluation schema remains unchanged

---

### 3️⃣ Regression Gates (Foundational)

**Goal:** Prevent silent quality regressions.

**Actions**

* Introduce a simple regression API or service function:

  * “Is this run better than baseline?”
* Baseline can be:

  * Best historical score
  * Named baseline run
* Define initial gate:

  * e.g. fail if score drops > X%

**Scope**

* Backend only (no UI polish yet)
* No automatic blocking of deploys yet — just detection + signal

**Acceptance**

* Regression check callable via service or endpoint
* Unit tests covering:

  * Pass
  * Fail
  * No-baseline cases

---

### 4️⃣ Leaderboard Pagination (Small but Important)

**Goal:** Prevent performance cliffs.

**Actions**

* Add `limit` / `offset` (or cursor) to leaderboard endpoint
* Update frontend to consume paginated results
* Default page size ~25–50

**Acceptance**

* No behavior regression
* Existing leaderboard tests updated or extended

---

## 🚫 Explicit Non-Goals for M18

Do **not**:

* Add Ray Tune / sweeps
* Generate large datasets
* Optimize judge prompts
* Add auth / permissions
* Add multi-judge ensembles

Those belong to M19+.

---

## 🧪 Quality Gates (Must Hold)

* CI fully green
* Mock judge tests unchanged
* New judge logic fully unit tested
* No schema changes that break M17 artifacts
* Coverage remains ≥ current levels

---

## 🏁 M18 Completion Criteria

You should be able to say:

* ✅ Evaluation supports **multiple judges**
* ✅ A **real LLM judge** can score runs
* ✅ Regression detection exists
* ✅ Leaderboard scales beyond trivial size
* ✅ System is still deterministic by default

At that point, the platform is **evaluation-credible** and **trustworthy** — ready for tuning and data generation.

---

### Suggested Commit Theme

```
M18: Judge abstraction, real LLM judge, and regression gates
```

---

If you want next:

* I can generate a **follow-up M19 prompt** (hyperparameter tuning + data readiness)
* Or draft the **exact `Judge` interface and class skeletons** for Cursor to start from
