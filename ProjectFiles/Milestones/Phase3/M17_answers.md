Here are **clear, locked answers** to the M17 clarifying questions, with rationale. These are intended to be **decisions**, not open-ended discussion, so Cursor can proceed without ambiguity.
(Questions referenced from )

---

## 1️⃣ Automatic Triggering

**Decision**

* ✅ **Yes**, evaluation should run **automatically** after a Tunix run reaches a **successful terminal state** (`completed`).
* ✅ **Yes**, it must also be possible to **manually trigger / re-trigger** evaluation for an existing run via API.

**Rationale**

* Automatic evaluation establishes evaluation as a **first-class, default part of the run lifecycle**, not an optional add-on.
* Manual re-triggering is required for:

  * New judge versions
  * Bug fixes in evaluation logic
  * Re-scoring historical runs for comparison/regression

**Concrete implication**

* Automatic trigger: async hook on run completion
* Manual trigger: `POST /api/tunix/runs/{id}/evaluate`
* Evaluation runs must be **idempotent** and **versioned**

---

## 2️⃣ Database & Leaderboard Performance

**Decision**

* ✅ Create a **dedicated Postgres table**: `tunix_run_evaluations`
* ✅ Store **key, sortable metrics as columns**
* ✅ Store the **full evaluation output as a JSON artifact**

**This exactly matches the recommendation.**

**Why this is the correct architecture**

* Leaderboards **cannot** be powered efficiently from blob artifacts
* Evaluation artifacts are for **auditability and deep inspection**
* Tables are for:

  * Sorting
  * Filtering
  * Comparing
  * Indexing

**Expected split of responsibility**

* **DB table**: `score`, `accuracy`, `verdict`, `judge_version`, `created_at`
* **Artifact (`evaluation.json`)**: full reasoning, per-metric breakdown, raw judge output

This mirrors how serious ML platforms separate **signals** from **records**.

---

## 3️⃣ Judge Implementation (Initial)

**Decision**

* ✅ Start with a **deterministic mock judge** for M17
* ❌ Do **not** implement a real Gemma judge yet

**What the mock judge should do**

* Deterministically score based on simple, transparent logic, e.g.:

  * Output length
  * Presence of required fields
  * Basic structural checks
* Always produce:

  * A numeric score
  * A verdict
  * A stable explanation payload

**Rationale**

M17 is about validating **evaluation infrastructure**, not judge quality.

Using a real LLM judge too early:

* Introduces nondeterminism
* Complicates CI
* Makes regressions ambiguous

**Rule of thumb**

> *“Infrastructure first, intelligence later.”*

A real `gemma-judge-v1` belongs in a **later milestone**, once evaluation is trusted.

---

## 4️⃣ API & Frontend Design

### API

**Decision**

* ✅ Use a **separate endpoint**:

  * `GET /api/tunix/runs/{id}/evaluation`
* ❌ Do not inline evaluation into `GET /api/tunix/runs/{id}`

**Rationale**

* Keeps run metadata lightweight
* Avoids over-fetching
* Allows evaluation to evolve independently
* Matches clean REST separation of concerns

---

### Frontend / Leaderboard UI

**Decision**

* ✅ Add a **new top-level page**: `/leaderboard`

**Rationale**

* Leaderboards are a **distinct conceptual object**, not a property of a single run
* A top-level page makes comparison first-class
* Avoids bloating the Runs page with cross-run logic

The Runs page can later link *to* the leaderboard, not host it.

---

## Final Summary (for Cursor)

* Evaluation runs **automatically** + **manually retriggerable**
* Use **DB table + artifact**, not artifact-only
* Start with a **deterministic mock judge**
* Expose evaluation via a **separate API endpoint**
* Implement leaderboard as a **top-level page**

These decisions keep M17:

* Tight
* Deterministic
* Auditable
* Aligned with enterprise ML platform norms

If you want, next I can:

* Turn these answers into a **single Cursor-ready M17 implementation prompt**
* Sketch the **exact DB schema** for `tunix_run_evaluations`
* Define the **mock judge contract** so it won’t be thrown away later
