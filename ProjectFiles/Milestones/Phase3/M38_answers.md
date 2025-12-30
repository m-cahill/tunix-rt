Below are **clear, decision-locking answers** to the M38 questions. Cursor can proceed immediately after this.

(Reference: your M38 questions doc )

---

## 1. Kaggle TPU Execution — Who Does What?

**Decision:** **I will execute the Kaggle TPU run.**

**Status & assumptions:**

* **TPU quota:** Yes / will verify before running.
* **HF_TOKEN in Kaggle Secrets:** Yes / will confirm.
* **Gemma license accepted:** Yes.

**Your role (Cursor):**

* Verify the notebook is ready (no code changes).
* Prepare/validate the evidence folder structure.
* Populate evidence files once I provide the real run data.
* Work on optional frontend coverage in parallel (see #5).

---

## 2. Evidence Folder Versioning

**Decision:** **Option A — Update the existing `submission_runs/m37_v1/`.**

**Rationale:**

* M37 established the infrastructure; M38 executes it.
* Treat M37+M38 as a **single evidence unit** to avoid fragmentation.
* Replace all `null` placeholders with real values.

*No new `m38_v1/` folder.*

---

## 3. Frontend Coverage — Target Components

**Decision:** **Option A — Run coverage and add only obvious gaps.**

**Scope control:**

* Generate a coverage report.
* Add **minimal** tests only where coverage is clearly lacking.
* Do **not** refactor components or chase perfect coverage.
* Stop once ~50% overall is reached or diminishing returns appear.

---

## 4. Per-Item Predictions — Scope Clarity

**Decision:** **Option A — Filesystem only (already done).**

**Clarification:**

* `predictions.jsonl` written to the run output directory is sufficient.
* **No database persistence** in M38.
* Skip if anything beyond trivial changes is required.

---

## 5. Timeline & Prioritization

**Decision:** **Option B — Parallel.**

**Plan:**

* I run the Kaggle TPU job and deliver:

  * Run ID, logs, metrics, checkpoints, samples.
* You simultaneously:

  * Prep evidence files for population.
  * Do optional frontend coverage uplift.
* Once TPU data lands, you finalize evidence and verify CI.

---

## Final Execution Order (Locked)

1. **I execute** the Kaggle TPU run (Gemma 2B, 200 steps).
2. **You prepare** evidence structure + run frontend coverage in parallel.
3. **I deliver** run artifacts and identifiers.
4. **You populate** `submission_runs/m37_v1/` with real values.
5. **Final checks:** GPU smoke, CI green, Transformers pinned `<5`.

This closes all M38 ambiguities.
