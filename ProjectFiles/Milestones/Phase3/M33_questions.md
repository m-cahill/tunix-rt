# M33 Clarifying Questions

**Milestone:** M33 — Kaggle "Submission Rehearsal" Run v1 + Evidence Lock  
**Date:** December 26, 2025  
**Status:** ✅ Complete — See M33_answers.md for responses

---

## Questions

### 1. Kaggle Execution Scope

The M33 plan requires actually running on Kaggle and capturing logs. Since I (Cursor AI) cannot physically execute Kaggle notebooks, should I:

- **(a)** Prepare everything (scripts, configs, templates, CI tests) so **you** can execute on Kaggle and fill in evidence?
- **(b)** Simulate a local "rehearsal run" using CPU/smoke mode and capture that as evidence?
- **(c)** Both — prepare for Kaggle + do a local rehearsal?

**Answer:**

---

### 2. Model Selection

The plan mentions Gemma 3 1B for "iteration speed." The notebook currently defaults to `google/gemma-3-1b-it`. Should M33 stick with this, or switch to `google/gemma-2-2b`?

**Answer:**

---

### 3. Dataset for the Run

The plan specifies `dev-reasoning-v2` (550 traces), but the notebook defaults to `golden-v2`. Should I:

- **(a)** Update the notebook default to `dev-reasoning-v2`?
- **(b)** Keep `golden-v2` and just document `dev-reasoning-v2` as an option?

**Answer:**

---

### 4. Evidence Folder Naming

M32 created `submission_runs/m32_v1/`. The plan references `submission_runs/m33_v1/`. Should I:

- **(a)** Create a **new** `m33_v1/` folder with fresh templates?
- **(b)** Rename/update the existing m32_v1 structure?

**Answer:**

---

### 5. Packaging Tool Enhancement

The plan shows:

```bash
python tools/package_submission.py --run-dir submission_runs/m33_v1
```

But the current tool uses `--include-output` (for training output dirs), not `--run-dir` (for evidence folders). Should I add a `--run-dir` argument that specifically bundles evidence files from a named run?

**Answer:**

---

### 6. CI Test Schema

Phase 4 asks for tests validating `run_manifest.json` and `eval_summary.json` have "required keys." What keys are mandatory?

**Proposed schema:**

```json
// run_manifest.json required fields
["run_version", "model_id", "dataset", "commit_sha", "timestamp"]

// eval_summary.json required fields
["run_version", "eval_set", "metrics", "evaluated_at"]
```

Is this correct, or should I add/remove fields?

**Answer:**

---

## Summary

Please provide answers to these 6 questions so I can finalize the task list and begin M33 implementation.
