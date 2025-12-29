# M36 Clarifying Questions

**Milestone:** M36 — Real Kaggle Execution + Evidence Lock v2 + Quick-Win Audit Uplift  
**Date:** December 27, 2025  
**Status:** ❓ Awaiting Responses

---

## Questions

### 1. Kaggle Execution Scope

The M36 plan requires "a real Kaggle GPU/TPU run" with evidence files populated with real values. Currently, `submission_runs/m35_v1/eval_summary.json` has `primary_score: null` and other null fields.

Since I (Cursor AI) cannot physically execute Kaggle notebooks, should I:

- **(a)** Prepare everything (notebook updates, evidence templates, docs) so **you** can run on Kaggle and paste real values?
- **(b)** Create a local CPU/smoke "rehearsal" path that produces approximate evidence (as done in M33)?
- **(c)** Both — prepare for Kaggle + provide a local rehearsal option?

**Recommendation:** (c) seems consistent with the M33 pattern.

**Answer:**

---

### 2. Notebook Eval Set Update

The notebook currently uses `EVAL_SET = "training/evalsets/eval_v1.jsonl"` (50 items). M36 requires using `eval_v2.jsonl` (100 items with the new scorecard structure).

Should I:

- **(a)** Change the default to `eval_v2.jsonl` and keep `eval_v1.jsonl` as a documented fallback?
- **(b)** Keep `eval_v1.jsonl` as default but add `eval_v2.jsonl` as a clearly marked option?

**Recommendation:** (a) — eval_v2 is the competition-grade eval set.

**Answer:**

---

### 3. Evidence Schema Additions

The M35 evidence `run_manifest.json` already includes an `eval_set` field. For M36, should I add:

- **`kaggle_notebook_url`** — URL to the Kaggle notebook version (to be filled manually)?
- **`kaggle_run_id`** — Kaggle's internal run identifier?

Or should these remain as optional fields in the `notes` section?

**Answer:**

---

### 4. Frontend Test Coverage Targets

M36 audit uplift mentions:
- "5-10 tests for `Leaderboard.tsx`" (currently ~2% coverage)
- "3-5 tests for `LiveLogs.tsx`" (currently ~11% coverage)

What functionality should I prioritize testing?

**For `Leaderboard.tsx`, I'm proposing:**
1. Renders loading state
2. Renders empty state ("No evaluated runs found")
3. Renders leaderboard data with correct columns
4. Filter inputs render and update state
5. Apply/Clear filter buttons work
6. Pagination buttons work
7. Error state displays correctly
8. Scorecard displays correctly
9. Primary score percentage formatting
10. Date formatting

**For `LiveLogs.tsx`, I'm proposing:**
1. Renders waiting state
2. Renders logs when received
3. Connection status indicator
4. Handles status changes
5. Auto-scroll behavior

Is this prioritization correct, or are there specific behaviors you want tested?

**Answer:**

---

### 5. React `act()` Warnings Scope

The M35 audit notes ~20 `act()` warnings in test output. Should I:

- **(a)** Fix only warnings in M36-touched files (`Leaderboard.tsx`, `LiveLogs.tsx`)?
- **(b)** Fix all warnings across the test suite (including `App.test.tsx`)?

**Recommendation:** (b) for a clean test output, but (a) if scope needs to be bounded.

**Answer:**

---

### 6. Docs Update Scope

The plan mentions updating `docs/evaluation.md` with:
> "Per-item predictions: current state + limitation + planned M37 artifact storage"

Should I create a new section in `evaluation.md`, or would a separate `docs/M36_KAGGLE_RUN.md` (as mentioned in Phase 1.2) be the primary doc deliverable?

**Answer:**

---

## Summary

Please provide answers to these 6 questions so I can finalize the task list and begin M36 implementation.
