# M35 Clarifying Questions

**Milestone:** M35 — Quality Loop 1 (Eval Signal Hardening + Leaderboard Fidelity + Regression Guardrails)  
**Date:** December 26, 2025  
**Status:** ⏳ Awaiting Answers

---

## Questions

### 1. Eval Set v2 Composition

The M35 plan requests **75–150 items** in `eval_v2.jsonl` with three sections:
- Core "answer correctness" (deterministic answers)
- "Trace sensitivity" (reasoning steps matter)
- "Edge cases" (formatting, whitespace, casing, numeric equivalence)

The current `eval_v1.jsonl` has 25 items (arithmetic, knowledge, word_problem, geometry, pattern, conversion, number_theory, probability).

**Questions:**
- **(a)** Should I **extend** `eval_v1.jsonl` to 100+ items (keeping same schema), or create a **fresh** `eval_v2.jsonl` with potentially different schema?
- **(b)** What percentage split between the three sections? My proposal: **60% core / 25% trace-sensitive / 15% edge cases**.
- **(c)** Should the "trace sensitivity" items include multi-step ground-truth traces, or just require longer reasoning without an explicit trace field?

**Answer:**

---

### 2. Scorecard Breakdown Categories

Phase 2.2 asks for "per-category breakdown (if eval items are labeled)". The current `eval_v1.jsonl` uses a `category` field (arithmetic, geometry, etc.).

**Questions:**
- **(a)** Should I keep these **existing categories** for eval_v2, or use a different taxonomy (e.g., "core / trace / edge" as primary, with subcategories)?
- **(b)** Should the scorecard show breakdown by **difficulty** as well (`easy`, `medium`, `hard`)?

**Answer:**

---

### 3. Leaderboard Filtering Scope

Phase 3.1 requests leaderboard filtering by `dataset`, `model_id`, `config`, and `date range`. The current API has no filters.

**Questions:**
- **(a)** Are all four filters **required** for M35, or should I prioritize some (e.g., dataset + model_id) and defer others (config, date) to later?
- **(b)** Should the filtering be **additive** (AND logic) or allow OR conditions?
- **(c)** For `config` filtering, should this match by config file name (e.g., `m34_best.yaml`) or by embedded config fields (e.g., `learning_rate: 1e-5`)?

**Answer:**

---

### 4. Run Comparison Per-Item Diff Table

Phase 3.2 requests a compare view with a "per-item diff table (item id, expected, predicted, correctness)."

**Questions:**
- **(a)** Does this require storing **per-item predictions** in the evaluation record? Currently `detailed_metrics` has `item_{trace_id}` entries, but not full predicted text.
- **(b)** The current `AnswerCorrectnessJudge` stores details in `detailed_metrics`, but these are lost once training output dirs are cleaned up. Should I persist the full per-item comparison data in the DB?
- **(c)** For runs evaluated with `MockJudge`, should the per-item table show "N/A" or be hidden?

**Answer:**

---

### 5. Regression Baseline Lock Mechanism

Phase 4.1 mentions a "baseline run definition or mechanism" with a regression check that fails on drops beyond a threshold.

**Questions:**
- **(a)** The current `RegressionService` already has baseline creation and a 5% tolerance check. Is M35 asking for **enhancements** (e.g., multiple baselines per dataset, auto-baseline on best run), or just **documenting/testing** the existing mechanism?
- **(b)** Should there be a **CI integration** that runs regression checks automatically, or is the API endpoint sufficient?
- **(c)** Should regression checks compare against `primary_score` specifically, or allow configurable metric (current behavior)?

**Answer:**

---

### 6. Determinism Check Implementation

Phase 4.2 requests a determinism check that runs evaluation twice and asserts identical results.

**Questions:**
- **(a)** Since `compute_primary_score()` is a pure function, determinism is guaranteed for identical input. Should the check focus on **end-to-end** determinism (same run → same eval)?
- **(b)** Should this be a **unit test**, a **standalone script** (`tools/check_determinism.py`), or a **CI step**?
- **(c)** For `MockJudge` (which uses hash-based scores), determinism is guaranteed. For `AnswerCorrectnessJudge`, it depends on DB state. Should the check use a **fixed snapshot** dataset?

**Answer:**

---

### 7. Evidence Folder and Packaging

Phase 5 requires `submission_runs/m35_v1/` with updated schema including non-null `primary_score`.

**Questions:**
- **(a)** Should I populate the evidence files with **real values** from a local run, or keep them as **templates** (like M34) for you to fill after Kaggle execution?
- **(b)** The M35 plan mentions `eval_v2` should be used — does this mean M35 evidence must show evaluation against `eval_v2.jsonl` specifically?

**Answer:**

---

### 8. Frontend Priorities

Phase 3.2 mentions "keep it lean: no fancy charts required, but ensure it's readable and stable."

**Questions:**
- **(a)** The current `RunComparison.tsx` shows loss curves and basic config. For M35, should I **add** the per-item diff table as a new section, or **replace** something?
- **(b)** Should the leaderboard UI show the **scorecard summary** (n_items, n_scored, n_skipped) inline, or as a collapsible detail panel?
- **(c)** Should the filtering UI be **inline inputs** (text fields/dropdowns above the table) or a **sidebar filter panel**?

**Answer:**

---

## Summary

Please provide answers to these 8 questions so I can finalize the M35 task list and begin implementation. I'll design solutions that maintain the codebase's modularity and test coverage standards (>70%).
