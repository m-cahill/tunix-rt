# M34 Clarifying Questions

**Milestone:** M34 — Optimization Loop 1: Ray Tune Mini-Sweep + Primary Score + Rehearsal v2 Evidence  
**Date:** December 26, 2025  
**Status:** 🔄 Awaiting Answers

---

## Questions

### 1. Primary Score Module Location

The M34 plan specifies creating `backend/tunix_rt_backend/scoring/primary_score.py`. However, there's already a `backend/tunix_rt_backend/scoring.py` file with `baseline_score()`. Should I:

- **(a)** Create a new `scoring/` subdirectory with `primary_score.py` inside it (and potentially move existing `scoring.py` content)?
- **(b)** Add `compute_primary_score()` to the existing `scoring.py` file?
- **(c)** Keep both — existing `scoring.py` for traces, new module for evaluation aggregation?

**Answer:**

---

### 2. Tuning Script Location

The plan mentions `backend/tools/run_tune_m34.py`, but there's already `scripts/run_m28_sweep.py` with Ray Tune integration. Should I:

- **(a)** Build on `scripts/run_m28_sweep.py` (rename or extend it)?
- **(b)** Create a new `backend/tools/run_tune_m34.py` (separate from existing script)?
- **(c)** Create both — keep M28 script for reference, add new M34 script with improved defaults?

**Answer:**

---

### 3. Primary Score Definition

The plan says `compute_primary_score(evaluation_rows) -> float | None` with formula `mean(answer_correctness)`. Currently, evaluations return:

- `score` — Judge's primary score (e.g., 0-100 from MockJudge)
- `metrics["answer_correctness"]` — 0.0-1.0 from AnswerCorrectnessJudge

Which value should `primary_score` aggregate?

- **(a)** `metrics["answer_correctness"]` (0.0-1.0 scale, mean across eval set)
- **(b)** The `score` field (0-100 scale)
- **(c)** Normalize both to 0-1 and prefer `answer_correctness` when present?

**Answer:**

---

### 4. Ray Tune Availability

The existing `TuningService` checks for Ray at import time and sets `RAY_AVAILABLE = False` if missing. For M34:

- **(a)** Do you have Ray installed locally for real sweep execution?
- **(b)** Should I focus on mocked/unit tests that don't require Ray?
- **(c)** Both — prepare real execution path + mocked tests for CI?

**Answer:**

---

### 5. Search Space for M34

The plan suggests a small search space. The existing M28 sweep uses:
- `learning_rate`: loguniform(1e-5, 1e-4)
- `batch_size`: choice([2, 4])
- `weight_decay`: uniform(0.0, 0.1)

Should M34 use the same, or expand to include:
- `warmup_steps`: choice([5, 10, 20])
- `max_steps`: choice([50, 100]) (for smoke-limited runs)

**Answer:**

---

### 6. Evidence Schema Update

M33's `eval_summary.json` already has `primary_score: null` (placeholder). For M34:

- **(a)** Should `primary_score` be required to be non-null for M34 evidence (real run)?
- **(b)** Should tests allow null for smoke runs but require non-null for production?

**Proposed M34 schema additions:**
```json
// run_manifest.json - add:
{
  "tuning_job_id": "optional UUID if from sweep",
  "trial_id": "optional string if from sweep"
}

// eval_summary.json - update:
{
  "primary_score": "number (required non-null for production runs)"
}
```

Is this correct?

**Answer:**

---

### 7. Config Promotion Format

The plan says create `training/configs/m34_best.yaml` from best trial params. Current configs use this structure:

```yaml
model:
  name: "google/gemma-2b-it"
  max_length: 512
training:
  learning_rate: 1.0e-5
  per_device_batch_size: 1
  # etc.
```

But Ray Tune returns flat params like `{"learning_rate": 2.3e-5, "batch_size": 4}`. Should I:

- **(a)** Manually map flat params to nested YAML structure?
- **(b)** Store best params as a flat JSON file alongside the config?
- **(c)** Both — YAML for training script, JSON for audit trail?

**Answer:**

---

## Summary

Please provide answers to these 7 questions so I can finalize the task list and begin M34 implementation.
