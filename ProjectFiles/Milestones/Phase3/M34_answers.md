Here are my answers to the **M34 clarifying questions**, with clear “do this” decisions Cursor can implement.

---

## 1) Primary Score module location

**Answer: (b) Add `compute_primary_score()` to the existing `backend/tunix_rt_backend/scoring.py`.**

Rationale: lowest churn, keeps imports stable, and avoids a refactor of existing `baseline_score()` just to satisfy a folder layout. We can always promote to a `scoring/` package later if it grows.

---

## 2) Tuning script location

**Answer: (a), with a light refactor. Build on `scripts/run_m28_sweep.py`, but extract shared logic so M34 can have clean defaults.**

Concrete approach:

* Extract reusable sweep runner logic into something like `backend/tunix_rt_backend/tuning/sweep_runner.py` (or similar).
* Keep `scripts/run_m28_sweep.py` as a thin wrapper (backward compatibility).
* Add a new thin wrapper with M34 defaults (either `scripts/run_m34_sweep.py` or `backend/tools/run_tune_m34.py`—pick one convention and stick to it).

This avoids duplication while still giving M34 a “single-command” entrypoint.

---

## 3) Primary score definition

**Answer: (c) Prefer `metrics["answer_correctness"]` (0–1). Fallback to normalized `score` when needed.**

Definition:

* If `metrics.answer_correctness` exists (float in `[0,1]`), use it.
* Else if `score` exists (often 0–100), normalize to `[0,1]` via `score / 100.0`.
* `primary_score = mean(valid_values)` across the eval set; if none exist, return `None`.

Why: “answer correctness” as a 0–1 metric is the cleanest primary objective for optimization and reporting. ([DeepEval][1])
Also, Ray Tune expects a single scalar metric for optimization. ([Ray Documentation][2])

---

## 4) Ray Tune availability

**Answer: (c) Both — implement a real execution path AND keep CI green with mocked/skip-tested coverage.**

Implementation guardrails:

* Real sweeps require `backend[tuning]` installed (Ray Tune available).
* CI should not require Ray; tests should mock/skip if `RAY_AVAILABLE` is false.

Ray Tune is designed around reporting a scalar metric and selecting best trials; keep that path “real” locally while CI validates wiring/schemas. ([Ray Documentation][3])

---

## 5) Search space for M34

**Answer: Use the existing M28 space, and add `warmup_steps`. Keep `max_steps` fixed (override), not tuned.**

Recommended M34 search space:

* `learning_rate`: loguniform(1e-5, 1e-4)
* `per_device_batch_size`: choice([1, 2, 4]) *(if your training loop supports it cleanly)*
* `weight_decay`: uniform(0.0, 0.1)
* `warmup_steps`: choice([0, 10, 20])

Recommended fixed overrides for sweep fairness/speed:

* `max_steps = 50` (or whatever is your “mini-sweep” budget)
* fixed dataset = `dev-reasoning-v2`
* fixed eval set = your standard eval file

Reason: Ray Tune works best when you tune a small, meaningful space early and keep runtime bounded. ([Ray Documentation][4])

---

## 6) Evidence schema update

**Answer: (b) Allow `primary_score=null` for smoke runs, but require non-null for the M34 “real rehearsal v2” evidence.**

Yes to your proposed schema additions.

I’d do it like this:

* Evidence schema *allows* null universally (for flexibility).
* M34 evidence tests add a stricter rule: if `run_version` starts with `m34_` (or equals `m34_v1`), then `primary_score` **must be non-null**.

Also: adding `tuning_job_id` and `trial_id` to `run_manifest.json` is correct and useful for provenance.

---

## 7) Config promotion format

**Answer: (c) Both — generate a nested YAML config for training + store best params as flat JSON for audit/provenance.**

Do both:

* `training/configs/m34_best.yaml` (ready-to-run)
* `submission_runs/m34_v1/best_params.json` (exact Ray Tune output, unmodified)
* Optional: write the mapping you applied into `run_manifest.json` (“promoted_from_trial_id”, “param_mapping”).

Mapping examples:

* `learning_rate` → `training.learning_rate`
* `batch_size`/`per_device_batch_size` → `training.per_device_batch_size`
* `weight_decay` → `training.weight_decay`
* `warmup_steps` → `training.warmup_steps`

This keeps the training interface stable and preserves the raw search output for auditability (very “enterprise-grade”).

---

If you want this turned into a **Cursor-ready M34 implementation prompt** (phases + acceptance criteria + exact files/tests), say so and I’ll generate it in the same style as your earlier milestone handoffs.

[1]: https://deepeval.com/guides/guides-answer-correctness-metric?utm_source=chatgpt.com "Answer Correctness Metric"
[2]: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html?utm_source=chatgpt.com "ray.tune.TuneConfig — Ray 2.53.0 - Ray Docs"
[3]: https://docs.ray.io/en/latest/tune/index.html?utm_source=chatgpt.com "Ray Tune: Hyperparameter Tuning — Ray 2.53.0 - Ray Docs"
[4]: https://docs.ray.io/en/latest/tune/faq.html?utm_source=chatgpt.com "Ray Tune FAQ — Ray 2.53.0"
