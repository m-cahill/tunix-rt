Below is a **Cursor handoff prompt** for the next milestone after **M34**. It assumes M34 delivered the mini-sweep plumbing (primary score + sweep runner + evidence artifacts) and now we shift to **Quality Loop 1**: make the evaluation signal *meaningful*, *stable*, and *actionable* (leaderboard + comparisons + regressions), so the next optimization passes aren’t “tuning to noise.” 

---

## Cursor Prompt — **M35: Quality Loop 1 (Eval Signal Hardening + Leaderboard Fidelity + Regression Guardrails)**

**Act as a Principal ML Engineer / Tech Lead.**
You are working in the `tunix-rt` repo. **M34 is complete and CI is green.** M34 introduced (at minimum): a **primary score** concept, a **Ray Tune mini-sweep runner**, and **evidence artifacts** in `submission_runs/…` plus tests validating evidence schemas. 

### Objective (M35)

Make the system’s evaluation/leaderboard loop **credible and decision-grade** by:

1. **Hardening evaluation semantics** (deterministic scoring rules, robust normalization, clear aggregation).
2. **Improving leaderboard fidelity** (the UI/API should answer “is run B better than run A?”).
3. **Adding regression guardrails** so improvements don’t silently break quality or evaluation correctness.

### Non-goals

* Do **not** expand model architecture.
* Do **not** build a huge new dataset pipeline unless required to strengthen evaluation.
* Do **not** add long-running training to CI.

---

# Phase 0 — Baseline Gate (must pass before changes)

1. Branch from `main`: `milestone/M35-quality-loop-1`.
2. Run the full local quality gate (use repo standard commands):

   * ruff check + format check
   * mypy
   * backend tests
   * frontend tests (if applicable)
   * e2e tests (if applicable)

**Acceptance:** CI would be green with no code changes.

---

# Phase 1 — Eval Set v2 (increase signal, reduce variance)

## 1.1 Create an eval set that’s *purpose-built* for scoring

* Add a new evalset file (keep it simple + versioned), e.g.:

  * `training/evalsets/eval_v2.jsonl`
* Target **~75–150 items** (enough to stabilize a mean score).
* Composition guidance:

  * A core “answer correctness” section (short deterministic answers).
  * A small “trace sensitivity” section (where reasoning steps matter but answer is still checkable).
  * A small “edge cases” section (formatting, whitespace, casing, numeric equivalence).

## 1.2 Add tooling to validate eval set schema

* Add a small validator in `backend/tools/` (or wherever tools live) that:

  * loads eval JSONL
  * validates required fields
  * prints summary stats (#items, category counts)
* Add unit tests validating `eval_v2.jsonl` loads and meets minimum size/category constraints.

**Acceptance:**

* `eval_v2.jsonl` exists, validated by tests.
* Validator prints useful stats and exits non-zero on schema failure.

---

# Phase 2 — Scoring semantics (deterministic, explicit, testable)

M34 introduced a `primary_score` concept. Now formalize it.

## 2.1 Define “Primary Score” precisely

Implement or refine `compute_primary_score(evaluations) -> float | None` so that:

* It uses one **canonical metric** (prefer `metrics["answer_correctness"]` if present).
* It is robust to missing rows.
* It is stable (no randomness, no dependence on ordering).

Add docstrings that explain:

* input schema (what evaluation rows must contain)
* how missing/invalid items are treated
* what the scale means (0–1 or 0–100 — pick one and standardize)

## 2.2 Add “Scorecard” outputs

Add an additional aggregator (separate from primary score) that returns a structured summary:

* `n_items`
* `n_scored`
* `n_skipped`
* per-category breakdown (if eval items are labeled)
* mean, stddev (and optionally a simple confidence interval if trivial)

**Acceptance:**

* Unit tests cover:

  * empty eval list
  * partial missing metrics
  * perfect/zero correctness
  * mixed correctness
* Aggregators are pure + deterministic.

*(If Ray Tune is used again later, it should optimize the exact value shown on leaderboard. Ray Tune expects reporting of metrics via its trainable/report mechanism—keep metric naming consistent.)* ([Ray][1])

---

# Phase 3 — Leaderboard & Run Comparison that answers real questions

## 3.1 Backend: leaderboard endpoints return decision-grade payloads

Update or add endpoints so the UI can:

* list runs sorted by `primary_score`
* filter by dataset, model_id, config, date range
* show scorecard metadata (n, skipped, category means)

## 3.2 Frontend: compare runs side-by-side (minimal but powerful)

Implement a compare view that shows:

* Run A vs Run B:

  * primary score
  * scorecard breakdown
  * per-item diff table (item id, expected, predicted, correctness)
* Highlight where scores differ most.

Keep it lean: no fancy charts required, but ensure it’s readable and stable.

**Acceptance:**

* Unit/API tests for endpoints.
* Frontend tests updated/added for compare view basic rendering.
* No broken API client exports (guardrails remain intact).

---

# Phase 4 — Regression Guardrails (prevent “goodharting” + silent breakage)

## 4.1 Add a regression baseline lock

* Ensure there is a “baseline run” definition or mechanism.
* Add a regression check that:

  * evaluates candidate run on `eval_v2`
  * compares to baseline
  * fails only when there is a clear drop beyond a threshold (configurable)

## 4.2 Determinism check (cheap but valuable)

Add a test/tool that runs evaluation twice (same run/evalset) and asserts:

* identical primary score
* identical scorecard counts
* (optionally) identical per-item correctness

**Acceptance:**

* Determinism check exists and is documented.
* Regression baseline check is present (even if it’s informational-only at first).

---

# Phase 5 — Evidence v2 + Packaging alignment

Create `submission_runs/m35_v1/` evidence folder with:

* `run_manifest.json` (include commit SHA, model_id, dataset, config, command)
* `eval_summary.json` (now must include **non-null** `primary_score` and scorecard fields)
* `kaggle_output_log.txt` (or equivalent run log)

Update packaging tool (if needed) so:

* `--run-dir submission_runs/m35_v1` bundles evidence cleanly
* archive naming uses correct prefix/version

Add/extend schema tests:

* required keys exist
* `primary_score` is non-null for M35 evidence

**Acceptance:**

* Evidence files committed (small text/json only).
* Packaging produces a zip that includes evidence.
* Evidence schema tests pass.

---

# Definition of Done (M35)

* CI green.
* `eval_v2.jsonl` exists, validated, and used by evaluation flows.
* Primary score is deterministic and fully tested.
* Leaderboard + run comparison enables real decision-making.
* Regression guardrails exist (baseline + determinism).
* `submission_runs/m35_v1` evidence committed and included in packaging output.

---

## Suggested implementation notes (do *not* overbuild)

* Prefer **pure functions** for aggregation so they’re trivial to test.
* Keep UI changes minimal but high-signal.
* Keep all new metrics/fields **backward compatible** where practical (or bump schema version explicitly).
* If Orbax checkpoint artifacts are involved, keep them out of git and only reference them via evidence metadata (Orbax is designed for structured checkpointing and management; don’t commit large artifacts). ([Google Cloud][2])

---

If you want **M36** to be the next step after M35, it should likely be: “Actual Kaggle GPU/TPU execution run + evidence lock with real (non-smoke) artifacts + final video capture,” but M35 should come first so you aren’t judging improvements with a shaky scoring signal.

[1]: https://docs.ray.io/en/latest/tune/api/trainable.html "Training in Tune (tune.Trainable, tune.report) — Ray 2.53.0"
[2]: https://cloud.google.com/blog/products/compute/unlock-faster-workload-start-time-using-orbax-on-jax?utm_source=chatgpt.com "Unlock faster workload start time using Orbax on JAX"
