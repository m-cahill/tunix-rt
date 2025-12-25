M24 should be your **first “real signal” milestone**: replace the last stub (real inference), run a **baseline experiment** end-to-end on `golden-v1`, and lock down **dependency determinism** so training results are reproducible. This is exactly what the M23 audit calls out next.  

Below is a **Cursor handoff prompt** for M24.

---

## Cursor Handoff Prompt — M24: Real Inference + Baseline Experiment + Reproducible Training Setup

### Context

* M23 hardened evaluation + contracts; `AnswerCorrectnessJudge` is real **but inference is still stubbed** (`generate_predictions` writes placeholder text). M24 must implement **real model inference** so evaluation is meaningful. 
* Full audit flags a **high-priority supply chain risk**: backend has no lockfile; fix before training to prevent dependency drift. 
* Goal: run “Base vs Trained” on `golden-v1`, persist metrics, and promote best model in Registry.

---

# Phase 0 — Baseline Gate (mandatory)

1. Pull `main`, ensure CI-equivalent local checks pass (backend/frontend/e2e).
2. Confirm `golden-v1` exists and exports consistently.
3. Confirm evaluation path currently fails if `predictions.jsonl` missing (expected behavior from M23).

**Exit:** baseline green and contracts verified.

---

# Phase 1 — Backend Dependency Lockfile (training safety, quick audit win)

**Decision:** adopt **uv lockfile** and use `--locked` in CI to guarantee deterministic installs. (`uv.lock` is designed for lock+sync workflows.) ([Astral Docs][1])

### Tasks

1. Add `uv` project config (if not already).
2. Generate and commit `backend/uv.lock`.
3. Update backend CI install step to use `uv sync --locked` (or equivalent) rather than free-resolving dependencies.
4. Add a CI check step: `uv lock --check` (or your chosen equivalent) so drift is caught immediately.

**Acceptance**

* `uv.lock` committed
* CI uses locked install path
* No dependency-resolution drift between runs ([Astral Docs][1])

**Guardrail:** keep optional extras (`[tuning]`, `[training]`, etc.) explicitly modeled so “core CI” doesn’t accidentally install heavyweight stacks.

---

# Phase 2 — Replace Stubbed Inference with Real Inference (highest priority)

M23 audit explicitly calls the stub: M24 must implement actual `model.generate()` logic. 

## 2.1 Implement real `generate_predictions()`

In `backend/tunix_rt_backend/services/tunix_execution.py`:

**Minimal contract**

* Inputs: `run`, `dataset_key`, `model_path` (or model artifact dir), generation config
* Output: `predictions.jsonl` with `{trace_id, prediction}`

**Recommended implementation**

* Use Hugging Face Transformers `generate()` (deterministic baseline: greedy decode).
* Load:

  * tokenizer
  * model (from promoted registry version dir or run output dir)
* Run inference for each dataset item prompt; write predictions.

Transformers `generate()` is the standard entry point for text generation and is configurable via generation parameters. ([Hugging Face][2])

**Determinism guardrails**

* `do_sample = false`
* set explicit `max_new_tokens`
* set seeds
* record generation config into `predictions_meta.json` (optional but recommended)

**Failure modes must be crisp**

* missing model files → fail with actionable message
* empty dataset → fail (you already enforce this pattern)
* write both `predictions.jsonl` and a small metadata sidecar

**Acceptance**

* `predictions.jsonl` contains real generated output (not placeholder)
* Judge consumes it and produces a non-trivial `answer_correctness`

---

# Phase 3 — Baseline Experiment Harness (Base vs Trained)

Goal: one clean experiment that proves the entire loop produces meaningful deltas.

## 3.1 Add “baseline inference-only run”

Add a run mode that:

* does **no training**
* generates predictions for `golden-v1`
* evaluates `answer_correctness`
* persists metrics

This is your “Base Model” score.

## 3.2 Add “micro-train then eval” run

Add a minimal training configuration that finishes quickly:

* tiny steps/epochs
* small batch
* short context
* produce model artifacts
* run the same inference+eval on `golden-v1`

This is your “Trained Model” score.

**Acceptance**

* Two runs exist in DB:

  * Base run with metrics + predictions
  * Trained run with metrics + predictions
* Metrics are visible in UI and persisted (already scaffolded in M22/M23)
* “Best” selection is meaningful (not constant 0/1 or placeholder)

---

# Phase 4 — Registry Promotion + Comparison (end-to-end)

## 4.1 Promote the best run/model version

* Promote the trained run (or whichever run scores higher) into Model Registry
* Ensure the promoted version stores:

  * metrics snapshot
  * dataset key/version
  * judge version (`answer_correctness@v1`)
  * sha256

## 4.2 Add a minimal comparison view

If you already show metrics in Run/Tuning/Registry, only add:

* “Compare to Base” (display delta in UI, no fancy chart required)

**Acceptance**

* Promotion works reliably end-to-end
* Registry version shows real score and provenance

---

# Phase 5 — Tests + CI Guardrails (keep it safe)

## 5.1 Unit tests for inference path

* Use a tiny fake model OR mock the model call to avoid heavy deps in default CI.
* Verify `predictions.jsonl` format and non-empty output.

## 5.2 Integration test (optional, mark as slow / training-smoke)

* A single test that runs inference over 2–3 golden items with a lightweight model, tagged `@smoke` or `@slow`.

**Acceptance**

* Default CI remains fast
* Training smoke remains scheduled/non-blocking unless you decide otherwise

---

# Definition of Done (M24)

* [ ] `generate_predictions()` produces **real** predictions via actual inference (no placeholder). 
* [ ] Backend dependency lockfile added and enforced in CI (no drift).  ([Astral Docs][1])
* [ ] Base vs Trained run executes end-to-end on `golden-v1` with meaningful metrics. 
* [ ] Best run promoted to Registry with correct provenance + metrics snapshot.
* [ ] CI green; heavy training/inference is either mocked or isolated behind smoke markers.

---

## Why M24 before “real training”

M23 proved the **contracts**; M24 makes the **signal real**. Once inference is real and deps are locked, you can begin larger training (M25+) without worrying that improvements are artifacts of drift or placeholder evaluation.  

If you want, I can also draft **M25** immediately after this as: “First full training run on RTX 4080 + checkpointing + small tuning smoke,” but M24 is the clean step that makes everything after it trustworthy.

[1]: https://docs.astral.sh/uv/concepts/projects/sync/?utm_source=chatgpt.com "Locking and syncing | uv - Astral Docs"
[2]: https://huggingface.co/docs/transformers/en/main_classes/text_generation?utm_source=chatgpt.com "Generation"
