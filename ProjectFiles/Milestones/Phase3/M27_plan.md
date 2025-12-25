Below is a **Cursor-ready handoff prompt for M27**, grounded directly in the **M26 audit and summary** you attached. This milestone deliberately shifts from *training readiness* to **end-to-end learning validation + evaluation loops**, without overreaching into premature tuning.  

---

## Cursor Handoff Prompt — **M27: End-to-End Training Validation & Evaluation Loop**

### Context

* **M26 is complete** and audited green (4.2/5). The system now supports:

  * JAX/Flax/Optax training with Orbax checkpointing
  * Deterministic dataset seeding (`golden-v2`, 100 traces)
  * Metrics artifacts + frontend visualization
  * GPU/CPU device selection and provenance
* The audits explicitly recommend that **M27 focus on validating that the system actually learns**, not adding new infrastructure. 

---

## 🎯 M27 North Star

**Prove end-to-end learning.**

By the end of M27, we should be able to:

1. Run a **full training loop** on `golden-v2`
2. Show **monotonic or near-monotonic loss decrease**
3. Automatically **evaluate checkpoints**
4. Surface **training + evaluation results** in a minimally polished UI
5. Do all of this **without breaking CI or increasing default costs**

This milestone is about *correctness and credibility*, not performance or tuning.

---

## Phase 0 — Baseline Gate (Non-Negotiable)

**Acceptance**

* CI remains fully green:

  * Backend (3.11 / 3.12)
  * Frontend tests
  * E2E smoke
* No training dependencies required in default CI jobs.
* All new logic is gated behind flags or optional workflows.

---

## Phase 1 — Full Convergence Training Run (Offline / Manual)

### Why

M26 proved *we can train*. M27 must prove *training converges*.

### Tasks

1. Add a **documented “full run” config**:

   * Example: `configs/train_golden_v2.yaml`
   * Fixed seed, batch size, LR, steps (e.g. 2–5k steps)
2. Run `train_jax.py` locally (CPU acceptable, GPU preferred):

   ```bash
   uv run python training/train_jax.py \
     --config configs/train_golden_v2.yaml \
     --dataset golden-v2 \
     --checkpoint_dir artifacts/checkpoints/golden-v2-run
   ```
3. Verify:

   * Loss decreases meaningfully over time
   * Checkpoints are written and resumable
   * Metrics artifacts are complete and readable

**Acceptance**

* Attach a short markdown note or artifact:

  * initial loss vs final loss
  * screenshot or JSON excerpt from `metrics.jsonl`

---

## Phase 2 — Checkpoint Evaluation Hook

### Why

Training without evaluation is incomplete; we need a feedback loop.

### Tasks

1. Implement an **evaluation function** (minimal, deterministic):

   * Input: checkpoint path + dataset key
   * Output: scalar score(s) (loss, accuracy proxy, judge score, etc.)
2. Trigger evaluation:

   * At end of training
   * Optionally every `N` checkpoints
3. Persist:

   * Evaluation results as artifact (`eval.json`)
   * Summary metrics back onto the `TunixRun` record (final only)

**Guardrail**

* Evaluation must be fast and deterministic.
* No new heavy deps unless already in `training` extras.

**Acceptance**

* A trained run has:

  * checkpoints
  * metrics.jsonl
  * eval.json
  * DB summary populated

---

## Phase 3 — Backend Wiring: Training → Evaluation → Run State

### Tasks

1. Extend run lifecycle states:

   * `training → evaluating → completed`
2. Ensure failures are explicit:

   * `evaluation_failed` vs `training_failed`
3. Add API support to fetch:

   * latest evaluation summary for a run

**Acceptance**

* Backend run state accurately reflects training + evaluation progression.

---

## Phase 4 — Frontend: Close the Loop

### Tasks

1. Extend the Run Detail page to show:

   * Final evaluation score(s)
   * Link to checkpoints / artifacts
2. Improve metrics UX slightly:

   * Streaming updates or periodic refresh (polling acceptable)
   * Clear “training completed” indicator

**Acceptance**

* A non-technical user can answer:

  > “Did this run train successfully, and how well did it do?”

---

## Phase 5 — Documentation & Narrative (Quick Win)

### Tasks

1. Add `docs/training_end_to_end.md`:

   * How to seed dataset
   * How to run full training
   * How to evaluate
   * What “success” looks like
2. Update `VISION.md` or equivalent:

   * Explicitly state that Tunix RT now supports **closed-loop training + evaluation**

This materially boosts audit score and external credibility.

---

## Definition of Done (M27)

* ✅ At least one **full training run** completed on `golden-v2`
* ✅ Loss demonstrably decreases
* ✅ Checkpoints evaluated automatically
* ✅ Evaluation results visible via API + UI
* ✅ CI remains green and unaffected
* ✅ Clear documentation exists for repeating the experiment

---

## What M27 Is *Not*

* ❌ Hyperparameter tuning (that’s later, now meaningful)
* ❌ Distributed or multi-GPU training
* ❌ Performance optimization
* ❌ Fancy dashboards

Those belong to **M28+**, once learning is proven.

---

## Why This Milestone Matters

After M27, you can honestly say:

> “This system doesn’t just log traces or run models — it **learns**, evaluates itself, and surfaces results end-to-end.”

That’s the inflection point where:

* Tuning becomes valuable
* Research demos become credible
* External reviewers (and acquirers) start paying attention

---

If you want, the **next milestone (M28)** naturally becomes:

> *“Evaluation rigor + tuning readiness (stable metrics, frozen datasets, Ray Tune activation)”*

But first: **prove learning**.
