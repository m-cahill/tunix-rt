Below is a **clean, authoritative Cursor handoff prompt** to open **M37: TPU Training for Submission** and formally close M36. This is written as a milestone kickoff, with scope control, guardrails, and enterprise-grade framing.

---

## 🚀 Cursor Handoff Prompt — **M37: TPU Training for Submission (Gemma + Tunix)**

```md
# Milestone M37 — TPU Training for Submission

## Context / Status
- M36 is COMPLETE and CLOSED.
  - GPU smoke testing is green using `smoke_tiny.yaml`
  - Pipeline validated end-to-end
  - Memory guardrails and CI stability are in place
- Determination is FINAL:
  - Gemma-scale models will NOT be trained on Kaggle GPU
  - Real training for submission MUST run on TPU (or equivalent large hardware)

We are now beginning **M37**, focused on **TPU-based training suitable for final submission**.

---

## M37 Goal (North Star)

Successfully run **real training** of a Gemma-family model on **Kaggle TPU** using Tunix/JAX,
producing:
- valid checkpoints
- training metrics
- sample outputs
- reproducible execution steps

This is the **submission-critical milestone**.

---

## Scope (Very Important)

### IN SCOPE
1. TPU execution path
2. Gemma-family model training (Gemma2 2B or Gemma3 1B — exact choice documented)
3. Tunix-compatible training loop
4. Stable configs for TPU
5. Minimal documentation for reproducibility
6. Guardrails to prevent GPU misuse

### OUT OF SCOPE
- Model quality optimization / hyperparameter sweeps
- Multi-run experiments
- Performance tuning beyond “it trains correctly”
- PyTorch migration
- New features unrelated to training

M37 is about **correctness + reproducibility**, not leaderboard chasing.

---

## Phase Breakdown

### Phase 0 — Close M36 (Administrative, No Code)
- Ensure M36 artifacts are committed:
  - `smoke_tiny.yaml`
  - smoke config override logic
  - memory guardrails
- Tag or note M36 as closed in repo/docs (if applicable)

---

### Phase 1 — TPU Execution Wiring

**Objective:** Make the training code run cleanly on Kaggle TPU.

Tasks:
1. Add explicit TPU detection / selection logic:
   - JAX should automatically bind to TPU when available
   - Log device platform clearly at startup (TPU vs GPU vs CPU)

2. Add a TPU-specific config:
   - e.g. `training/configs/submission_tpu.yaml`
   - Derived from `submission_gemma_flax.yaml`
   - TPU-appropriate batch sizes, seq length, optimizer
   - No smoke overrides here — this is “real training”

3. Verify no GPU-only assumptions remain:
   - CUDA-specific env vars should not break TPU
   - Allocator flags should be gated or TPU-safe

---

### Phase 2 — Gemma Model Selection & Load

**Objective:** Lock and document the exact model used for submission.

Tasks:
1. Select ONE of:
   - Gemma 2B
   - Gemma 3 1B  
   (Choice must be documented with rationale.)

2. Ensure:
   - Correct Hugging Face model ID
   - Correct revision / branch (e.g. `revision="flax"` if required)
   - TPU-compatible dtype (bf16 preferred)

3. Add a **preflight log**:
   - Model name
   - Parameter count (approx)
   - Target device (TPU)

This is for auditability and write-up clarity.

---

### Phase 3 — End-to-End TPU Training Run (Minimal)

**Objective:** Prove the model actually trains on TPU.

Tasks:
1. Run a **short but real** training job on TPU:
   - More than smoke (e.g. hundreds or thousands of steps)
   - But not a full-scale, hours-long run yet

2. Verify:
   - Training progresses
   - Loss is finite
   - No device or memory errors
   - Checkpoints are written
   - Metrics are logged

3. Save:
   - One or more checkpoints
   - A small set of sample generations

---

### Phase 4 — Guardrails & Documentation

**Objective:** Make this robust and reviewer-friendly.

Tasks:
1. Add a **runtime guardrail**:
   - If model > 1B params AND device == GPU AND training:
     - Warn loudly or refuse to run
     - Suggest TPU explicitly

2. Add minimal documentation:
   - “How to run TPU training for submission”
   - Required Kaggle settings (TPU v3-8)
   - Exact command / notebook cell

Keep this short and precise.

---

## Definition of Done (M37)

M37 is complete when:

- ✅ Training runs successfully on Kaggle TPU
- ✅ Gemma model loads and trains without hacks
- ✅ Checkpoints + metrics are produced
- ✅ TPU config is committed
- ✅ GPU misuse is prevented by guardrails
- ✅ Clear instructions exist to reproduce the run

At that point, the project is **submission-ready**.

---

## Engineering Tone / Constraints

- Prefer explicit configs over magic conditionals
- Keep diffs minimal and reviewable
- Add guardrails instead of comments
- Assume an external reviewer will read the logs and configs
- This is **enterprise-grade research code**, not a hackathon notebook

---

## Final Note

This milestone is about **credibility**:
- Correct hardware
- Correct tooling
- Correct framing

Do not attempt to “make GPU work anyway.”
That decision is closed.

Proceed with M37.
```

---

If you want, next I can:

* Help you choose **Gemma 2B vs Gemma 3 1B** strategically
* Draft the **exact Kaggle TPU notebook cell**
* Create an **ADR-style decision record** for M36→M37 (very strong for reviewers)

Just tell me which one to do next.
