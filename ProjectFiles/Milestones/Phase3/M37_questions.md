# M37 Clarifying Questions

Before proceeding with M37 implementation, I need clarity on the following items:

---

## 1. Model Selection (Critical)

The M37 plan mentions choosing between **Gemma 2B** and **Gemma 3 1B**. However, M36 documentation explicitly states:

> "Gemma 2 2B and Gemma 3 are NOT supported by FlaxAutoModelForCausalLM"

The current working config uses `google/gemma-2b` (Gemma 1) with `revision="flax"`.

**Question:** Which model should we use for M37 TPU training?

- **Option A:** Stick with `google/gemma-2b` (Gemma 1, Flax-native, proven to work)
- **Option B:** Attempt `google/gemma-2-2b-it` (Gemma 2, may require PyTorch→Flax conversion)
- **Option C:** Attempt `google/gemma-3-1b-it` (Gemma 3, likely no Flax support yet)

**My recommendation:** Option A — the infrastructure is already validated for Gemma 1 2B. Switching models adds risk for minimal benefit in a submission milestone.

---

## 2. TPU Device Selection

Current `train_jax.py` has device options: `["auto", "cpu", "gpu"]` — no explicit `tpu` option.

JAX auto-detects TPU when running on Kaggle TPU, so `auto` should work. However, the M37 plan mentions:

> "Add explicit TPU detection / selection logic"

**Question:** Should I:

- **Option A:** Add `--device tpu` as an explicit option (with validation that TPU is available)
- **Option B:** Keep `auto` and add TPU-specific logging only (JAX handles detection)
- **Option C:** Both — add `tpu` option but also improve logging for `auto` to show TPU detection clearly

---

## 3. TPU-Specific Config

The plan mentions:

> "Add a TPU-specific config: e.g. `training/configs/submission_tpu.yaml`"

**Question:** Do you want a new config file, or should we reuse `submission_gemma_flax.yaml`?

- **Option A:** Create new `submission_tpu.yaml` (explicit TPU settings, different batch sizes)
- **Option B:** Reuse `submission_gemma_flax.yaml` (already works, less config sprawl)

**If Option A:** What should differ from the existing config? Larger batch sizes? Different optimizer settings?

---

## 4. Training Duration

The plan says:

> "More than smoke (e.g. hundreds or thousands of steps)"

Current `submission_gemma_flax.yaml` has `num_steps: 100`.

**Question:** How many steps for the M37 TPU validation run?

- Minimal validation: 50-100 steps (~15-30 min)
- Short production: 200-500 steps (~1-2 hours)
- Full training: 1000+ steps (~4+ hours)

---

## 5. Guardrail Strictness

The plan mentions:

> "If model > 1B params AND device == GPU AND training: Warn loudly or refuse to run"

Current behavior is **warn only**. M36 implemented this.

**Question:** Should M37 upgrade this to a **hard block** (exit with error), or keep it as a warning?

- **Option A:** Hard block — refuse to run Gemma on GPU (force user to use TPU or smoke config)
- **Option B:** Keep warning — let users attempt at their own risk

---

## 6. Evidence Structure

M36 created `submission_runs/m36_v1/`. 

**Question:** For M37 TPU run evidence, should I:

- **Option A:** Create new `submission_runs/m37_v1/` folder
- **Option B:** Update `submission_runs/m36_v1/` with TPU evidence (overwrite GPU smoke evidence)
- **Option C:** Create both — keep M36 as smoke reference, M37 as TPU production

---

## 7. Per-Item Artifact Storage (Scope Confirmation)

M36 audit mentioned:

> "M37-03: Per-item artifact storage — predictions.jsonl persisted"

The current pipeline already writes `predictions.jsonl` to the output directory.

**Question:** Is additional persistence (e.g., database storage, API endpoint) required, or is file-based persistence sufficient for M37?

---

## 8. Kaggle TPU Access

**Question:** Do you have access to Kaggle TPU quota? 

Kaggle sometimes has limited TPU availability or requires additional verification.

If access is uncertain, should I design the M37 deliverables to be "TPU-ready" without requiring a live TPU run?

---

## 9. Minimum Viable Scope

Given this is a submission milestone, what is the absolute minimum for M37 to be considered DONE?

My understanding from the plan:

1. ✅ Training runs on Kaggle TPU (even briefly)
2. ✅ Gemma loads and trains without errors
3. ✅ Checkpoints + metrics are produced
4. ✅ TPU config is committed
5. ✅ GPU guardrails prevent Gemma OOM
6. ✅ Clear reproduction instructions

**Question:** Is there anything else critical that would block submission if missing?

---

## Summary

Please respond to these questions so I can finalize the implementation plan. Key decisions needed:

| # | Topic | Your Choice |
|---|-------|-------------|
| 1 | Model | A (Gemma 1 2B) / B (Gemma 2) / C (Gemma 3) |
| 2 | TPU device option | A (add --device tpu) / B (auto only) / C (both) |
| 3 | TPU config | A (new file) / B (reuse existing) |
| 4 | Training steps | Number: ___ |
| 5 | GPU guardrail | A (hard block) / B (warning) |
| 6 | Evidence folder | A (new m37_v1) / B (overwrite m36) / C (both) |
| 7 | Per-item storage | File only / DB storage needed |
| 8 | TPU access | Yes / Not yet / Design TPU-ready only |
| 9 | Anything else? | ___ |

Once I have your answers, I'll create the detailed todo list and begin implementation.
