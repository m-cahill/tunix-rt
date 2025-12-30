# M37 Milestone Summary — TPU Training for Submission

**Status:** ✅ COMPLETE
**Date:** December 30, 2025
**Commit:** `b49fb4d94f871fc6e7ee2387fe965bda1f169a5e`
**Branch:** `milestone/model-pivot-gemma2b-flax`

---

## Executive Summary

M37 established a **production-ready TPU execution path** for the Tunix Hackathon submission. The milestone focused on ensuring Gemma 2B can be trained reliably on Kaggle TPU v3-8, with explicit device selection, guardrails to prevent GPU misuse, and comprehensive documentation.

### Key Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| `--device tpu` option | ✅ | `training/train_jax.py` |
| GPU hard block | ✅ | `training/train_jax.py` |
| TPU config | ✅ | `training/configs/submission_tpu.yaml` |
| Notebook m37_v1 | ✅ | `notebooks/kaggle_submission.ipynb` |
| Evidence folder | ✅ | `submission_runs/m37_v1/` |
| Documentation | ✅ | `tunix-rt.md`, inline comments |

---

## Technical Changes

### 1. TPU Device Support (`train_jax.py`)

Added explicit `--device tpu` argument with validation:

```python
parser.add_argument("--device", type=str, 
    choices=["auto", "cpu", "gpu", "tpu"], default=None)
```

TPU device selection with clear error handling:

```python
elif requested_device == "tpu":
    try:
        tpus = jax.devices("tpu")
        if not tpus:
            raise RuntimeError("No TPU found")
        print(f"   Device request: TPU ({len(tpus)} cores available)")
    except RuntimeError:
        print("❌ Device 'tpu' requested but no TPU devices found.")
        print("   On Kaggle: Settings → Accelerator → TPU v3-8")
        sys.exit(1)
```

### 2. Enhanced Device Logging

All training runs now log:
- Platform (CPU/GPU/TPU)
- Device count
- All devices (useful for multi-chip TPU)

```
Platform: TPU
Device Count: 8
Active Device: TpuDevice(id=0, ...)
   [0] TpuDevice(id=0, ...)
   [1] TpuDevice(id=1, ...)
   ...
```

### 3. GPU Hard Block

Upgraded from warning (M36) to hard block (M37):

```python
if is_large_model and device_req == "gpu" and not is_smoke:
    print("❌ ERROR: Large Model + GPU Training Blocked")
    print("   Models >1B params cause OOM on consumer GPUs (e.g., T4).")
    print("   This was empirically verified in M36.")
    print("   Solutions:")
    print("   • Use --device tpu (Kaggle TPU v3-8 has 64GB)")
    print("   • Use --smoke_config with a tiny model for smoke tests")
    sys.exit(1)
```

### 4. TPU Config (`submission_tpu.yaml`)

New config optimized for Kaggle TPU v3-8:

```yaml
model:
  name: "google/gemma-2b"
  revision: "flax"
  max_length: 512

training:
  num_steps: 200
  batch_size: 8
  per_device_batch_size: 8
  learning_rate: 1.0e-5
  device: "tpu"
```

### 5. Kaggle Notebook (m37_v1)

Updated notebook with:
- Version bump to `m37_v1`
- Separate `DEVICE_SMOKE` (auto) and `DEVICE_FULL` (tpu) variables
- Pre-flight TPU detection before training cell
- Clear error with instructions if GPU detected

```python
backend = jax.default_backend()
if backend != "tpu":
    print("❌ ERROR: TPU NOT DETECTED")
    print("   To fix: Settings → Accelerator → TPU v3-8")
    raise RuntimeError("TPU required for full training.")
```

### 6. Evidence Folder (`submission_runs/m37_v1/`)

Created evidence templates for TPU run:

| File | Purpose |
|------|---------|
| `run_manifest.json` | TPU hardware fields, training params |
| `eval_summary.json` | Scorecard template |
| `kaggle_output_log.txt` | Expected output format |

---

## Decision Record

### Model Selection: Gemma 1 2B (Confirmed)

**Decision:** Use `google/gemma-2b` with `revision="flax"`

**Rationale:**
- Proven to load and train in Flax
- Gemma 2/3 NOT supported by `FlaxAutoModelForCausalLM`
- M37 is submission-critical; no time for model migration experiments

### GPU Guardrail: Hard Block (Upgraded)

**Decision:** Exit with error if Gemma + GPU + non-smoke

**Rationale:**
- M36 empirically verified GPU OOM
- Warnings were ignored; hard block prevents wasted time
- Clear error message with actionable solutions

### Training Steps: 200 (Evidence Run)

**Decision:** 200 steps for M37 TPU validation

**Rationale:**
- Long enough to prove stability
- Short enough to conserve TPU quota
- Not meant to be final production quality run

---

## Files Changed

| File | Change |
|------|--------|
| `training/train_jax.py` | Added `--device tpu`, GPU hard block, enhanced logging |
| `training/configs/submission_tpu.yaml` | NEW — TPU-optimized config |
| `notebooks/kaggle_submission.ipynb` | Updated to m37_v1, TPU detection |
| `submission_runs/m37_v1/run_manifest.json` | NEW — evidence template |
| `submission_runs/m37_v1/eval_summary.json` | NEW — scorecard template |
| `submission_runs/m37_v1/kaggle_output_log.txt` | NEW — output format guide |
| `tunix-rt.md` | Added M37 summary |
| `ProjectFiles/Milestones/Phase3/M37_questions.md` | Clarifying questions |
| `ProjectFiles/Milestones/Phase3/M37_answers.md` | Decision answers |

---

## Test Results

| Suite | Count | Status |
|-------|-------|--------|
| Backend | 399 | ✅ Pass |
| Frontend | 75+ | ✅ Pass |
| E2E | 3 specs | ✅ Pass |
| Pre-commit | All hooks | ✅ Pass |

---

## Definition of Done (M37)

| Criterion | Status |
|-----------|--------|
| ✅ Training code supports `--device tpu` | Complete |
| ✅ GPU hard block for Gemma | Complete |
| ✅ TPU config committed | Complete |
| ✅ Notebook updated for TPU | Complete |
| ✅ Evidence folder created | Complete |
| ✅ Documentation updated | Complete |

---

## What's Next: M38

M37 established the **infrastructure**. M38 should execute the **real TPU training**:

1. **Execute TPU run on Kaggle** — actually train Gemma 2B on TPU v3-8
2. **Populate evidence files** — fill in real values from run
3. **Frontend coverage uplift** — target 50%+ for remaining components
4. **Per-item artifact storage** — persist predictions.jsonl to database

---

## Retrospective

### What Went Well
- Clear decision-making via Q&A format prevented scope creep
- GPU hard block is a significant UX improvement
- Documentation-first approach made implementation straightforward

### What Could Improve
- Should have included a local TPU simulation test (Colab TPU)
- Evidence templates could include validation schema

### Lessons Learned
- Submission milestones should focus on **correctness over features**
- Hard guardrails are more effective than warnings
- Explicit device selection is better than "auto" for production runs

---

## Appendix: Commands

### Run Smoke Test (GPU/CPU)
```bash
python training/run_train_jax.py \
    --config training/configs/submission_tpu.yaml \
    --smoke_config training/configs/smoke_tiny.yaml \
    --output ./output/smoke_run \
    --dataset dev-reasoning-v2 \
    --device auto \
    --smoke_steps 2
```

### Run Full Training (TPU)
```bash
python training/run_train_jax.py \
    --config training/configs/submission_tpu.yaml \
    --output ./output/tpu_run \
    --dataset dev-reasoning-v2 \
    --device tpu \
    --save_every_steps 50
```

### Package Submission
```bash
python backend/tools/package_submission.py --run-dir submission_runs/m37_v1
```
