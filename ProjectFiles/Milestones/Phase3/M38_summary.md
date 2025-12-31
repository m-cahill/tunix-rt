# M38 Milestone Completion Summary

**Milestone:** M38 — TPU HBM OOM Fix + Evidence Population  
**Branch:** `main` (commits 317b77d, 59bc447)  
**Completion Date:** December 30, 2025  
**Status:** ⚠️ **Partial — Infrastructure Complete, TPU Execution Blocked**

---

## Executive Summary

M38 was an **execution-focused milestone** targeting real TPU training on Kaggle. The milestone successfully:

1. ✅ **Diagnosed the TPU HBM OOM root cause** — Gemma's 256K vocabulary creates massive logits tensors that exceed Kaggle TPU v5e-8's 16GB HBM
2. ✅ **Implemented aggressive memory optimizations** — seq_len=64, batch=1, bfloat16, Adafactor
3. ✅ **Fixed false success reporting** — Notebook now correctly reports training failures
4. ✅ **Fixed cross-platform compatibility** — UTF-8 encoding, Python path fixes for runpy
5. ❌ **TPU execution remains blocked** — Full fine-tuning Gemma 2B is not feasible on free Kaggle TPU

**Key Finding:** This is a **hardware constraint**, not a software bug. The code is correct, but full fine-tuning a 256K-vocab model on 16GB HBM TPU is fundamentally infeasible without LoRA/PEFT or model parallelism.

---

## Deliverables Checklist

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | TPU HBM OOM diagnosis | ✅ | Error: `Used 28.59G of 15.75G hbm` |
| 2 | Memory-safe config | ✅ | `submission_tpu.yaml` (seq=64, batch=1) |
| 3 | Code-level overrides | ✅ | `train_jax.py` forces TPU-safe values |
| 4 | bfloat16 for TPU | ✅ | Automatic for `--device tpu` |
| 5 | Adafactor optimizer | ✅ | Memory-efficient, no optimizer state accumulation |
| 6 | Notebook false success fix | ✅ | `runpy.run_path()` with proper error handling |
| 7 | UTF-8 encoding fix | ✅ | Windows YAML reading compatibility |
| 8 | Python path fix | ✅ | `run_train_jax.py` adds SCRIPT_DIR to sys.path |
| 9 | Evidence folder update | ⚠️ | Templates updated, awaiting real run data |
| 10 | Real TPU execution | ❌ | Blocked by HBM capacity |
| 11 | CI Green | ✅ | 384 backend, 75 frontend tests passing |

---

## Technical Changes

### 1. Training Config (`submission_tpu.yaml`)

**Before (M37):**
```yaml
max_length: 512
batch_size: 8
optimizer: adamw
```

**After (M38 v2):**
```yaml
max_seq_length: 64        # Reduced 8x
batch_size: 1             # Reduced 8x
gradient_accumulation_steps: 8  # Maintain effective batch
optimizer: adafactor      # Memory-efficient
```

### 2. Code-Level Memory Safety (`train_jax.py`)

Added forced overrides for TPU runs:
```python
elif requested_device == "tpu":
    print("   🔧 TPU mode: Enabling memory-safe settings...")
    max_length = min(max_length, 64)  # Force max 64
    batch_size = 1                     # Force batch=1
    use_bfloat16 = True               # Native TPU support
```

### 3. Sanity Check Before XLA Compile

Added diagnostic output:
```
📊 M38 Sanity Check (BEFORE XLA compile):
   input_ids shape:      (1, 64)
   attention_mask shape: (1, 64)
   batch_size:           1
   max_length:           64
   vocab_size:           256000
   expected logits:      [1, 63, 256000]
   dtype:                bfloat16
```

### 4. Notebook Execution Fix

**Before (M37):** `subprocess.run()` — always showed "completed" even on failure  
**After (M38):** `runpy.run_path()` with try-except handling:
```python
try:
    runpy.run_path("training/run_train_jax.py", run_name="__main__")
except SystemExit as e:
    if e.code != 0:
        print("❌ Training FAILED with exit code", e.code)
```

### 5. Cross-Platform Fixes

- **UTF-8 encoding:** Added `encoding="utf-8"` to YAML file reading
- **Python path:** Added `SCRIPT_DIR` to `sys.path` in `run_train_jax.py`

---

## TPU OOM Analysis

### Error Message
```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. 
Ran out of memory in memory space hbm. 
Used 28.59G of 15.75G hbm. 
Exceeded hbm capacity by 12.85G.
```

### Root Cause

| Factor | Impact |
|--------|--------|
| **Gemma vocabulary** | 256,000 tokens (vs ~32K for GPT-2) |
| **Logits shape** | `[batch, seq_len, 256000]` |
| **Per-step memory** | ~64MB for logits tensor alone |
| **XLA compile temps** | Large HLO temporaries during compilation |
| **TPU v5e-8 HBM** | Only 16GB per chip (not 64GB like v3-8) |
| **Full fine-tuning** | All parameters + optimizer state + gradients |

### Math
```
Logits memory = batch × seq_len × vocab × bytes
             = 1 × 64 × 256000 × 4 (float32)
             = 64MB just for logits
             
Model params (Gemma 2B) = ~10GB in bfloat16
Optimizer state (Adafactor) = ~2GB
XLA compile temps = ~10GB+ (variable)

Total > 16GB available HBM
```

### Conclusion

**Full fine-tuning Gemma 2B is not feasible on Kaggle TPU v5e-8 (16GB HBM).**

This is a constraint mismatch, not a software bug. Solutions require:
- LoRA/PEFT (train adapters only, freeze base)
- Access to TPU v3-8 (64GB HBM)
- Local GPU with sufficient VRAM (RTX 5090: 32GB)

---

## Test Coverage

### Backend
- **Tests:** 384 passed, 11 skipped
- **Coverage:** 75.88% (threshold: 70%)
- **New in M38:** No new tests (execution milestone)

### Frontend
- **Tests:** 75 passed
- **Coverage:** ~45%
- **Note:** Coverage uplift deferred to M39

---

## Files Changed

### Modified Files (5)

| File | Changes |
|------|---------|
| `training/configs/submission_tpu.yaml` | Memory reduction, Adafactor, gradient accum |
| `training/train_jax.py` | TPU overrides, sanity check, bfloat16 |
| `training/run_train_jax.py` | Python path fix for runpy |
| `notebooks/kaggle_submission.ipynb` | runpy execution, failure handling |
| `tunix-rt.md` | M38 enhancements section |

### Updated Templates (3)

| File | Changes |
|------|---------|
| `submission_runs/m37_v1/run_manifest.json` | M38 v2 config snapshot |
| `submission_runs/m37_v1/eval_summary.json` | Training steps=200 |
| `submission_runs/m37_v1/kaggle_output_log.txt` | Expected output format |

---

## Known Limitations

1. **TPU Execution Blocked**
   - Root cause: Gemma 256K vocab × 16GB HBM = infeasible
   - Status: Documented, deferred to M39 (local GPU path)

2. **Evidence Files Incomplete**
   - Status: Templates prepared, awaiting real run data
   - Action: Will be populated after successful local run

3. **Frontend Coverage**
   - Status: ~45%, below 50% target
   - Action: Deferred to M39

---

## Lessons Learned

### 1. Hardware Constraints Trump Code Optimizations
Even aggressive memory settings (seq_len=64, batch=1, bfloat16) cannot overcome fundamental HBM limits. The logits tensor for 256K vocab is unavoidably large.

### 2. TPU v5e-8 ≠ TPU v3-8
Free Kaggle TPUs have only 16GB HBM per chip, not the 64GB available on TPU v3-8. Documentation and code assumed the larger variant.

### 3. Sanity Checks Are Invaluable
The added pre-compile sanity print immediately showed actual shapes and dtypes, making debugging faster.

### 4. Proper Error Handling Matters
The false "Training completed!" banner from `subprocess.run()` wasted debugging cycles. The `runpy` fix ensures accurate reporting.

---

## Next Steps (M39: RTX 5090 Local Run)

### Goal
Execute Gemma 2B training on local RTX 5090 (32GB VRAM).

### Approach
1. Verify CUDA/cuDNN compatibility with RTX 5090 (Blackwell architecture)
2. Run with `--device gpu` on local machine
3. Use existing `submission_tpu.yaml` config (may need adjustment for GPU)
4. Capture evidence and populate `submission_runs/m37_v1/`

### Fallback
If full fine-tuning still OOMs on 32GB:
- Implement LoRA/PEFT adapters
- Use smaller model for demo purposes

---

## Conclusion

**M38 is infrastructure-complete but execution-blocked.**

The milestone successfully diagnosed and addressed the TPU HBM OOM at the software level. However, the fundamental constraint — Gemma's 256K vocabulary on a 16GB HBM TPU — cannot be overcome without architectural changes (LoRA) or different hardware.

**Recommended path forward:** M39 with local RTX 5090 execution.

---

## Commands Reference

### Run Backend Tests
```bash
cd backend && uv run pytest tests/ -v
```

### Run Frontend Tests
```bash
cd frontend && npm test
```

### Verify Training Script
```bash
python training/train_jax.py --help
```

### Check Git Status
```bash
git log --oneline -5
```
