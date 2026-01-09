# M43 Milestone Summary: Full GPU Training Run

**Milestone:** M43  
**Status:** ✅ Complete  
**Date Completed:** 2026-01-08  
**Hardware:** NVIDIA GeForce RTX 5090 (32GB, sm_120)

---

## Executive Summary

M43 executed a **full production training run** on the RTX 5090 GPU. This was an **evidence-only milestone** — no code changes, no submission modifications. The goal was psychological closure and hardware validation before final video/submission.

> "M43 was executed post-submission-freeze.  
> M42 remains the authoritative submission artifact regardless of M43 outcome."

---

## Training Results

| Metric | Value |
|--------|-------|
| Model | google/gemma-2b (PyTorch) |
| Dataset | dev-reasoning-v2 (550 samples) |
| Training Backend | PyTorch/Transformers |
| Steps | 138 (1 epoch) |
| Runtime | 73.2 seconds |
| Throughput | 7.51 samples/sec, 1.88 steps/sec |
| Loss (start) | 2.21 |
| Loss (end) | 0.76 |
| Final Train Loss | 1.02 |

### Loss Curve (logged every 10 steps)

| Step | Loss | Learning Rate | Epoch |
|------|------|---------------|-------|
| 10 | 2.21 | 1.87e-5 | 0.07 |
| 20 | 1.42 | 1.72e-5 | 0.15 |
| 30 | 1.24 | 1.58e-5 | 0.22 |
| 40 | 0.95 | 1.43e-5 | 0.29 |
| 50 | 0.87 | 1.29e-5 | 0.36 |
| 60 | 0.82 | 1.14e-5 | 0.44 |
| 70 | 0.73 | 1.00e-5 | 0.51 |
| 80 | 0.91 | 8.55e-6 | 0.58 |
| 90 | 0.87 | 7.10e-6 | 0.65 |
| 100 | 0.90 | 5.65e-6 | 0.73 |
| 110 | 0.93 | 4.20e-6 | 0.80 |
| 120 | 0.78 | 2.75e-6 | 0.87 |
| 130 | 0.76 | 1.30e-6 | 0.95 |

---

## Hardware Configuration

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 32 GB |
| Compute Capability | sm_120 (Blackwell) |
| Driver | 576.88 |
| CUDA (PyTorch) | 12.8 |
| PyTorch | 2.11.0.dev20260102+cu128 (nightly) |
| Python | 3.12.10 |

---

## JAX CUDA Attempt (Failed)

Initially attempted JAX/Flax path as specified in M43 plan. **Failed due to Windows limitation:**

- JAX CUDA (`jax[cuda12]`) only available for JAX ≤0.4.21
- JAX 0.4.21 incompatible with Flax 0.12 (requires JAX ≥0.8.1)
- No CUDA plugin available for Windows

**Resolution:** Pivoted to PyTorch path (already proven on 5090 in M40). Same Gemma 2B model, different backend.

---

## Evaluation Pass

Ran inference on 20 examples from `eval_v2.jsonl`:
- Model loads successfully ✅
- Generation works ✅
- Outputs saved to `predictions.jsonl` ✅

**Note:** Model outputs show training data pattern learning rather than Q&A behavior. This is expected for:
- 138 steps of training
- Base model (gemma-2b) instead of instruction-tuned (gemma-2b-it)
- SFT without proper chat template formatting

For production quality, would need:
- More training steps (1000+)
- Instruction-tuned base model
- Better prompt formatting

---

## Artifacts Produced

| File | Description |
|------|-------------|
| `env_snapshot.txt` | Python, PyTorch, CUDA versions |
| `gpu_snapshot.txt` | nvidia-smi output |
| `training_log.txt` | Full training output with loss curves |
| `training_output/final_model/` | Saved model checkpoint (5GB) |
| `training_output/metrics.jsonl` | Training metrics |
| `predictions.jsonl` | Eval generation outputs (20 examples) |
| `eval_log.txt` | Evaluation script output |
| `m43_summary.md` | This file |

---

## Impact on Submission

**None.**

- M42 submission ZIP remains unchanged
- No code was modified
- No configuration was modified
- This run is evidence-only, post-freeze

---

## Lessons Learned

1. **JAX CUDA is Linux-only** for modern versions. Windows requires WSL2.
2. **PyTorch nightly cu128** works excellently on RTX 5090 (sm_120).
3. **Gemma 2B fits comfortably** in 32GB VRAM with bfloat16.
4. **138 steps** is too few for meaningful instruction-following behavior.
5. **The 5090 is fast** — 7.5 samples/sec on 2B parameter model.

---

## Conclusion

M43 successfully validated the local GPU training pipeline on production hardware. While the model's evaluation outputs aren't production-quality (expected for this training duration), the **infrastructure works correctly**:

- ✅ GPU acceleration active
- ✅ Training loop completes
- ✅ Checkpoints save correctly
- ✅ Inference works
- ✅ Evidence captured

**Next steps:** Record demo video, submit to Kaggle.
