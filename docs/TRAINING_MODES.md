# Tunix RT — Training Modes

This document describes the two training modes in the Tunix RT pipeline and their
intended use cases.

---

## Overview

| Mode           | Model              | Hardware | Purpose                        |
|----------------|--------------------|---------|---------------------------------|
| **GPU Smoke**  | `sshleifer/tiny-gpt2` | Kaggle GPU (T4) / Local | Pipeline validation |
| **TPU Full**   | `google/gemma-2b` (Flax) | Kaggle TPU v3-8 | Competition training |

---

## GPU Smoke Testing

**Purpose:** Validate the end-to-end pipeline without training a real model.

### Why a tiny model?

Gemma 2B (and larger) models require ~6-8GB+ VRAM just to load parameters.
Combined with:
- optimizer state (Adam: 2x param memory)
- activations / gradient graph
- JAX's memory allocation behavior

...they consistently OOM on consumer GPUs (T4: 16GB) even with aggressive optimizations:
- bfloat16 weights
- Adafactor optimizer
- batch_size=1, max_length=64
- platform allocator (no preallocation)

This was **empirically verified** through multiple smoke runs on Kaggle.

### Smoke Configuration

```yaml
# training/configs/smoke_tiny.yaml
model:
  name: "sshleifer/tiny-gpt2"
  max_length: 64

training:
  num_steps: 2
  per_device_batch_size: 1
  optimizer: "adafactor"
```

### Running Smoke Tests

```bash
python training/run_train_jax.py \
    --config training/configs/submission_gemma_flax.yaml \
    --smoke_config training/configs/smoke_tiny.yaml \
    --output ./output/smoke_run \
    --dataset dev-reasoning-v2 \
    --device auto \
    --smoke_steps 2
```

**Pass Criteria:**
- Completes 2 training steps
- Writes checkpoint + metrics artifacts
- Exit code 0

---

## TPU Full Training (Gemma)

**Purpose:** Train real competition models with sufficient hardware.

### Why TPU?

Kaggle TPU v3-8 provides:
- 64GB HBM per chip
- 8 chips total (512GB aggregate)
- Optimized for JAX/Flax workloads

This is the hardware profile assumed by the Gemma Flax weights and the competition
starter notebooks.

### Full Training Configuration

```yaml
# training/configs/submission_gemma_flax.yaml
model:
  name: "google/gemma-2b"
  revision: "flax"
  max_length: 512

training:
  num_steps: 100  # Adjust as needed
  per_device_batch_size: 4
  optimizer: "adamw"
```

### Running Full Training

```bash
python training/run_train_jax.py \
    --config training/configs/submission_gemma_flax.yaml \
    --output ./output/full_run \
    --dataset dev-reasoning-v2 \
    --device auto
```

**Note:** On Kaggle, set **Accelerator: TPU v3-8** in notebook settings.

---

## Guardrails

### Runtime Warning

If you attempt to train a large model (Gemma, Llama, etc.) on GPU,
the training script will print a warning:

```
================================================================================
⚠️  WARNING: Large model detected on GPU!
    Model: google/gemma-2b
    This model typically requires >16GB VRAM which exceeds consumer GPUs.
================================================================================
```

The training will proceed, but OOM is likely.

### Transformers Version Pin

We pin `transformers>=4.40,<5` because:
- Transformers v5 removes TF/JAX/Flax support entirely
- Our Flax training code depends on `FlaxAutoModelForCausalLM`

This pin is enforced in:
- `backend/pyproject.toml`
- `notebooks/kaggle_submission.ipynb` install cell

---

## Decision Record (ADR)

**Context:** During M36 development, we discovered that Gemma 2B Flax consistently
OOMs on Kaggle T4 GPUs despite aggressive memory optimizations.

**Decision:** Split training into two modes:
1. GPU Smoke — tiny model for pipeline validation
2. TPU Full — Gemma models for actual training

**Consequences:**
- CI and local dev can validate the pipeline quickly on any GPU
- Competition training is explicitly routed to TPU
- No ambiguity about what hardware is required for each use case

**Status:** FINAL (M36)

---

## Quick Reference

| Task                      | Command Flag                          |
|---------------------------|---------------------------------------|
| Run smoke test            | `--smoke_config ... --smoke_steps 2`  |
| Run full training         | (omit smoke flags)                    |
| Force GPU for large model | `--force_gpu_large_model` (future)    |
| Check config              | `--config path/to/config.yaml`        |
