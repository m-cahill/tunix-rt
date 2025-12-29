# Kaggle Submission Guide

This document describes the single-session execution path for the Tunix RT competition submission.

## Overview

This guide provides a reproducible, single-session workflow for:
1. Building a training dataset
2. Training a model using JAX/Flax
3. Running evaluation
4. Generating leaderboard scores

## Prerequisites

- TPU/GPU runtime (Kaggle provides both)
- Python 3.11+
- JAX/Flax installed
- Dataset files available

## Quick Start

### Option 1: Use the Notebook

Open and run `notebooks/kaggle_submission.ipynb` sequentially. All cells are documented.

### Option 2: Command Line

```bash
# 1. Install dependencies
pip install -e ".[dev,tunix]"

# 2. Build dataset (recommended: dev-reasoning-v2 with 550 traces)
python backend/tools/seed_dev_reasoning_v2.py

# 3. Train model (bounded time)
# Note: Use Gemma 2 2B (Flax-compatible). Gemma 3 1B is NOT supported by Flax.
python training/train_jax.py \
  --config training/configs/submission_gemma2_2b.yaml \
  --output ./output/kaggle_run \
  --dataset dev-reasoning-v2 \
  --device auto

# 4. Generate predictions
python training/eval_generate.py \
  --checkpoint ./output/kaggle_run \
  --eval_set training/evalsets/eval_v2.jsonl \
  --output ./output/predictions.jsonl

# 5. Score predictions
python training/eval_report.py \
  --predictions ./output/predictions.jsonl \
  --eval_set training/evalsets/eval_v2.jsonl
```

## Single-Session Constraints

Kaggle TPU sessions have time and weekly caps:
- ~9 hours per session
- ~20 hours per week

Our training is optimized for completion within these constraints:
- `--max_steps` limits training iterations
- `--time_budget_minutes` (optional) adds time-based cutoff
- Checkpointing every 50 steps for resume capability

## Reproducibility

All runs use deterministic seeds:
- Dataset building: `seed=42`
- Training: `--seed 42`
- Evaluation: Fixed eval set

To reproduce results exactly:
1. Use the same dataset version
2. Use the same model checkpoint
3. Use the same hyperparameters

## Artifacts

After training, the following artifacts are generated:
- `checkpoint_*`: Model weights (Orbax format)
- `metrics.jsonl`: Training metrics (loss, step, time)
- `predictions.jsonl`: Evaluation predictions
- `predictions_meta.json`: Prediction metadata

## Evaluation Metrics

The primary metric is `answer_correctness`:
- Range: 0.0 to 100.0
- Definition: Mean of normalized exact-match scores
- Higher is better

See `docs/evaluation.md` for full scoring semantics.

## Troubleshooting

### Out of Memory

Reduce batch size or model size:
```bash
python training/train_jax.py --batch_size 2 --model_name google/gemma-2-2b
```

### Timeout

Use fewer steps:
```bash
python training/train_jax.py --max_steps 50
```

### Dataset Empty

Verify dataset was built:
```bash
ls -l backend/datasets/golden-v2/
```

## Rehearsal Run Verified (M33)

The following rehearsal run was verified on 2025-12-26:

```bash
# Local CPU smoke run with dev-reasoning-v2 dataset (550 traces)
python training/train_jax.py \
  --config training/configs/sft_tiny.yaml \
  --output ./output/m33_rehearsal \
  --dataset dev-reasoning-v2 \
  --device cpu \
  --smoke_steps 2
```

**Expected output:**
```
🚀 Starting SFT Training (JAX/Flax)...
   Device request: CPU
   Model: distilgpt2
   ...
   Tokenizing dataset...
   Training...
   🛑 Smoke steps limit reached (2). Stopping.
```

**Available datasets:**
- `dev-reasoning-v2`: 550 traces (70/20/10 composition) - **recommended for training**
- `golden-v2`: 100 traces - for quick sanity/calibration
- `dev-reasoning-v1`: 200 traces - legacy smoke testing

## References

- Training guide: `docs/training_end_to_end.md`
- Evaluation semantics: `docs/evaluation.md`
- JAX/Flax benchmarks: `training/bench_jax.py`
