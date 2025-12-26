# Tunix RT - Submission Artifacts

This document describes the contents of the submission bundle and how to reproduce the results.

---

## Archive Contents

The submission archive (`tunix_rt_m33_<date>_<sha>.zip`) contains:

### Notebooks

| File | Purpose |
|------|---------|
| `notebooks/kaggle_submission.ipynb` | Main Kaggle notebook (recommended) |
| `notebooks/kaggle_submission.py` | Python script equivalent |

### Documentation

| File | Purpose |
|------|---------|
| `docs/kaggle_submission.md` | Kaggle workflow guide |
| `docs/submission_checklist.md` | Final submission checklist |
| `docs/submission_freeze.md` | Frozen configuration snapshot |
| `docs/submission_artifacts.md` | This file |
| `docs/evaluation.md` | Evaluation semantics |
| `docs/training_end_to_end.md` | Complete training guide |

### Training Configurations

| File | Purpose |
|------|---------|
| `training/configs/submission_gemma3_1b.yaml` | Gemma 3 1B config (recommended) |
| `training/configs/submission_gemma2_2b.yaml` | Gemma 2 2B config (alternative) |
| `training/configs/sft_tiny.yaml` | Smoke test config |

### Evaluation

| File | Purpose |
|------|---------|
| `training/evalsets/eval_v1.jsonl` | 25-item evaluation set |

### Dataset Manifests

| File | Purpose |
|------|---------|
| `datasets/dev-reasoning-v2/manifest.json` | Dev dataset manifest (550 traces, recommended) |
| `datasets/golden-v2/manifest.json` | Golden dataset manifest (100 traces) |
| `datasets/dev-reasoning-v1/manifest.json` | Dev dataset manifest (200 traces) |

### Evidence Files (M33+)

| File | Purpose |
|------|---------|
| `submission_runs/m33_v1/run_manifest.json` | Run configuration and provenance |
| `submission_runs/m33_v1/eval_summary.json` | Evaluation results and primary score |
| `submission_runs/m33_v1/kaggle_output_log.txt` | Console output from rehearsal run |

**Note:** Dataset JSONL files are not included in the archive. They are generated at runtime using the seed scripts.

---

## Why Each Artifact Exists

### Notebooks

- **kaggle_submission.ipynb**: Primary entry point for Kaggle. Contains the complete workflow with smoke/full run modes, proper error handling, and submission summary.
- **kaggle_submission.py**: Command-line equivalent for local testing or CI automation.

### Configurations

- **submission_gemma3_1b.yaml**: Production config targeting Gemma 3 1B-IT (instruction-tuned). Optimized for reasoning tasks with conservative hyperparameters.
- **submission_gemma2_2b.yaml**: Alternative config for Gemma 2 2B. Larger model with adjusted batch size for memory constraints.
- **sft_tiny.yaml**: Minimal config for pipeline validation. Uses distilgpt2 for fast testing.

### Evaluation

- **eval_v1.jsonl**: Curated 25-item evaluation set covering arithmetic, word problems, geometry, and knowledge questions. Used to measure `answer_correctness` metric.

### Dataset Manifests

- **manifest.json**: Contains provenance metadata (build ID, trace count, seed, timestamps). Enables reproducibility verification without including full dataset files.

---

## How to Reproduce the Run

### Prerequisites

1. **Accept Gemma License**
   - Visit https://huggingface.co/google/gemma-3-1b-it
   - Accept the license agreement
   - Set HF_TOKEN environment variable if needed

2. **Install Dependencies**
   ```bash
   pip install jax[cuda12] flax optax orbax-checkpoint transformers datasets
   ```

### Option 1: Kaggle Notebook

1. Upload `notebooks/kaggle_submission.ipynb` to Kaggle
2. Select TPU or GPU runtime
3. Run cells 1-3 (Setup, Configuration, Build Dataset)
4. Run cell 4a (Smoke Run) to validate
5. Run cell 4b (Full Training Run)
6. Run cells 5-7 (Predictions, Evaluation, Summary)

### Option 2: Command Line

```bash
# Build dataset
python backend/tools/seed_golden_v2.py

# Train model
python training/train_jax.py \
  --config training/configs/submission_gemma3_1b.yaml \
  --output ./output/submission_run \
  --dataset golden-v2 \
  --device auto

# Generate predictions
python training/eval_generate.py \
  --checkpoint ./output/submission_run \
  --eval_set training/evalsets/eval_v1.jsonl \
  --output ./output/submission_run/predictions.jsonl

# Score predictions
python training/eval_report.py \
  --predictions ./output/submission_run/predictions.jsonl \
  --eval_set training/evalsets/eval_v1.jsonl
```

### Option 3: Python Script

```bash
python notebooks/kaggle_submission.py \
  --model_name google/gemma-3-1b-it \
  --dataset golden-v2 \
  --max_steps 100 \
  --device auto
```

---

## Expected Outputs

After a successful run, you should have:

| Artifact | Location | Description |
|----------|----------|-------------|
| Checkpoints | `output/*/checkpoint_*` | Orbax-formatted model weights |
| Metrics | `output/*/metrics.jsonl` | Training loss per step |
| Predictions | `output/*/predictions.jsonl` | Model predictions on eval set |
| Metadata | `output/*/predictions_meta.json` | Prediction run metadata |
| Results | `output/*/eval_results.json` | Evaluation scores |

---

## Verification Checklist

- [ ] Archive extracts cleanly
- [ ] Notebook paths are correct for Kaggle working directory
- [ ] All referenced files exist in archive
- [ ] Smoke run completes without errors
- [ ] Full run produces expected artifacts
- [ ] Evaluation score is recorded

---

## Related Documentation

- [Submission Freeze](submission_freeze.md) — Configuration snapshot
- [Submission Checklist](submission_checklist.md) — Final verification
- [Kaggle Submission Guide](kaggle_submission.md) — Detailed workflow
