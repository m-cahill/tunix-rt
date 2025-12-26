# Tunix RT - Final Submission Checklist

**Competition:** Google Tunix Hack - Train a model to show its work  
**Deadline:** January 12, 2026

This checklist ensures a complete, reproducible submission.

---

## 1. Environment Setup

- [ ] Python 3.11+ installed
- [ ] Dependencies installed: `pip install -e ".[dev,training]"`
- [ ] JAX/Flax verified: `python -c "import jax; print(jax.devices())"`
- [ ] GPU/TPU access confirmed (if using Kaggle runtime)

## 2. Dataset Selection

| Dataset | Traces | Use Case |
|---------|--------|----------|
| `dev-reasoning-v1` | 200 | Smoke testing, local development |
| `golden-v2` | 100 | Calibration, final submission |

**Provenance notes:**
- All datasets are deterministically seeded (`seed=42`)
- Manifests include build metadata (`backend/datasets/{key}/manifest.json`)
- Dataset composition is documented in manifest `stats` field

**Verification:**
```bash
# List available datasets
ls backend/datasets/
# Verify dataset contents
head -3 backend/datasets/golden-v2/dataset.jsonl
```

## 3. Training Configuration

**Recommended config for final run:**
```yaml
# training/configs/train_golden_v2.yaml or custom
model:
  name: google/gemma-2-2b  # or gemma-2-2b-it for instruction-tuned
  max_length: 512

training:
  num_steps: 100-1000  # Adjust based on available time
  learning_rate: 1.0e-5
  per_device_batch_size: 4
  seed: 42
```

**Key parameters:**
- `--smoke_steps N`: Limit training for quick validation
- `--device cpu|gpu|auto`: Device selection
- `--save_every_steps N`: Checkpoint frequency

## 4. Training Execution

```bash
# Final training run
python training/train_jax.py \
  --config training/configs/train_golden_v2.yaml \
  --output ./output/final_submission \
  --dataset golden-v2 \
  --device auto \
  --save_every_steps 50
```

**Expected output directory structure:**
```
output/final_submission/
├── checkpoint_50/
├── checkpoint_100/
├── checkpoint_final/
├── metrics.jsonl
└── run_manifest.json
```

## 5. Evaluation & Scoring

```bash
# Generate predictions
python training/eval_generate.py \
  --checkpoint ./output/final_submission \
  --eval_set training/evalsets/eval_v1.jsonl \
  --output ./output/final_submission/predictions.jsonl

# Score predictions
python training/eval_report.py \
  --predictions ./output/final_submission/predictions.jsonl \
  --eval_set training/evalsets/eval_v1.jsonl
```

**Primary metric:** `answer_correctness` (0-100, higher is better)

## 6. Artifacts to Export

**Required artifacts for submission:**

| Artifact | Path | Description |
|----------|------|-------------|
| Model checkpoint | `checkpoint_final/` | Orbax-formatted weights |
| Training metrics | `metrics.jsonl` | Loss per step |
| Predictions | `predictions.jsonl` | Eval set predictions |
| Run manifest | `run_manifest.json` | Config + provenance |

**Recommended bundle naming:**
```
tunix_submission_{run_id}_{dataset}_{timestamp}/
├── checkpoint_final/
├── metrics.jsonl
├── predictions.jsonl
├── predictions_meta.json
├── run_manifest.json
└── evaluation_report.json
```

## 7. Video Requirements

Per competition rules ([Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon)):

- [ ] **Length:** 3 minutes or less
- [ ] **Platform:** Published to YouTube (public or unlisted)
- [ ] **Submission:** Attached to Kaggle Media Gallery

**Suggested content outline:**
1. Problem statement (15 sec)
2. Approach overview (45 sec)
3. Training demonstration (60 sec)
4. Results and evaluation (45 sec)
5. Key takeaways (15 sec)

## 8. Final Sanity Checks

### Code Quality
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy tunix_rt_backend` passes
- [ ] All tests pass: `uv run pytest`

### Reproducibility
- [ ] Training can be reproduced from documented commands
- [ ] Dataset seed is fixed and documented
- [ ] Model checkpoint loads successfully

### Submission Artifacts
- [ ] All required files present in bundle
- [ ] Predictions file is valid JSONL
- [ ] Evaluation metrics recorded

### Kaggle-Specific
- [ ] Notebook runs end-to-end in Kaggle environment
- [ ] Video uploaded and linked
- [ ] Submission deadline confirmed

---

## Quick Reference

**Dry-run command (verified M30):**
```bash
python training/train_jax.py \
  --config training/configs/sft_tiny.yaml \
  --output ./output/dry_run \
  --dataset dev-reasoning-v1 \
  --device cpu \
  --smoke_steps 2
```

**Full training command:**
```bash
python training/train_jax.py \
  --config training/configs/train_golden_v2.yaml \
  --output ./output/final_submission \
  --dataset golden-v2 \
  --device auto
```

---

## Related Documentation

- [Kaggle Submission Guide](kaggle_submission.md)
- [Training End-to-End](training_end_to_end.md)
- [Evaluation Semantics](evaluation.md)
- [Model Registry](model_registry.md)
