# Submission Execution Runbook — M32

**Version:** m32_v1  
**Purpose:** Step-by-step guide to execute, document, and package a competition submission

---

## Prerequisites

1. **Local Environment:**
   - Python 3.12+ with `uv` installed
   - Node.js 18+ (for frontend)
   - Git (for commit SHA tracking)

2. **Kaggle Environment:**
   - Kaggle account with GPU access
   - Notebook verified accelerator (T4 GPU minimum)

3. **Files Ready:**
   - `notebooks/kaggle_submission.ipynb` (refactored in M31)
   - `training/configs/submission_gemma3_1b.yaml` (recommended config)
   - `training/evalsets/eval_v1.jsonl` (evaluation dataset)

---

## Step 1: Local Smoke Test (5 min)

Before Kaggle, validate the pipeline locally:

```bash
# From repo root
python training/train_jax.py \
  --config training/configs/sft_tiny.yaml \
  --output ./output/smoke_test_m32 \
  --dataset dev-reasoning-v2 \
  --device cpu \
  --smoke_steps 2
```

**Expected Output:**
- `output/smoke_test_m32/metrics.jsonl` exists
- `output/smoke_test_m32/checkpoints/` has Orbax checkpoint

---

## Step 2: Kaggle Notebook Execution

### 2a. Upload Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook" → "File" → "Import Notebook"
3. Upload `notebooks/kaggle_submission.ipynb`

### 2b. Configure Environment
1. Set Accelerator: **GPU T4 x2** (or P100)
2. Set Internet: **On** (for model download)
3. Set Environment: **Python 3.10**

### 2c. Run Smoke Test (Cell 4a)
1. Execute cells 1-3 (setup)
2. Execute Cell 4a: "Smoke Run (2 steps)"
3. Verify output shows:
   - `🚀 Starting SFT Training (JAX/Flax)...`
   - `🛑 Smoke steps limit reached (2). Stopping.`

### 2d. Run Full Training (Cell 4b)
1. Modify config cell if needed:
   - `model_name`: `google/gemma-3-1b-it` (default)
   - `dataset`: `golden-v2` or `dev-reasoning-v2`
   - `max_steps`: 100 (default)
2. Execute Cell 4b: "Full Training Run"
3. Wait for completion (~10-30 min depending on GPU)

### 2e. Capture Output
1. Copy the full console output from Cell 4b
2. Note the final metrics:
   - Final loss
   - Steps completed
   - Eval score (if `--eval_after_train` was used)

---

## Step 3: Record Evidence

### 3a. Create Run Manifest

Create `submission_runs/m32_v1/run_manifest.json`:

```json
{
  "run_version": "m32_v1",
  "model_id": "google/gemma-3-1b-it",
  "dataset": "golden-v2",
  "config_path": "training/configs/submission_gemma3_1b.yaml",
  "max_steps": 100,
  "commit_sha": "<insert git sha>",
  "timestamp": "<YYYY-MM-DDTHH:MM:SSZ>",
  "environment": {
    "platform": "kaggle",
    "accelerator": "T4 x2",
    "python_version": "3.10"
  },
  "notes": "Full training run for M32 submission"
}
```

### 3b. Create Eval Summary

Create `submission_runs/m32_v1/eval_summary.json`:

```json
{
  "run_version": "m32_v1",
  "eval_set": "training/evalsets/eval_v1.jsonl",
  "metrics": {
    "final_loss": null,
    "eval_score": null,
    "answer_correctness": null
  },
  "predictions_path": null,
  "evaluated_at": "<YYYY-MM-DDTHH:MM:SSZ>"
}
```

*Fill in actual values after training completes.*

### 3c. Save Kaggle Output Log

Save console output to `submission_runs/m32_v1/kaggle_output_log.txt`:
- Truncate if over 50KB
- Remove any sensitive information (API keys, tokens)

---

## Step 4: Export and Submit

### 4a. Publish Notebook
1. In Kaggle, click "Save & Run All"
2. Wait for execution to complete
3. Click "Publish" → Set visibility to "Public"
4. Copy the notebook URL

### 4b. Update Checklist

Edit `docs/submission_checklist.md`:
```markdown
- [x] Kaggle Notebook URL: https://www.kaggle.com/code/...
- [x] YouTube Video URL: https://youtu.be/...
- [x] Submission Timestamp: YYYY-MM-DD HH:MM UTC
```

---

## Step 5: Package Artifacts

```bash
# From repo root
python backend/tools/package_submission.py
```

This creates `submission/tunix_rt_m32_<date>_<sha>.zip` containing:
- Notebooks
- Configs
- Documentation
- Evidence files (if tracked)

---

## Evidence Files Reference

| File | Purpose | Tracked in Git? |
|------|---------|-----------------|
| `run_manifest.json` | Run configuration and metadata | ✅ Yes |
| `eval_summary.json` | Evaluation results | ✅ Yes |
| `kaggle_output_log.txt` | Console output (scrubbed) | ✅ Yes |
| `checkpoints/` | Model weights | ❌ No (gitignored) |
| `predictions.jsonl` | Generated predictions | ❌ No (large) |

---

## Placeholders (Fill After Execution)

| Field | Value |
|-------|-------|
| YouTube URL | `<TBD>` |
| Kaggle Notebook URL | `<TBD>` |
| Submission Timestamp | `<TBD>` |
| Model ID | `google/gemma-3-1b-it` |
| Dataset | `golden-v2` or `dev-reasoning-v2` |
| Max Steps | `100` |
| Final Eval Score | `<TBD>` |

---

## Troubleshooting

### Kaggle GPU Not Available
- Try P100 instead of T4
- Check Kaggle quota limits

### Training Fails with OOM
- Reduce `per_device_batch_size` in config
- Use `--gradient_checkpointing`

### Evaluation Not Running
- Ensure `--eval_after_train` flag is passed
- Check that `eval_v1.jsonl` path is correct

---

*Last Updated: M32 — December 2025*
