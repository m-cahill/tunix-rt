# M36 Kaggle Execution Runbook

This document provides step-by-step instructions for executing the Tunix RT training pipeline on Kaggle and capturing evidence files.

**Milestone:** M36 — Real Kaggle Execution + Evidence Lock v2  
**Version:** m36_v4  
**Eval Set:** eval_v2.jsonl (100 items)

---

## Prerequisites

1. **Kaggle Account** with verified phone number (required for GPU/TPU)
2. **Hugging Face Token** with access to Gemma models
3. **Internet Access** enabled in Kaggle notebook (the notebook clones the repository automatically)

---

## Step 1: Set Up Kaggle Notebook

### 1.1 Create New Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Select **GPU** or **TPU** as the accelerator (Settings → Accelerator)
4. Enable **Internet** access (Settings → Internet → On) — **Required for cloning!**

### 1.2 Upload Notebook

Upload `notebooks/kaggle_submission.ipynb` to the notebook:

- Click **"File" → "Import Notebook"**
- Upload from your local copy or link to GitHub

**Note:** The notebook will clone the tunix-rt repository automatically in Cell 2. You don't need to upload other files.

### 1.3 Add Secrets

Add your Hugging Face token:

1. Go to **Settings → Secrets**
2. Add a secret named `HF_TOKEN` with your Hugging Face token
3. In the notebook, add this cell before training:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
```

---

## Step 2: Configuration

The notebook defaults are configured for M36:

| Parameter | Default Value | Notes |
|-----------|---------------|-------|
| `MODEL_NAME` | `google/gemma-3-1b-it` | Competition-allowed model |
| `DATASET` | `dev-reasoning-v2` | 550 traces |
| `EVAL_SET` | `training/evalsets/eval_v2.jsonl` | 100 items (M36) |
| `MAX_STEPS` | `100` | Increase for full training |
| `SEED` | `42` | Deterministic |

Modify these in Cell 6 (Configuration) if needed.

---

## Step 3: Execute Cells

Run cells in order:

| Cell | Section | Purpose | Expected Time |
|------|---------|---------|---------------|
| 2 | 0. Clone Repository | Clone tunix-rt repo | ~30 sec |
| 4 | 1. Setup | Install dependencies | ~1 min |
| 6 | 2. Configuration | Set parameters | instant |
| 8 | 3. Build Dataset | Seed dataset | ~30 sec |
| 10 | 4a. Smoke Run | **Validate pipeline** | ~5 min |
| 12 | 4b. Full Training | **Full training** | 1-2 hours |
| 14 | 5. Generate Predictions | Create predictions | ~10 min |
| 16 | 6. Evaluate & Score | Score predictions | ~2 min |
| 18 | 7. Submission Summary | Display results | instant |

**Important:** 
- **Cell 2 must run first** to clone the repository (all paths depend on this!)
- Run cell 10 (smoke run) before cell 12 to validate the pipeline works

---

## Step 4: Capture Evidence

After cell 18 (Submission Summary) completes, copy the **RESULT SUMMARY** block from the output:

```
============================================================
         RESULT SUMMARY (copy to evidence files)
============================================================
model_id: google/gemma-3-1b-it
dataset: dev-reasoning-v2
eval_set: training/evalsets/eval_v2.jsonl
primary_score: 0.XXXX
final_loss: 0.XXXX
n_items: 100
n_scored: XX
============================================================
```

### 4.1 Update run_manifest.json

Edit `submission_runs/m36_v1/run_manifest.json`:

```json
{
  "run_version": "m36_v1",
  "model_id": "google/gemma-3-1b-it",
  "dataset": "dev-reasoning-v2",
  "eval_set": "training/evalsets/eval_v2.jsonl",
  "config_path": "training/configs/m34_best.yaml",
  "command": "python training/train_jax.py --dataset dev-reasoning-v2 ...",
  "commit_sha": "<your-commit-sha>",
  "timestamp": "<ISO timestamp from run>",
  "tuning_job_id": null,
  "trial_id": null,
  "kaggle_notebook_url": "<paste Kaggle notebook URL>",
  "kaggle_notebook_version": "<version number from Version History>",
  "kaggle_run_id": null,
  "notes": "M36 Evidence Lock v2 - Real Kaggle GPU run"
}
```

**To find Kaggle notebook URL and version:**
1. Click **"File" → "Save Version"** to save the notebook
2. Go to **"Version History"** (top right)
3. Copy the version URL (e.g., `https://www.kaggle.com/code/username/notebook/versions/1`)
4. Note the version number (e.g., `1`)

### 4.2 Update eval_summary.json

Edit `submission_runs/m36_v1/eval_summary.json`:

```json
{
  "run_version": "m36_v1",
  "eval_set": "training/evalsets/eval_v2.jsonl",
  "metrics": {
    "answer_correctness": <from RESULT SUMMARY>,
    "final_loss": <from RESULT SUMMARY>
  },
  "primary_score": <from RESULT SUMMARY>,
  "scorecard": {
    "n_items": 100,
    "n_scored": <from RESULT SUMMARY>,
    "n_skipped": <100 - n_scored>,
    "stddev": null,
    "section_scores": {
      "core": <if displayed>,
      "trace_sensitive": <if displayed>,
      "edge_case": <if displayed>
    },
    "category_scores": {},
    "difficulty_scores": {}
  },
  "predictions_path": "./output/kaggle_run/predictions.jsonl",
  "evaluated_at": "<ISO timestamp>",
  "notes": "M36 evidence - Real Kaggle GPU run with eval_v2.jsonl"
}
```

### 4.3 Update kaggle_output_log.txt

Copy the full cell output from Kaggle (cells 10-18) to `submission_runs/m36_v1/kaggle_output_log.txt`.

Include at minimum:
- Setup output (JAX version, devices)
- Training loss samples (every 10 steps or so)
- Final RESULT SUMMARY block
- Any errors or warnings

---

## Step 5: Package Submission

After updating evidence files:

```bash
python backend/tools/package_submission.py --run-dir submission_runs/m36_v1
```

This creates: `submission/tunix_rt_m36_<date>_<sha>.zip`

---

## Step 6: Verify

1. **Check evidence files are complete:**
   ```bash
   cd backend && uv run pytest tests/test_evidence_files.py -v -k "M36"
   ```

2. **Check primary_score is non-null:**
   ```bash
   cat submission_runs/m36_v1/eval_summary.json | grep primary_score
   # Should NOT show: "primary_score": null
   ```

3. **Commit evidence:**
   ```bash
   git add submission_runs/m36_v1/
   git commit -m "feat(m36): Add real Kaggle run evidence"
   ```

---

## Troubleshooting

### "Out of memory" on Kaggle

- Reduce `MAX_STEPS` to 50
- Use Gemma 3 1B instead of Gemma 2 2B
- Reduce batch size in training config

### "Model not found" error

- Ensure you've accepted the Gemma license on Hugging Face
- Verify `HF_TOKEN` secret is correctly set

### Session timeout

- Kaggle GPU sessions last ~12 hours
- Save checkpoints frequently (`--save_every_steps 25`)
- If interrupted, resume from last checkpoint

---

## Quick Reference

| File | Location | Purpose |
|------|----------|---------|
| Notebook | `notebooks/kaggle_submission.ipynb` | Main execution |
| Evidence manifest | `submission_runs/m36_v1/run_manifest.json` | Run metadata |
| Evidence summary | `submission_runs/m36_v1/eval_summary.json` | Eval results |
| Evidence log | `submission_runs/m36_v1/kaggle_output_log.txt` | Cell outputs |
| Eval set | `training/evalsets/eval_v2.jsonl` | 100 eval items |
| Packaging tool | `backend/tools/package_submission.py` | Create zip |

---

## See Also

- [Submission Checklist](submission_checklist.md)
- [Evaluation Specification](evaluation.md)
- [Training End-to-End](training_end_to_end.md)
