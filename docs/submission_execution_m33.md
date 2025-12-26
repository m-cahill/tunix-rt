# Submission Execution Runbook — M33

**Version:** m33_v1  
**Purpose:** Execute Kaggle rehearsal run with evidence capture and packaging

---

## Overview

M33 builds on M32's execution runbook with:
1. **Evidence capture** — `submission_runs/m33_v1/` folder with structured JSON
2. **`--run-dir` packaging** — Bundle evidence files in submission archive
3. **dev-reasoning-v2** — 550-trace dataset as default

---

## Quick Start (Local Rehearsal)

```bash
# 1. Run local smoke rehearsal
python training/train_jax.py \
  --config training/configs/sft_tiny.yaml \
  --output ./output/m33_rehearsal \
  --dataset dev-reasoning-v2 \
  --device cpu \
  --smoke_steps 2

# 2. Package with evidence
python backend/tools/package_submission.py --run-dir submission_runs/m33_v1
```

---

## Evidence Files

All evidence is stored in `submission_runs/m33_v1/`:

| File | Purpose | Required Fields |
|------|---------|-----------------|
| `run_manifest.json` | Run configuration | run_version, model_id, dataset, config_path, command, commit_sha, timestamp |
| `eval_summary.json` | Evaluation results | run_version, eval_set, metrics, primary_score, evaluated_at |
| `kaggle_output_log.txt` | Console output | Non-empty text |

---

## Packaging with Evidence

The `--run-dir` flag bundles evidence files:

```bash
python backend/tools/package_submission.py --run-dir submission_runs/m33_v1
```

**Output:** `submission/tunix_rt_m33_<date>_<sha>.zip`

**Contents include:**
- Notebooks and scripts
- Training configs
- Dataset manifests
- Evidence files from `submission_runs/m33_v1/`

---

## Kaggle Execution Steps

### Step 1: Upload Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Import `notebooks/kaggle_submission.ipynb`
3. Set accelerator: **GPU T4 x2** (or TPU)
4. Enable Internet access

### Step 2: Run Smoke Test
1. Execute cells 1-3 (Setup, Configuration, Build Dataset)
2. Execute Cell 4a (Smoke Run)
3. Verify: "✅ Smoke run completed successfully!"

### Step 3: Run Full Training
1. Execute Cell 4b (Full Training Run)
2. Wait for completion (~10-30 min on GPU)
3. Note final loss value

### Step 4: Evaluate
1. Execute Cell 5 (Generate Predictions)
2. Execute Cell 6 (Evaluate & Score)
3. Execute Cell 7 (Submission Summary)

### Step 5: Capture Evidence

After Kaggle execution, update evidence files:

```json
// submission_runs/m33_v1/run_manifest.json
{
  "run_version": "m33_v1",
  "model_id": "google/gemma-3-1b-it",
  "dataset": "dev-reasoning-v2",
  "config_path": "training/configs/submission_gemma3_1b.yaml",
  "command": "<actual command from Cell 4b>",
  "commit_sha": "<current commit>",
  "timestamp": "<execution time ISO 8601>",
  "environment": {
    "platform": "kaggle",
    "accelerator": "<T4 x2 or TPU>",
    "python_version": "3.10"
  }
}
```

```json
// submission_runs/m33_v1/eval_summary.json
{
  "run_version": "m33_v1",
  "eval_set": "training/evalsets/eval_v1.jsonl",
  "metrics": {
    "final_loss": <from metrics.jsonl>,
    "answer_correctness": <from Cell 6>
  },
  "primary_score": <answer_correctness value>,
  "evaluated_at": "<time ISO 8601>"
}
```

Copy console output to `kaggle_output_log.txt`.

---

## CI Tests

Evidence files are validated by CI:

```bash
cd backend
uv run pytest tests/test_evidence_files.py -v
```

Tests verify:
- All required fields present
- Valid JSON format
- model_id is competition-allowed model
- Non-empty log file

---

## Video Requirements

Per [Kaggle timeline](https://www.kaggle.com/competitions/google-tunix-hackathon/overview/timeline):

- ⚠️ **MUST be 3 minutes or less**
- Platform: YouTube (public or unlisted)
- Submission: Kaggle Media Gallery

---

## Related Documentation

- [M32 Execution Runbook](submission_execution_m32.md) — Detailed step-by-step
- [Submission Checklist](submission_checklist.md) — Final verification
- [Submission Freeze](submission_freeze.md) — Frozen configuration
- [Kaggle Submission Guide](kaggle_submission.md) — Overview
