# Training Pipeline — Production Integration Guide

**Audience:** DevOps, ML Engineers  
**Purpose:** Run tunix-rt training jobs (local testing → production deployment)  
**Last Updated:** 2025-12-21 (M11)

---

## Overview

The tunix-rt training pipeline supports two modes:

1. **Local Mode** - Smoke testing and development (no Tunix API required)
2. **Production Mode** - Real fine-tuning jobs via Tunix API (M12+)

This guide covers **both modes** with step-by-step instructions.

---

## Prerequisites

### All Modes
- Python 3.11+
- tunix-rt backend installed: `pip install -e backend[training]`
- PostgreSQL database with traces

### Production Mode Only
- Tunix API credentials (`TUNIX_API_KEY`)
- TPU/GPU quota allocated
- Output storage (GCS bucket or local volume)

---

## Mode 1: Local Smoke Testing (Current - M11)

### Purpose
Validate training scripts, config files, and dataset exports **without running actual training**.

### Quick Start

```bash
# 1. Navigate to training directory
cd training

# 2. Run training script with --dry-run flag
python train_sft_tunix.py \
  --config configs/sft_tiny.yaml \
  --dry-run

# Expected output:
# ✅ Config loaded: configs/sft_tiny.yaml
# ✅ Manifest validated: artifacts/datasets/test-v1.json
# ✅ Output dir: artifacts/training_runs/run_abc123
# ✅ Dry-run mode: Exiting without training
# Exit code: 0
```

### What --dry-run Does

| Step | Action | Production | Dry-Run |
|------|--------|------------|---------|
| Load config YAML | ✅ | ✅ | ✅ |
| Validate required fields | ✅ | ✅ | ✅ |
| Compute output paths | ✅ | ✅ | ✅ |
| Validate manifest schema | ✅ | ✅ | ✅ |
| Initialize models/datasets | ✅ | ❌ | **SKIPPED** |
| Submit Tunix job | ✅ | ❌ | **SKIPPED** |
| Run training loop | ✅ | ❌ | **SKIPPED** |

### Configuration Files

Training configs live in `training/configs/`:

```yaml
# training/configs/sft_tiny.yaml
dataset:
  manifest_path: "backend/datasets/test-v1/manifest.json"
  format: "training_example"  # or "tunix_sft"

model:
  base_model: "google/gemma-2b-it"
  # Future: Tunix-specific model config

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-5
  warmup_steps: 100

output:
  run_dir: "artifacts/training_runs"
  checkpoint_every: 500
```

### Smoke Tests (CI)

Backend includes smoke tests via subprocess:

```bash
cd backend
pytest tests/test_training_scripts_smoke.py -v

# Tests:
# - test_training_script_dry_run_exits_zero
# - test_training_script_validates_config
# - test_training_script_missing_config_fails
```

---

## Mode 2: Production Training via Tunix API (M12+)

### Status
**⚠️ NOT YET IMPLEMENTED** - Deferred to M12.

### Planned Workflow

#### Step 1: Build Dataset

```bash
# Build dataset via tunix-rt API
curl -X POST http://localhost:8000/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "high_card_duel_prod",
    "dataset_version": "v1",
    "filters": {"source": "ungar"},
    "limit": 1000,
    "selection_strategy": "latest"
  }'

# Response:
# {
#   "dataset_key": "high_card_duel_prod-v1",
#   "build_id": "abc123...",
#   "trace_count": 1000,
#   "manifest_path": "backend/datasets/high_card_duel_prod-v1/manifest.json"
# }
```

#### Step 2: Export Dataset (Training Format)

```bash
# Export as TrainingExample JSONL
curl "http://localhost:8000/api/datasets/high_card_duel_prod-v1/export.jsonl?format=training_example" \
  > training/data/high_card_duel_prod_v1.jsonl
```

#### Step 3: Configure Tunix Job

```yaml
# training/configs/sft_production.yaml
dataset:
  manifest_path: "backend/datasets/high_card_duel_prod-v1/manifest.json"
  format: "training_example"

model:
  base_model: "google/gemma-7b-it"
  tunix_job_config:
    accelerator: "tpu-v3-8"
    region: "us-central1"

training:
  num_epochs: 5
  batch_size: 16
  learning_rate: 1e-5
  warmup_ratio: 0.1

tunix:
  api_base_url: "https://api.tunix.ai/v1"
  project_id: "tunix-rt-prod"
  experiment_name: "high-card-duel-sft-v1"

output:
  run_dir: "gs://tunix-rt-artifacts/training_runs"  # GCS bucket
  checkpoint_every: 1000
```

#### Step 4: Set Environment Variables

```bash
# Required for production mode
export TUNIX_API_KEY="tunix_sk_..."
export TUNIX_MODE="real"  # Default is "mock"
export DATABASE_URL="postgresql+asyncpg://..."

# Optional
export TUNIX_PROJECT_ID="tunix-rt-prod"
export TRAINING_OUTPUT_BUCKET="gs://tunix-rt-artifacts"
```

#### Step 5: Submit Training Job

```bash
cd training
python train_sft_tunix.py --config configs/sft_production.yaml

# Expected output:
# ✅ Config loaded: configs/sft_production.yaml
# ✅ Tunix client initialized (real mode)
# ✅ Dataset loaded: 1000 training examples
# ✅ Job submitted to Tunix: job_abc123xyz
# 🔗 Monitor: https://dashboard.tunix.ai/jobs/job_abc123xyz
# ⏳ Polling for completion...
```

#### Step 6: Monitor Training

```bash
# Watch job status
python training/eval_generate.py --job-id job_abc123xyz --watch

# Tunix dashboard (browser):
# https://dashboard.tunix.ai/jobs/job_abc123xyz
```

#### Step 7: Download Trained Model

```bash
# Auto-download on completion
python training/eval_generate.py \
  --job-id job_abc123xyz \
  --download \
  --output-dir artifacts/models/high_card_duel_v1
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TUNIX_MODE` | No | `mock` | `mock` (local) or `real` (production) |
| `TUNIX_API_KEY` | Production | - | Tunix API authentication token |
| `TUNIX_BASE_URL` | No | `https://api.tunix.ai/v1` | Tunix API endpoint |
| `TUNIX_PROJECT_ID` | Production | - | Tunix project identifier |
| `DATABASE_URL` | Yes | - | PostgreSQL connection for trace loading |
| `TRAINING_OUTPUT_DIR` | No | `artifacts/training_runs` | Local output directory |
| `TRAINING_OUTPUT_BUCKET` | Production | - | GCS bucket for production artifacts |

---

## Secrets Management

### Local Development
```bash
# .env file (git-ignored)
TUNIX_API_KEY=tunix_sk_dev_...
DATABASE_URL=postgresql+asyncpg://localhost/tunix_rt_dev
```

### Production (Docker/K8s)
```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: tunix-rt-training-secrets
type: Opaque
stringData:
  tunix-api-key: "tunix_sk_prod_..."
  database-url: "postgresql+asyncpg://..."
```

### CI/CD
```yaml
# .github/workflows/training-nightly.yml (future)
env:
  TUNIX_API_KEY: ${{ secrets.TUNIX_API_KEY }}
  TUNIX_MODE: real
```

---

## Validation Checklist

Before submitting a production job:

- [ ] Dataset built and exported (`/api/datasets/build`)
- [ ] Dataset manifest contains expected trace count
- [ ] Config YAML passes `--dry-run` validation
- [ ] Tunix API key is valid (`tunix auth test`)
- [ ] TPU/GPU quota is available
- [ ] Output bucket/directory is writable
- [ ] Cost estimate approved ($X per TPU-hour)

---

## Troubleshooting

### Issue: `--dry-run` exits with error

**Symptoms:**
```
FileNotFoundError: Manifest not found: backend/datasets/test-v1/manifest.json
```

**Solution:**
```bash
# Build dataset first
curl -X POST http://localhost:8000/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "test", "dataset_version": "v1", "limit": 10}'
```

---

### Issue: Tunix API authentication fails

**Symptoms:**
```
TunixAPIError: 401 Unauthorized - Invalid API key
```

**Solution:**
1. Verify API key: `echo $TUNIX_API_KEY`
2. Test auth: `curl -H "Authorization: Bearer $TUNIX_API_KEY" https://api.tunix.ai/v1/auth/test`
3. Regenerate key if expired

---

### Issue: TPU quota exceeded

**Symptoms:**
```
TunixAPIError: 429 Too Many Requests - TPU quota exceeded
```

**Solution:**
1. Check quota: https://console.cloud.google.com/iam-admin/quotas
2. Request increase or wait for quota reset
3. Use smaller accelerator: `tpu-v3-8` → `tpu-v2-8`

---

### Issue: Training OOM (Out of Memory)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solution:**
1. Reduce `batch_size` in config (16 → 8 → 4)
2. Use gradient accumulation: `gradient_accumulation_steps: 4`
3. Upgrade to larger accelerator: `tpu-v3-8` → `tpu-v3-32`

---

## Performance Expectations

### Local Dry-Run
- **Duration:** <5 seconds
- **Memory:** <500 MB
- **Output:** Validation messages only

### Production Training (Example: Gemma-7B, 1000 examples)
- **Duration:** ~2-4 hours on TPU-v3-8
- **Cost:** ~$8-12 USD (TPU time + storage)
- **Checkpoints:** Every 1000 steps (~500 MB each)
- **Final Model:** ~14 GB (Gemma-7B weights)

---

## Cost Optimization

1. **Use --dry-run first** to validate config (free)
2. **Start small:** Test with 10 examples before scaling to 1000
3. **Checkpoint frequently:** Resume from failures
4. **Use spot instances:** 60-90% cheaper (if Tunix supports)
5. **Clean old runs:** Delete checkpoints older than 30 days

---

## Next Steps (M12+)

When Tunix integration is complete:

1. Implement `TunixClient` (see ADR-006)
2. Add production training tests (manual/nightly CI)
3. Set up monitoring (Tunix dashboard + custom metrics)
4. Document model deployment (serving via Tunix or export)

---

## References

- [ADR-006: Tunix API Abstraction](adr/ADR-006-tunix-api-abstraction.md)
- [M09 Dataset Format Specification](M09_DATASET_FORMAT.md)
- [M09 Training Quickstart](M09_TRAINING_QUICKSTART.md)
- [Tunix API Docs](https://docs.tunix.ai) (future)

---

**Questions?** Open an issue or contact the tunix-rt team.

