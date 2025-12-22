# M09 Training Quickstart

**Milestone:** M09 - Reproducible Training Loop v1  
**Last Updated:** December 21, 2025

---

## Quick Start (5 Minutes)

### 1. Prerequisites

```bash
# Install backend with training extras
cd backend
pip install -e ".[dev,training]"

# Verify JAX installed
python -c "import jax; print(f'JAX {jax.__version__}')"
```

### 2. Generate or Import Traces

**Option A: Use UNGAR** (if installed)
```bash
curl -X POST http://localhost:8000/api/ungar/high-card-duel/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 100, "seed": 42, "persist": true}'
```

**Option B: Import existing traces** (via batch endpoint)
```bash
curl -X POST http://localhost:8000/api/traces/batch \
  -H "Content-Type: application/json" \
  -d @my_traces.json
```

### 3. Build Dataset

```bash
curl -X POST http://localhost:8000/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "quickstart",
    "dataset_version": "v1",
    "filters": {"source": "ungar"},
    "limit": 50,
    "selection_strategy": "latest"
  }'
```

### 4. Export for Training

```bash
curl "http://localhost:8000/api/datasets/quickstart-v1/export.jsonl?format=tunix_sft" \
  > training_data.jsonl
```

### 5. Run Training (Simulation Mode)

```bash
python training/train_sft_tunix.py \
  --config training/configs/sft_tiny.yaml \
  --data training_data.jsonl \
  --output artifacts/training_runs/quickstart_run
```

**Note:** Without Tunix installed, this creates manifests and validates the pipeline.

---

## Installing Tunix (Optional - For Actual Training)

Tunix is not on PyPI yet. Install from GitHub with a pinned commit:

```bash
# Recommended commit (tested with M09):
pip install git+https://github.com/google-deepmind/tunix.git@<COMMIT_SHA>

# See https://github.com/google-deepmind/tunix for latest stable commit
```

---

## Full Training Workflow

### Step 1: Prepare Data

```bash
# Build dataset
curl -X POST http://localhost:8000/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "my_training_set",
    "dataset_version": "v1",
    "filters": {"source": "ungar", "game": "high_card_duel"},
    "limit": 200,
    "selection_strategy": "random",
    "seed": 42
  }' | jq '.'

# Export
curl "http://localhost:8000/api/datasets/my_training_set-v1/export.jsonl?format=tunix_sft" \
  > my_data.jsonl

# Validate
python backend/training/sft_smoke.py my_data.jsonl --samples 32
```

### Step 2: Configure Training

Edit `training/configs/sft_tiny.yaml`:

```yaml
model:
  name: "google/gemma-2b-it"
  max_length: 512

training:
  num_steps: 100  # Increase for real training
  learning_rate: 1.0e-5
  per_device_batch_size: 2
  seed: 42
```

### Step 3: Run Training

```bash
python training/train_sft_tunix.py \
  --config training/configs/sft_tiny.yaml \
  --data my_data.jsonl \
  --output artifacts/training_runs/my_run_$(date +%Y%m%d_%H%M%S)
```

### Step 4: Evaluate

```bash
RUN_DIR="artifacts/training_runs/my_run_20251221_100000"

# Generate pre-training outputs
python training/eval_generate.py \
  --model base \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output $RUN_DIR/eval_before.jsonl

# Generate post-training outputs  
python training/eval_generate.py \
  --model $RUN_DIR/checkpoint-final \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output $RUN_DIR/eval_after.jsonl

# Create comparison report
python training/eval_report.py \
  --before $RUN_DIR/eval_before.jsonl \
  --after $RUN_DIR/eval_after.jsonl \
  --output $RUN_DIR/delta_report.md

# View report
cat $RUN_DIR/delta_report.md
```

---

## Reproducibility

Every training run creates a manifest:

```json
{
  "run_id": "uuid",
  "created_at": "timestamp",
  "dataset_key": "my_training_set-v1",
  "training_config": {...},
  "recipe": "trace_sft_v1",
  "seed": 42,
  "git_sha": "abc123",
  "artifacts_path": "/path/to/artifacts"
}
```

**To reproduce a run:**
1. Use same dataset (by `dataset_key`)
2. Use same config
3. Use same seed
4. Results should be identical (modulo GPU/CPU differences)

---

## Troubleshooting

### "Tunix not installed"
```bash
# Install from GitHub
pip install git+https://github.com/google-deepmind/tunix.git@<SHA>
```

### "JAX not found"
```bash
pip install -e backend[training]
```

### "Dataset empty"
Check filters match your traces:
```bash
curl "http://localhost:8000/api/traces?limit=5" | jq '.data[].payload.meta'
```

### "Format validation failed"
Ensure dataset exported with `format=tunix_sft`:
```bash
head -1 my_data.jsonl | jq '.prompts' | grep '<start_of_turn>'
```

---

## See Also

- `docs/M09_EVAL_LOOP.md` - Evaluation workflow details
- `docs/M09_DATASET_FORMAT.md` - Dataset format specification
- `training/README.md` - Training scripts reference

