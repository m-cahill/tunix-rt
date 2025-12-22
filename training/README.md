# Tunix RT - Training Scripts

This directory contains scripts for running training and evaluation loops with Tunix SFT.

## Structure

```
training/
├── README.md                    # This file
├── configs/                     # Training configurations
│   └── sft_tiny.yaml           # Minimal SFT config for testing
├── evalsets/                    # Static evaluation sets
│   └── eval_v1.jsonl           # Default eval set
├── train_sft_tunix.py          # Main Tunix SFT training script
├── eval_generate.py            # Generate eval outputs (pre/post training)
└── eval_report.py              # Create delta report from eval results
```

## Prerequisites

### Required

- Python 3.11+
- tunix-rt backend installed: `pip install -e backend/`

### Optional (for actual training)

- **Tunix** - Google's SFT/GRPO library:
  ```bash
  pip install git+https://github.com/google-deepmind/tunix.git@<COMMIT_SHA>
  ```

- **JAX** - For running training:
  ```bash
  pip install -e backend[training]
  ```

## Quick Start

### 1. Prepare a Dataset

```bash
# Generate UNGAR traces (if you have UNGAR installed)
curl -X POST http://localhost:8000/api/ungar/high-card-duel/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 100, "seed": 42, "persist": true}'

# Build a dataset
curl -X POST http://localhost:8000/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "hcd_baseline",
    "dataset_version": "v1",
    "filters": {"source": "ungar", "game": "high_card_duel"},
    "limit": 100,
    "selection_strategy": "latest"
  }'

# Export for training
curl "http://localhost:8000/api/datasets/hcd_baseline-v1/export.jsonl?format=tunix_sft" \
  > training_data.jsonl
```

### 2. Run Training (Optional - requires Tunix)

```bash
python training/train_sft_tunix.py \
  --config training/configs/sft_tiny.yaml \
  --data training_data.jsonl \
  --output artifacts/training_runs/my_run
```

### 3. Evaluate Model

```bash
# Generate outputs with base model
python training/eval_generate.py \
  --model base \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output artifacts/training_runs/my_run/eval_before.jsonl

# Generate outputs with trained model
python training/eval_generate.py \
  --model artifacts/training_runs/my_run/checkpoint-final \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output artifacts/training_runs/my_run/eval_after.jsonl

# Create comparison report
python training/eval_report.py \
  --before artifacts/training_runs/my_run/eval_before.jsonl \
  --after artifacts/training_runs/my_run/eval_after.jsonl \
  --output artifacts/training_runs/my_run/delta_report.md
```

## Determinism & Reproducibility

All training scripts enforce reproducibility through:

- **Run Manifests** - Every training run creates `run_manifest.json` with:
  - Dataset used (key, build_id)
  - Model configuration
  - Random seed
  - Git commit SHA
  - Timestamp

- **Seeds** - All random operations use explicit seeds
- **Dataset Manifests** - Datasets are versioned and immutable

## File Formats

### Training Data (JSONL)

Format: `tunix_sft` (Gemma chat template)

```jsonl
{"id": "...", "prompts": "<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>", "final_answer": "...", "metadata": {...}}
```

### Eval Set (JSONL)

Static evaluation prompts:

```jsonl
{"id": "eval-001", "prompt": "What is 2+2?", "expected_answer": "4"}
{"id": "eval-002", "prompt": "Name the capital of France", "expected_answer": "Paris"}
```

### Run Manifest (JSON)

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-21T10:00:00Z",
  "dataset_key": "hcd_baseline-v1",
  "training_config": {
    "model": "google/gemma-2b-it",
    "learning_rate": 1e-5,
    "steps": 50
  },
  "recipe": "trace_sft_v1",
  "seed": 42,
  "git_sha": "ec59ac8",
  "artifacts_path": "artifacts/training_runs/550e8400-..."
}
```

## Script Details

### train_sft_tunix.py

Runs Tunix SFT training with minimal configuration.

**Key Features:**
- Loads dataset JSONL
- Tokenizes using Gemma template
- Runs SFT for N steps (configurable)
- Saves checkpoint + metrics
- Creates run manifest

**Graceful Degradation:**
- Exits with clear error if Tunix not installed
- Provides installation instructions

### eval_generate.py

Generates model outputs for evaluation prompts.

**Supports:**
- Base model (no training)
- Trained checkpoint
- Deterministic generation (seeded)

### eval_report.py

Creates markdown delta report comparing pre/post training.

**Includes:**
- Average score change
- Individual example comparisons
- Statistical summary

## Artifacts

All training outputs go to `artifacts/training_runs/<run_id>/`:

```
artifacts/training_runs/<run_id>/
├── run_manifest.json         # Reproducibility metadata
├── metrics.jsonl             # Training metrics per step
├── checkpoint-final/         # Trained model checkpoint
├── eval_before.jsonl         # Pre-training eval outputs
├── eval_after.jsonl          # Post-training eval outputs
└── delta_report.md           # Comparison report
```

**Note:** `artifacts/` is gitignored - do not commit training outputs.

## Configuration

See `configs/sft_tiny.yaml` for example training configuration.

Key parameters:
- `model`: Base model (e.g., `google/gemma-2b-it`)
- `learning_rate`: SFT learning rate
- `steps`: Number of training steps
- `batch_size`: Training batch size
- `seed`: Random seed

## Troubleshooting

### "Tunix not installed"

```bash
pip install git+https://github.com/google-deepmind/tunix.git@<SHA>
```

See `docs/M09_TRAINING_QUICKSTART.md` for recommended commit SHA.

### "JAX not found"

```bash
pip install -e backend[training]
```

This installs JAX with CPU support. For GPU:

```bash
pip install -U "jax[cuda12]"
```

### "Dataset not found"

Ensure you've built a dataset first using `POST /api/datasets/build`.

## See Also

- `docs/M09_TRAINING_QUICKSTART.md` - Step-by-step tutorial
- `docs/M09_EVAL_LOOP.md` - Evaluation workflow details
- `docs/M09_DATASET_FORMAT.md` - Dataset format specification
- `backend/training/` - Training library code (schemas, renderers)

