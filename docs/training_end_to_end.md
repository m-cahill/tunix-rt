# End-to-End Training Guide (M27)

This guide walks through the full loop: Dataset -> Training -> Evaluation -> Results.

## 1. Setup

Ensure you have the backend and training dependencies installed:

```bash
uv pip install -e "backend[training]"
```

## 2. Seed Dataset

We use `golden-v2`, a deterministic synthetic dataset of 100 traces.

```bash
# Set correct DB URL if needed
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
python backend/tools/seed_golden_v2.py
```

This creates the dataset in the database and generates a manifest at `backend/datasets/golden-v2/manifest.json`.

## 3. Training (Offline / Manual)

You can run training manually using the JAX script.

```bash
uv run python training/train_jax.py \
  --config training/configs/train_golden_v2.yaml \
  --dataset golden-v2 \
  --output artifacts/training_runs/golden_v2_run \
  --eval_after_train
```

**What happens:**
1.  Loads `golden-v2` dataset (resolving path automatically).
2.  Trains `distilgpt2` for 100 epochs (~2500 steps).
3.  Saves checkpoints to `artifacts/training_runs/golden_v2_run/checkpoints`.
4.  Saves final model to `artifacts/training_runs/golden_v2_run/final_model`.
5.  **Evaluation**: Runs `eval_generate.py` using the final model against `training/evalsets/eval_v1.jsonl`.
6.  Saves evaluation traces to `artifacts/training_runs/golden_v2_run/eval_results.jsonl`.

## 4. Evaluation

To run evaluation separately:

```bash
uv run python training/eval_generate.py \
  --model artifacts/training_runs/golden_v2_run/final_model \
  --eval-set training/evalsets/eval_v1.jsonl \
  --output artifacts/eval_results_manual.jsonl
```

## 5. Tunix Integration (Backend Worker)

When running via the UI ("Run with Tunix"):
1.  The backend worker executes the training script.
2.  It streams logs to the UI.
3.  Upon completion, it automatically triggers inference (generating `predictions.jsonl`).
4.  The system then runs a Judge (e.g. `AnswerCorrectnessJudge`) to score the run.
5.  Results are visible in the "Run History" or "Run Details" panel.

## Success Criteria

*   **Loss**: Should decrease from ~3.5 to ~0.5 (or lower) on `golden-v2`.
*   **Evaluation**: The model should output coherent answers for the simple math/repetition tasks in `golden-v2` (if they overlap with `eval_v1` or if we inspect generation).
