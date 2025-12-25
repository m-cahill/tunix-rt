# M27 Clarifying Questions

1.  **Dataset Generation (`golden-v2`)**: The plan refers to `golden-v2` (100 traces). Currently, `backend/tools/seed_golden_dataset.py` generates `golden-v1`. Should I update this script (or create a new one) to generate `golden-v2` with the specified parameters?
2.  **Training Script CLI**: `training/train_jax.py` currently takes a file path via `--data`. The plan uses `--dataset golden-v2`. Should I update `train_jax.py` to resolve dataset names to paths (e.g., `datasets/{name}/dataset.jsonl`), or should I stick to passing the direct file path?
3.  **Evaluation Implementation**: `training/eval_generate.py` currently contains mock generation logic. Should I implement actual model loading and generation (similar to `train_jax.py`) to run inference against the trained checkpoints?
4.  **Evaluation Trigger**: Phase 2 mentions triggering evaluation "At end of training". Should this be done:
    *   Directly inside `train_jax.py` (calling the eval function)?
    *   As a separate step in the backend worker (after the training process exits)?
    *   Via a shell script wrapper?
    *   (Given Phase 3 "Backend Wiring", doing it in the worker seems most robust, but doing it in `train_jax.py` is simpler for "offline" runs. Which is preferred?)
5.  **Frontend Location**: The plan mentions extending the "Run Detail page". The frontend structure doesn't have a `pages` directory. Is `src/components/Tuning.tsx` the main view for runs, or should I create a dedicated "Run Detail" component/view?
