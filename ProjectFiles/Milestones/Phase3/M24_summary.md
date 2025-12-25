# Milestone M24 Summary: Real Inference & Baseline Experiment

**Status**: ✅ Complete
**Date**: 2025-12-24

## 🏆 Achievements

### 1. Real Inference (`generate_predictions`)
*   Replaced the M23 placeholder with **real model inference** using `transformers` and `distilgpt2` (CPU-compatible).
*   **Determinism**: Configured for greedy decoding (`do_sample=False`, `num_beams=1`) to ensure reproducible smoke tests.
*   **Architecture**: Implemented synchronous inference logic wrapper running in `asyncio.to_thread` to keep the API responsive.

### 2. Supply Chain Security (`uv`)
*   **Adopted `uv`**: Installed `uv` and generated `backend/uv.lock` to guarantee deterministic builds.
*   **CI Hardening**: Updated `.github/workflows/ci.yml` to use `uv sync --locked` for backend, e2e, and security jobs.
*   **Lockfile**: Committed `backend/uv.lock` (3,000+ lines of pinned dependencies).

### 3. Baseline Experiment Machinery
*   **Training Script**: Updated `training/train_sft_tunix.py` to support a **PyTorch fallback** (`transformers.Trainer`) when JAX/Tunix is missing. This allows running a "Real Tiny Training" loop (LoRA-style or full finetune of small models) without heavy JAX dependencies in the default environment.
*   **Execution**: Updated `tunix_execution.py` to execute the internal python script directly, enabling the full "Train -> Inference -> Eval" loop to run locally or in CI.

### 4. UI Comparison
*   **Metrics Display**: Updated `frontend/src/App.tsx` and `api/client.ts` to fetch and display run metrics (e.g., `answer_correctness`) directly in the Run History table.
*   **Comparison**: Enabling at-a-glance comparison of "Base" vs "Trained" scores.

## 📉 Artifacts
*   `backend/uv.lock`: Dependency lockfile.
*   `backend/tests/test_m24_inference.py`: New unit tests for inference logic (mocked).
*   `backend/tunix_rt_backend/services/tunix_execution.py`: Real inference implementation.
*   `training/train_sft_tunix.py`: Hybrid JAX/Torch training script.
*   `backend/alembic/versions/9b0c1d2e3f4g_add_metrics_to_tunix_runs.py`: Schema migration for metrics.

## ⚠️ Known Issues
*   **Coverage Dip:** Coverage dropped slightly (69%) due to new inference lines; `test_generate_predictions_success` showed instability in audit. Immediate fix planned for M25 start.

## ⏭️ Next Steps (M25: Full Training)
*   **Full JAX Pipeline**: Implement the real Tunix/JAX training path integration.
*   **Coverage Restoration**: Fix the inference integration test to restore 70%+ coverage.
*   **GPU Acceleration**: Configure for GPU execution if available.
