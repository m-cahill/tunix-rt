# M42 Evidence Index

**Milestone:** M42 — Final Submission Package  
**Date:** 2026-01-08  
**Machine:** Windows 10, RTX 5090 (32GB VRAM)  
**Status:** ✅ Complete

---

## Purpose

This folder contains evidence artifacts proving the submission is functional, tested, and reproducible.

---

## Evidence Folders Required

| Folder | Purpose | Status |
|--------|---------|--------|
| `submission_runs/m40_v1/` | GPU enablement proof (RTX 5090, CUDA 12.8, PyTorch nightly) | ✅ Complete |
| `submission_runs/m41_v1/` | Frontend polish proof (clean test output, 75 tests) | ✅ Complete |
| `submission_runs/m42_v1/` | Final test runs, dependency snapshots, this index | ✅ Complete |

---

## M42 Contents

### Test Run Outputs

| File | Contents | Result |
|------|----------|--------|
| `test_run_outputs/backend_tests.txt` | pytest output with coverage | 384 passed, 11 skipped, 75.79% coverage |
| `test_run_outputs/frontend_tests.txt` | vitest output | 75 passed, 7 test files |

### Dependency Snapshots

| File | Contents |
|------|----------|
| `pip_freeze_backend.txt` | Exact package versions for backend environment |

**Note:** GPU environment (`pip_freeze_training_pt_gpu.txt`) is optional and requires `.venv-gpu` activation.

---

## M40 Contents (GPU Enablement)

| File | Purpose |
|------|---------|
| `env_info.txt` | Python and environment details |
| `torch_version.txt` | PyTorch nightly version (2.11.0.dev+cu128) |
| `cuda_check.txt` | `torch.cuda.is_available()` verification |
| `nvidia_smi.txt` | GPU detection (RTX 5090, Driver 576.88, CUDA 12.9) |
| `training_log.txt` | GPU smoke test output (31.7 samples/sec) |
| `README.md` | M40 milestone summary |

---

## M41 Contents (Frontend Polish)

| File | Purpose |
|------|---------|
| `README.md` | Milestone summary |
| `frontend_tests_clean.txt` | Clean test output (75 tests, no warnings) |

---

## Known Benign Warnings

1. **Backend:** 11 tests skipped (UNGAR/Tunix not installed, SKIP LOCKED requires PostgreSQL)
2. **Frontend:** jsdom navigation error, mock fetch edge case — do not affect test correctness

---

## Reproducibility Notes

1. **Backend dependencies:** Locked via `uv.lock` (source of truth)
2. **Frontend dependencies:** Locked via `package-lock.json`
3. **GPU environment:** Requires PyTorch nightly cu128 (fragile, documented in CONTRIBUTING.md)
4. **Training configs:** YAML files in `training/configs/`
5. **Eval sets:** JSONL files in `training/evalsets/`

---

## Authoritative Documentation

| Topic | Location |
|-------|----------|
| Demo guide | `docs/DEMO.md` |
| Video checklist | `docs/submission/VIDEO_CHECKLIST.md` |
| GPU setup | `CONTRIBUTING.md` (GPU Development section) |
| Submission checklist | `docs/submission_checklist.md` |
| Training modes | `docs/TRAINING_MODES.md` |

---

## Verification Commands

```powershell
# Verify backend tests
cd backend
uv run pytest --cov=tunix_rt_backend -v

# Verify frontend tests
cd frontend
npm run test

# Verify GPU (requires .venv-gpu)
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

