# M40 Clarifying Questions

Before proceeding with RTX 5090 GPU enablement, please clarify the following:

---

## 1. Training Script Path Discrepancy

The M40 plan references:
```bash
python training/run_train_torch.py --config training/configs/submission_local.yaml ...
```

However, this file **does not exist**. The actual PyTorch training script from M39 is:
```
training_pt/train.py
```

**Question:** Should I:
- **(A)** Create a new `training/run_train_torch.py` wrapper (analogous to `training/run_train_jax.py`)?
- **(B)** Use the existing `training_pt/train.py` directly?
- **(C)** Rename/move `training_pt/train.py` to `training/run_train_torch.py`?

---

## 2. Virtual Environment Strategy

The M40 plan suggests creating a **fresh** `.venv-gpu` environment. The project currently uses `.venv`.

**Question:** Should I:
- **(A)** Replace the existing `.venv` entirely with a GPU-capable one?
- **(B)** Create a separate `.venv-gpu` for GPU work (keeping `.venv` as fallback)?
- **(C)** Update the existing `.venv` in-place with nightly PyTorch?

---

## 3. CUDA Driver Version Confirmation

The M39 error message states RTX 5090 (sm_120) requires **CUDA 12.8 or 13.0**.

**Question:** Can you confirm your NVIDIA driver version by running:
```powershell
nvidia-smi
```

Specifically, I need to know:
- The **Driver Version** (e.g., 565.xx or higher)
- The **CUDA Version** shown in nvidia-smi (this indicates driver-supported CUDA level)

This determines whether we can use `cu128` nightly wheels or need a driver update first.

---

## 4. PyTorch Nightly Index URL

The M40 plan suggests `cu124` wheels, but the error message indicates `cu128` is needed.

**Question:** Should I:
- **(A)** Try `--index-url https://download.pytorch.org/whl/nightly/cu128` (if available)?
- **(B)** Start with `cu124` and fall back if needed?
- **(C)** Build PyTorch from source (last resort)?

---

## 5. Config File

The plan references `training/configs/submission_local.yaml` which doesn't exist.

Available GPU configs are:
- `training/configs/submission_local_gpu.yaml` (Gemma 2B, CUDA, bfloat16)
- `training/configs/submission_local_gpu_gpt2.yaml` (if it exists)

**Question:** Which model should we use for the GPU validation run?
- **(A)** GPT-2 (small, fast, good for smoke test)
- **(B)** Gemma 2B (production target, but larger)
- **(C)** Create a new `submission_local.yaml` with specific settings?

---

## 6. Evidence Directory

The plan specifies `submission_runs/m40_gpu_enablement/` for evidence.

**Question:** Is this path correct, or should it follow the existing pattern: `submission_runs/m40_v1/`?

---

**Awaiting your responses before proceeding.**
