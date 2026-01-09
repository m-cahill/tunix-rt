Perfect — that’s the right call. Getting **real GPU acceleration on the RTX 5090** is the single highest-leverage technical unlock right now, and it cleanly deserves its own milestone.

Below is a **Cursor-ready M40 prompt**, written to be unambiguous, execution-oriented, and scoped so you don’t get dragged into frontend work prematurely.

---

# **M40 — Enable Native RTX 5090 GPU Acceleration (PyTorch)**

## 🎯 Milestone Goal

Enable **true GPU-accelerated PyTorch training** on the **RTX 5090 (Blackwell, sm_120)** for the existing training pipeline.

**Success means:**

* PyTorch **detects and uses CUDA** (not CPU fallback)
* Training logs explicitly show `device=cuda`
* GPU memory is allocated (verified via `nvidia-smi`)
* A short training run completes successfully on GPU
* Evidence artifacts are saved for submission review

This milestone is **purely about GPU enablement**, not model quality or frontend polish.

---

## 📌 Context (Do Not Skip)

* M39 validated the **PyTorch training pipeline**, but:

  * PyTorch fell back to **CPU**
  * Cause: RTX 5090 uses **compute capability sm_120**, unsupported by stable PyTorch wheels
* We are now intentionally addressing that gap

---

## 🧠 Constraints & Assumptions

* Hardware: **RTX 5090, 32GB VRAM**
* OS: Local development machine (not Kaggle / not TPU)
* CUDA drivers already installed (confirmed via `nvidia-smi`)
* Python environment exists but **can be replaced if needed**
* Training code is already correct — do **not refactor training logic unless required for CUDA**

---

## 🛠️ Phase 1 — Choose and Install a PyTorch Build That Supports sm_120

### Option A (Preferred): PyTorch Nightly with CUDA 12.x

1. **Create a fresh virtual environment** (do not reuse existing one):

   ```bash
   python -m venv .venv-gpu
   source .venv-gpu/bin/activate  # or Windows equivalent
   ```

2. Install **PyTorch nightly** with CUDA support:

   ```bash
   pip install --pre torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/nightly/cu124
   ```

3. Verify CUDA availability:

   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   print(torch.cuda.get_device_capability(0))
   ```

**Expected result:**

* `cuda.is_available() == True`
* Device capability reports `(12, 0)` or equivalent

---

### Option B (Fallback): Build PyTorch from Source

Only do this **if nightly wheels fail**.

High-level steps:

* Install CUDA Toolkit ≥ 12.8
* Clone PyTorch
* Build with:

  ```bash
  TORCH_CUDA_ARCH_LIST="12.0" python setup.py install
  ```

⚠️ This is slower and higher risk — prefer Option A unless blocked.

---

## 🛠️ Phase 2 — Install Project Dependencies (GPU-Safe)

Inside the GPU venv:

```bash
cd backend
pip install -e ".[training]"
```

Notes:

* Do **not** install TPU / JAX extras
* Do **not** mix CPU-only torch packages
* Resolve any dependency conflicts explicitly (document them)

---

## 🧪 Phase 3 — GPU Verification Smoke Test

Create and run a **minimal GPU sanity script**:

```python
import torch

x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU computation successful")
```

Also confirm:

```bash
nvidia-smi
```

shows **active memory usage** from Python.

---

## 🚀 Phase 4 — Run Training on GPU

Run the existing training entrypoint:

```bash
python training/run_train_torch.py \
  --config training/configs/submission_local.yaml \
  --output ./output/m40_gpu_run \
  --dataset dev-reasoning-v2 \
  --device cuda \
  --max_steps 50
```

### Required log confirmations:

* Explicit log line: `Using device: cuda`
* No warnings about unsupported compute capability
* Training speed noticeably higher than CPU run
* GPU memory usage visible in `nvidia-smi`

---

## 📂 Phase 5 — Evidence Capture

Create:

```
submission_runs/
└── m40_gpu_enablement/
    ├── env_info.txt
    ├── torch_cuda_check.txt
    ├── training_log.txt
    ├── nvidia_smi.txt
    └── README.md
```

Include in `README.md`:

* PyTorch version
* CUDA version
* GPU model
* Compute capability
* Confirmation that training ran on GPU

---

## ✅ Definition of Done (Strict)

M40 is complete **only if**:

* [ ] `torch.cuda.is_available()` returns `True`
* [ ] RTX 5090 is explicitly detected
* [ ] Training runs on `device=cuda`
* [ ] GPU memory is allocated during training
* [ ] Evidence artifacts are saved
* [ ] No silent CPU fallback

---

## 🚫 Explicit Non-Goals (Do Not Do)

* ❌ No frontend changes
* ❌ No dataset changes
* ❌ No hyperparameter tuning
* ❌ No TPU/JAX work
* ❌ No performance benchmarking beyond basic confirmation

---

## 🔜 After M40 (Do Not Implement Yet)

* M41: Frontend polish + demo UX
* M42: Video production + narrative
* M43 (optional): PyTorch → multi-GPU or optimization pass

---

## 🧭 If Blocked

If GPU enablement fails:

* Capture full error logs
* Identify whether failure is:

  * PyTorch wheel limitation
  * CUDA version mismatch
  * Driver incompatibility
* Stop and report — **do not hack around silently**

---

### Final Instruction to Cursor

> Execute M40 exactly as scoped.
> Prioritize correctness and explicit verification over speed.
> Do not advance to frontend or demo work until GPU acceleration is proven.

---

If you want, next I can:

* Pre-decide **nightly vs source build** based on your OS
* Generate a **one-command GPU verification script**
* Help plan **M41 frontend polish** once GPU is locked

Just say the word.
