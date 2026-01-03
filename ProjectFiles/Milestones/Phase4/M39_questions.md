# M39 Clarifying Questions

Based on my analysis of your environment and the M39 plan, I have the following questions before proceeding:

---

## 🖥️ Environment Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| Python | ✅ 3.14.2 | Newer than 3.11+ requirement |
| Node.js | ✅ v24.12.0 | Newer than 18+ requirement |
| Docker | ✅ Working | PostgreSQL container healthy |
| PostgreSQL | ✅ Healthy | Running on port 5433 |
| **RTX 5090** | ✅ **32GB VRAM** | Perfect for M39 training |
| CUDA Toolkit (nvcc) | ⚠️ Not in PATH | May be installed but not configured |
| JAX | ❌ Not installed | Needs CUDA-enabled version |
| PyTorch | ❌ Not installed | Potential fallback option |
| pip | ⚠️ Works via `python -m pip` | Not directly in PATH |
| uv | ❌ Not installed | Recommended package manager |

---

## 1. CUDA Toolkit Installation

Your `nvidia-smi` works (GPU detected), but `nvcc` is not in PATH. This suggests either:
- CUDA Toolkit is not installed, or
- CUDA Toolkit is installed but not added to PATH

**Question:** 
- **Option A:** Should I guide you through CUDA Toolkit installation first?
- **Option B:** Do you already have CUDA installed somewhere (e.g., via NVIDIA driver bundle)?

For JAX GPU training, we need either:
- Full CUDA Toolkit (12.x recommended for RTX 5090/Blackwell)
- OR use `jax[cuda12_pip]` which bundles CUDA libraries

---

## 2. Python 3.14 Compatibility Concern

You have Python **3.14.2** — this is very cutting-edge (likely a preview build). Some packages may have compatibility issues:
- JAX/Flax may not have wheels for 3.14 yet
- NumPy/SciPy compatibility is uncertain

**Question:**
- **Option A:** Proceed with Python 3.14 and troubleshoot if needed
- **Option B:** Should I guide you to install Python 3.11 or 3.12 alongside?

Python 3.12 is the safest bet for JAX/Flax compatibility.

---

## 3. Training Framework — JAX vs PyTorch

The M39 plan mentions PyTorch as an **optional fallback** if JAX GPU is problematic.

Given your fresh environment, we have a choice:

| Option | Pros | Cons |
|--------|------|------|
| **JAX First** | Existing code works | JAX GPU setup more complex on Windows |
| **PyTorch First** | Easier GPU setup, better Windows support | Requires new training script |
| **Both** | Maximum flexibility | More setup work |

**Question:** Which path do you prefer?
- **Option A:** Try JAX GPU first (existing code), fall back to PyTorch if issues
- **Option B:** Go straight to PyTorch (skip JAX GPU debugging)
- **Option C:** Install both now, decide later

---

## 4. Package Manager — pip vs uv

The project uses `uv.lock` for reproducible dependencies, but `uv` is not installed.

**Question:**
- **Option A:** Install `uv` and use `uv sync` (recommended by project)
- **Option B:** Use `pip install` directly (simpler, but may have version drift)

---

## 5. Backend Dependencies — Full Install Scope

Before running tests, I need to install dependencies. The project has optional extras:
- `dev` — Testing/linting tools
- `training` — JAX/Flax/PyTorch for model training
- `tuning` — Ray Tune for hyperparameter sweeps

**Question:** Should I install everything?
```powershell
cd backend
python -m pip install -e ".[dev,training,tuning]"
```

Or minimal for M39 focus:
```powershell
cd backend  
python -m pip install -e ".[dev,training]"
```

---

## 6. M39 Evidence Folder

The plan specifies creating `submission_runs/m39_v1/`. 

**Question:** Should I create this folder structure now, or wait until training runs successfully?

---

## Summary — Recommended Path Forward

Based on your environment, here's my recommended approach:

1. **Fix CUDA/JAX setup first** — Install CUDA Toolkit 12.x or use `jax[cuda12_pip]`
2. **Install backend dependencies** — `pip install -e ".[dev,training]"`
3. **Try JAX GPU training** — If it works, great; if not, pivot to PyTorch
4. **Capture evidence** — Populate `submission_runs/m39_v1/`

Please answer the 6 questions above and I'll finalize the task list and begin execution!

---

## Quick Decision Summary

| # | Question | Options |
|---|----------|---------|
| 1 | CUDA Setup | A) Guide CUDA install / B) Already have it |
| 2 | Python 3.14 | A) Proceed / B) Install 3.11/3.12 |
| 3 | Training Framework | A) JAX first / B) PyTorch first / C) Both |
| 4 | Package Manager | A) uv / B) pip directly |
| 5 | Install Scope | A) Full / B) Minimal |
| 6 | Evidence Folder | A) Create now / B) Wait |
