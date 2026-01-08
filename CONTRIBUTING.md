# Contributing to Tunix RT

## Development Setup

### Backend (Python)

We use `uv` for package management.

1. **Install uv**:
   ```bash
   pip install uv
   ```

2. **Sync Dependencies**:
   ```bash
   cd backend
   uv sync --all-extras
   ```
   To install specific groups (e.g. minimal dev):
   ```bash
   uv sync
   ```
   To install training dependencies (JAX/Torch):
   ```bash
   uv sync --extra training
   ```

3. **Database**:
   Run Postgres (e.g. via Docker):
   ```bash
   docker compose up -d postgres
   ```
   Apply migrations:
   ```bash
   cd backend
   uv run alembic upgrade head
   ```

4. **Run Server**:
   ```bash
   uv run uvicorn tunix_rt_backend.app:app --reload
   ```

### Frontend (React/Vite)

1. **Install**:
   ```bash
   cd frontend
   npm install
   ```

2. **Run Dev Server**:
   ```bash
   npm run dev
   ```

## Code Quality

### Python (Ruff + Mypy)

Run linting and formatting before committing:

```bash
cd backend
# 1. Lint and fix imports
uv run ruff check --fix .
# 2. Format code
uv run ruff format .
# 3. Type check
uv run mypy .
```

### Testing

Run tests with `pytest`:

```bash
cd backend
# Unit tests
uv run pytest -m unit
# Integration tests (requires DB)
uv run pytest -m integration
# Full suite
uv run pytest
```

## E2E Testing

Playwright tests are in `e2e/`.

```bash
cd e2e
npm install
npm test
```

## Training Workflow

Training dependencies are optional. To run training scripts locally:

1. Install extras: `uv sync --extra training`
2. Run benchmark: `uv run python training/bench_jax.py`
3. Run training: `uv run python training/train_jax.py --config ...` or via API.

## GPU Development (RTX 5090 / Blackwell)

For local GPU training on RTX 5090 (sm_120 Blackwell architecture), a separate environment with PyTorch nightly is required.

### Setup GPU Environment

```powershell
# From project root
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1

# Install PyTorch nightly with CUDA 12.8 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install training dependencies
pip install transformers accelerate pyyaml
```

### Verify GPU Access

```powershell
.\.venv-gpu\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True, Device: NVIDIA GeForce RTX 5090
```

### Run GPU Training

```powershell
.\.venv-gpu\Scripts\python.exe training_pt/train.py `
  --config training/configs/m40_gpu_smoke.yaml `
  --output output/gpu_test `
  --dataset dev-reasoning-v2 `
  --device cuda
```

### Notes

- **Why separate venv?** PyTorch nightly cu128 is required for sm_120 support; stable PyTorch wheels only support up to sm_90.
- **Version:** torch 2.11.0.dev+cu128 (as of M40, January 2026)
- **Driver Requirement:** NVIDIA driver 576.xx or later with CUDA 12.8+ support
- **Fragility Warning:** Nightly wheels may break without notice. If issues arise:
  1. Check [PyTorch nightly status](https://download.pytorch.org/whl/nightly/cu128)
  2. Try a specific date: `pip install --pre torch==2.11.0.dev20260102+cu128 ...`
  3. Evidence snapshot: `submission_runs/m42_v1/pip_freeze_training_pt_gpu.txt`

## Database Migrations

When modifying SQLAlchemy models (`backend/tunix_rt_backend/db/models/`):

1. Generate migration:
   ```bash
   uv run alembic revision --autogenerate -m "description"
   ```
2. Review the generated file in `backend/alembic/versions/`.
3. Apply:
   ```bash
   uv run alembic upgrade head
   ```
