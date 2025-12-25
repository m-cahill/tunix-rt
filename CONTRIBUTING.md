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
