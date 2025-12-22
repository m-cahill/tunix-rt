# M09 Baseline - Pre-Implementation State

**Date:** December 21, 2025  
**Commit:** `ec59ac8fe5c445dcc2f9db859015f6f58fe98354`  
**Previous Milestone:** M8 Complete (Dataset & Training Bridge v1)

---

## Purpose

This document captures the complete state of tunix-rt **before** M09 implementation begins. It serves as a reference point to measure M09's impact and ensure no regressions.

---

## Test Metrics

### Backend Tests

```
88 passed, 6 skipped in 4.84s
```

**Test Breakdown:**
- `test_health.py`: 2 tests (app health, settings)
- `test_redi_health.py`: 5 tests (mock/real modes, caching)
- `test_traces.py`: 26 tests (CRUD, validation, pagination)
- `test_scoring.py`: 5 tests (baseline scorer, compare endpoint)
- `test_helpers.py`: 3 tests (validation helpers)
- `test_datasets.py`: 13 tests (build, export, manifests)
- `test_renderers.py`: 9 tests (Tunix SFT rendering)
- `test_ungar.py`: 11 tests (5 passed, 6 skipped - requires optional UNGAR)
- `test_ungar_availability.py`: 5 tests (availability checks)
- `test_settings.py`: 9 tests (configuration validation)

**Total Backend Tests:** 88 passing (94 total with skipped)

### Frontend Tests

```
11 tests passing
```

- App health display tests
- Trace upload/fetch tests  
- Comparison UI tests

### E2E Tests

```
7 tests passing (Playwright)
```

- Health check smoke test
- Trace upload and retrieval
- Trace comparison flow
- UNGAR panel visibility

**Total Project Tests:** 106 passing (88 backend + 11 frontend + 7 E2E)

---

## Coverage Metrics

**Backend Coverage (core only):**

```
Coverage: 79%
Statements: 517 total, 97 missed, 420 covered
Gate: ≥70% (PASSING ✅)
```

**Key Module Coverage:**
- `app.py`: 328 statements, 87% coverage
- `scoring.py`: 11 statements, 100% coverage
- `redi_client.py`: 32 statements, 81% coverage
- `training/renderers.py`: 16 statements, 100% coverage
- `helpers/`: ~100% across all modules
- `schemas/`: ~100% across all modules

**Coverage Configuration:** `.coveragerc` omits optional integrations:
- `tunix_rt_backend/integrations/ungar/*`
- Test files and migrations

---

## API Endpoints (12 total)

### Health (2)
1. `GET /api/health` - App health check
2. `GET /api/redi/health` - RediAI integration health

### Traces (4)
3. `POST /api/traces` - Create trace
4. `GET /api/traces/{trace_id}` - Get trace by ID
5. `GET /api/traces` - List traces (paginated)
6. `GET /api/traces/compare` - Compare two traces

### Scoring (1)
7. `POST /api/traces/{trace_id}/score` - Score a trace

### UNGAR (Optional - 3)
8. `GET /api/ungar/status` - Check UNGAR availability
9. `POST /api/ungar/high-card-duel/generate` - Generate game traces
10. `GET /api/ungar/high-card-duel/export.jsonl` - Export UNGAR traces

### Datasets (2)
11. `POST /api/datasets/build` - Build versioned dataset
12. `GET /api/datasets/{dataset_key}/export.jsonl` - Export dataset (formats: trace, tunix_sft)

---

## Database Schema

### Tables (2)

**1. `traces`**
- `id` (UUID, PRIMARY KEY)
- `created_at` (TIMESTAMPTZ, indexed)
- `trace_version` (VARCHAR(64))
- `payload` (JSONB) - Full ReasoningTrace data

**2. `scores`**
- `id` (UUID, PRIMARY KEY)
- `trace_id` (UUID, FK to traces, indexed)
- `criteria` (VARCHAR(64), indexed)
- `score` (FLOAT)
- `details` (JSON)
- `created_at` (TIMESTAMPTZ)

### Migrations (3)

1. `001_create_traces_table.py` - Initial traces table
2. `f8f1393630e4_add_traces_created_at_index.py` - Performance index
3. `f3cc010ca8a6_add_scores_table.py` - Scoring system

---

## Project Structure

### Backend Modules

```
backend/tunix_rt_backend/
├── app.py (328 lines) - Main FastAPI app with 12 endpoints
├── settings.py (64 lines) - Pydantic configuration
├── redi_client.py (70 lines) - Mock/real RediAI client
├── scoring.py (50 lines) - Baseline scorer
├── db/
│   ├── base.py - Async SQLAlchemy setup
│   └── models/
│       ├── trace.py - Trace ORM model
│       └── score.py - Score ORM model
├── schemas/
│   ├── trace.py - ReasoningTrace, TraceStep
│   ├── score.py - ScoreRequest, ScoreResponse
│   ├── dataset.py - DatasetBuildRequest, DatasetManifest
│   └── ungar.py - UNGAR request/response schemas
├── helpers/
│   ├── validation.py - get_trace_or_404 helper
│   ├── datasets.py - Manifest I/O, dataset stats
│   └── trace_conversion.py - Trace format conversions
├── integrations/
│   └── ungar/ - Optional UNGAR integration (availability, high_card_duel)
└── training/
    ├── __init__.py
    └── renderers.py - Tunix SFT prompt rendering (Gemma template)
```

### Training Infrastructure (M8)

```
backend/training/
└── sft_smoke.py - Standalone smoke test script (198 lines)
```

**Note:** M09 will expand this structure significantly.

---

## Dependencies

### Backend Core
- `fastapi>=0.115.5`
- `uvicorn>=0.34.0`
- `sqlalchemy>=2.0.36` (async)
- `asyncpg>=0.30.0` (PostgreSQL driver)
- `alembic>=1.14.0` (migrations)
- `pydantic>=2.10.3`
- `pydantic-settings>=2.6.1`
- `httpx>=0.28.1`

### Backend Development
- `pytest>=8.3.4`
- `pytest-asyncio>=0.24.0`
- `pytest-cov>=6.0.0`
- `ruff>=0.8.4`
- `mypy>=1.13.0`

### Backend Optional Extras

**`[ungar]`** - Game trace generation
```
ungar @ git+https://github.com/DigitalPhonetics/UNGAR.git@0e29e104
```

**`[training]`** - Training smoke test (M8)
```
jax[cpu]>=0.4.20
flax>=0.7.0
optax>=0.1.7
```

### Frontend
- React 18
- TypeScript 5
- Vite 6
- Vitest for testing

### E2E
- Playwright
- Chromium browser

---

## Configuration

**Environment Variables (14 validated settings):**

| Variable | Default | Validation |
|----------|---------|------------|
| `BACKEND_PORT` | 8000 | 1-65535 |
| `DATABASE_URL` | postgresql+asyncpg://... | Required |
| `DB_POOL_SIZE` | 5 | 1-50 |
| `DB_MAX_OVERFLOW` | 10 | 0-50 |
| `DB_POOL_TIMEOUT` | 30 | 1-300 seconds |
| `TRACE_MAX_BYTES` | 1048576 (1MB) | 1KB-10MB |
| `REDIAI_MODE` | mock | "mock" or "real" |
| `REDIAI_BASE_URL` | http://localhost:8080 | Valid URL |
| `REDIAI_HEALTH_PATH` | /health | Must start with "/" |
| `REDIAI_HEALTH_CACHE_TTL_SECONDS` | 30 | 0-300 |

---

## Known Issues & Limitations (Pre-M09)

### From M8 Audit

**Low Priority Issues:**
1. Python-level filtering fetches `limit × 10` (acceptable at current scale)
2. Format validation in endpoint vs Pydantic Literal
3. Test fixtures duplicated across files
4. Smoke script not type-checked by mypy

**Missing Tests:**
1. No test for `format=tunix_sft` export
2. No test for invalid format parameter

**None of these block M09 implementation.**

---

## Dataset System (M8)

### Dataset Manifests

**Storage:** `backend/datasets/{dataset_key}/manifest.json`

**Manifest Structure:**
```json
{
  "dataset_key": "name-version",
  "build_id": "uuid",
  "dataset_name": "name",
  "dataset_version": "version",
  "dataset_schema_version": "1.0",
  "created_at": "timestamp",
  "filters": {},
  "selection_strategy": "latest|random",
  "seed": null,
  "trace_ids": [],
  "trace_count": 0,
  "stats": {
    "avg_step_count": 0,
    "min_step_count": 0,
    "max_step_count": 0,
    "avg_total_chars": 0
  }
}
```

### Export Formats (M8)

1. **`trace`** - Raw trace data (prompt, trace_steps, final_answer)
2. **`tunix_sft`** - Gemma chat template formatted:
   ```
   <start_of_turn>user
   {prompt}<end_of_turn>
   <start_of_turn>model
   Reasoning:
   1. {step1}
   2. {step2}
   Answer: {final_answer}<end_of_turn>
   ```

---

## CI/CD State

### GitHub Actions Workflows

**1. Main CI (`.github/workflows/ci.yml`)**
- Path filtering (backend, frontend, e2e)
- Backend: ruff, mypy, pytest with 70% coverage gate
- Frontend: vitest, build
- E2E: Playwright with PostgreSQL service

**2. UNGAR Integration (`.github/workflows/ungar-integration.yml`)**
- Manual dispatch + nightly
- Non-blocking
- Full UNGAR test suite (6 additional tests)

**3. Training Smoke (`.github/workflows/training-smoke.yml`)** - M8
- Manual dispatch + nightly (2 AM UTC)
- Non-blocking
- Validates dataset schema and SFT format

---

## Documentation

### Existing Docs (M8)

1. `README.md` - User quickstart, API examples
2. `tunix-rt.md` - Complete technical reference (831 lines)
3. `VISION.md` - Project goals and roadmap
4. `SECURITY_NOTES.md` - Security baseline
5. `docs/M08_BASELINE.md` - M8 pre-implementation state
6. `docs/M08_SUMMARY.md` - M8 completion summary (810 lines)
7. `docs/M08_PROGRESS.md` - M8 implementation tracking
8. `docs/M07_UNGAR_INTEGRATION.md` - UNGAR guide (236 lines)
9. `docs/adr/` - 4 Architecture Decision Records

---

## M09 Scope Preview

**What M09 Will Add:**

### Phase 1: Dataset Contract
- `TrainingExample` schema (new abstraction)
- Enhanced Gemma IT formatting helpers
- Snapshot tests for format stability

### Phase 2: Exporters
- Extend dataset export with `training_example` format
- Bulk trace import endpoint (`POST /api/traces/batch`)

### Phase 3: Training Runner
- Top-level `training/` folder for scripts
- `train_sft_tunix.py` - Actual Tunix SFT runner (10-50 steps)
- Config files (`configs/sft_tiny.yaml`)
- Run manifest tracking

### Phase 4: Evaluation Loop
- Static eval set (`training/evalsets/eval_v1.jsonl`)
- `eval_generate.py` - Generate pre/post training outputs
- `eval_report.py` - Compute delta metrics
- Artifacts stored in `artifacts/training_runs/<run_id>/`

### Phase 5: CI Guardrails
- Updated `.coveragerc` to omit `training/` from coverage
- Enhanced training smoke workflow

### Documentation
- `docs/M09_DATASET_FORMAT.md`
- `docs/M09_TRAINING_QUICKSTART.md`
- `docs/M09_EVAL_LOOP.md`

---

## Success Criteria for M09

**Must Achieve:**
1. ✅ All M8 tests still passing (88 backend, 11 frontend, 7 E2E)
2. ✅ Coverage remains ≥70% (acceptable slight dip due to new code)
3. ✅ No breaking changes to existing APIs
4. ✅ Actual Tunix SFT run completes (even if tiny - 10-50 steps)
5. ✅ Eval loop produces before/after comparison
6. ✅ Main CI stays green (optional workflows for training depth)

**New Capabilities:**
1. TrainingExample schema with deterministic rendering
2. Bulk trace import for eval results
3. End-to-end training + eval workflow documented
4. Run manifests for reproducibility

---

## Baseline Validation

**Commands to reproduce this baseline:**

```bash
# Backend tests
cd backend
python -m pytest -v

# Coverage
python -m pytest --cov=tunix_rt_backend --cov-report=term

# Frontend tests  
cd ../frontend
npm run test

# E2E tests
cd ../e2e
npm run test

# Git state
git rev-parse HEAD  # ec59ac8
git status          # Should be clean
```

---

**Baseline Established:** ✅  
**Ready for M09 Implementation:** ✅  
**Last Updated:** December 21, 2025

