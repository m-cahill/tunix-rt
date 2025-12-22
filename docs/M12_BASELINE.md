# M12 Baseline State

**Date:** December 22, 2025  
**Baseline Commit:** `4a3bdc39cedf75f061946987c0aa08e34104f9fb`  
**Commit Message:** `docs(m11): populate M11 milestone summary`  
**Previous Milestone:** M11 Complete ✅

---

## Executive Summary

This baseline captures the state of tunix-rt **before M12 (Tunix Integration Skeleton)** begins. M11 established production-grade architecture with service layer extraction and optional dependency patterns. M12 will add Tunix integration as a **mock-first artifact emitter** following the UNGAR pattern.

---

## Test Metrics

### Backend
- **Total Tests:** 146 passing, 10 skipped (optional dependencies)
- **Coverage:** 84% line coverage (exceeding 70% gate)
- **Test Markers:** unit, integration, ungar, training

### Frontend
- **Total Tests:** 16 passing
- **Coverage:** 77% line coverage (exceeding 60% gate)

### E2E
- **Total Tests:** 5 tests (Playwright)
- **Status:** All passing

---

## Current Architecture

### Backend Structure
```
backend/tunix_rt_backend/
├── app.py (588 lines, thin controllers)
├── services/
│   ├── traces_batch.py
│   ├── datasets_export.py
│   ├── datasets_builder.py
│   └── ungar_generator.py
├── integrations/
│   └── ungar/
│       ├── availability.py
│       └── high_card_duel.py
├── schemas/
│   ├── trace.py
│   ├── dataset.py
│   ├── score.py
│   └── ungar.py
└── helpers/
    ├── datasets.py
    └── traces.py
```

### Export Formats (Existing)
1. **trace** - Raw trace data
2. **tunix_sft** - Gemma chat template with reasoning steps
3. **training_example** - Prompt/response pairs

### Optional Dependencies
- **UNGAR** (backend[ungar]) - High Card Duel trace generator
- **Training** (backend[training]) - JAX/Flax for training smoke tests

---

## API Endpoints (Current)

### Traces
- `POST /api/traces` - Create trace
- `GET /api/traces/{id}` - Get trace by ID
- `GET /api/traces` - List traces (paginated)
- `POST /api/traces/batch` - Batch import
- `POST /api/traces/{id}/score` - Score trace
- `GET /api/traces/compare` - Compare two traces

### Datasets
- `POST /api/datasets/build` - Build dataset manifest
- `GET /api/datasets/{dataset_key}/export.jsonl` - Export dataset

### UNGAR (Optional)
- `GET /api/ungar/status` - Check UNGAR availability
- `POST /api/ungar/high-card-duel/generate` - Generate traces
- `GET /api/ungar/high-card-duel/export.jsonl` - Export UNGAR traces

### Health
- `GET /api/health` - Application health
- `GET /api/redi/health` - RediAI health

---

## Quality Gates (Current)

| Gate | Status | Value |
|------|--------|-------|
| Backend Tests | ✅ PASS | 146/146 |
| Backend Coverage | ✅ PASS | 84% (≥70%) |
| Frontend Tests | ✅ PASS | 16/16 |
| Frontend Coverage | ✅ PASS | 77% (≥60%) |
| E2E Tests | ✅ PASS | 5/5 |
| Linting (ruff) | ✅ PASS | 0 errors |
| Type Checking (mypy) | ✅ PASS | 0 errors |
| Security (pip-audit) | ✅ PASS | 0 vulnerabilities |

---

## CI/CD Configuration

### Workflows
- **ci.yml** - Main CI pipeline (path-filtered)
- **ungar-integration.yml** - Optional UNGAR tests (non-blocking)

### CI Jobs
1. **changes** - Path filtering
2. **backend** - Ruff, mypy, pytest, coverage
3. **frontend** - Vitest, build
4. **e2e** - Playwright tests
5. **security-backend** - pip-audit
6. **security-secrets** - gitleaks

---

## Known Issues (Pre-M12)

**None blocking.** All gates green.

**Deferred items:**
- npm audit (4 moderate dev dependencies) - deferred to future milestone
- Vite 7 / Vitest 4 upgrade - requires dedicated testing

---

## M12 Scope (Planned)

### What M12 Will Add
1. **Tunix availability shim** (no runtime dependency)
2. **Tunix JSONL export** (reuse tunix_sft format)
3. **Tunix manifest generator** (YAML configs)
4. **Three new endpoints** (/api/tunix/status, /sft/export, /sft/manifest)
5. **Minimal frontend panel** (status + export/manifest buttons)
6. **Default tests** (Tunix not installed, 501 responses)

### What M12 Will NOT Do
- Install/import Tunix runtime
- Execute training jobs
- Add TPU orchestration
- Create new export formats

---

## Files to Monitor for Changes

### Backend (Expected Changes)
- `backend/tunix_rt_backend/app.py` - New endpoints
- `backend/tunix_rt_backend/integrations/tunix/` - New module
- `backend/tunix_rt_backend/services/tunix_export.py` - New service
- `backend/tunix_rt_backend/schemas/tunix.py` - New schemas
- `backend/tests/test_tunix.py` - New tests

### Frontend (Expected Changes)
- `frontend/src/App.tsx` - Tunix panel
- `frontend/src/App.test.tsx` - Panel tests

### Configuration (Expected Changes)
- `backend/pyproject.toml` - Optional tunix extra (if needed)
- `.coveragerc` - Omit patterns for optional Tunix code

---

## Acceptance Criteria for M12

| Criterion | Target |
|-----------|--------|
| Default CI green | ✅ Required |
| Backend tests passing | ≥146 (baseline) |
| Backend coverage | ≥84% (baseline) |
| Frontend tests passing | ≥16 (baseline) |
| New endpoints (3) | /status, /sft/export, /sft/manifest |
| Tunix endpoints work without Tunix | 501/503 graceful degradation |
| JSONL export uses existing format | tunix_sft |
| Manifest generation works | Valid YAML output |
| Frontend panel functional | Status + 2 buttons |

---

## Performance Baseline

**Batch Import (1000 traces):** ~1.2s  
**Dataset Export (100 traces, tunix_sft):** ~103ms  
**Database Pool:** 5 connections, 10 overflow

---

## Commit History (Recent)

```
4a3bdc3 - docs(m11): populate M11 milestone summary
45e9f47 - test(m11): add 5 frontend component tests - coverage now 77%
9c0fb60 - test(m11): add training script dry-run smoke tests via subprocess
4b78512 - docs(m11): update tunix-rt.md with M11 architecture changes
7a92a3d - test(m11): add service tests for UNGAR generator and dataset builder
```

---

## Baseline Verification

**Command to reproduce:**
```bash
cd backend
pytest --cov=tunix_rt_backend --cov-report=term -v
cd ../frontend
npm run test
cd ../e2e
npx playwright test
```

**Expected Results:**
- Backend: 146 passed, 10 skipped, 84% coverage
- Frontend: 16 passed, 77% coverage
- E2E: 5 passed

---

**Baseline Status:** ✅ **VERIFIED**  
**Ready for M12:** Yes  
**Next Step:** Begin Phase 1 - Tunix availability shim + schemas
