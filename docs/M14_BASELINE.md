# M14 Baseline State

**Date:** December 22, 2025  
**Baseline Milestone:** M13 Complete ✅  
**Previous Milestone:** M13 - Tunix Runtime Execution

---

## Executive Summary

This baseline captures the state of tunix-rt **before M14 (Tunix Run Registry)** begins. M13 established **optional, gated Tunix execution** following the UNGAR pattern with dry-run and local execution modes. M14 will add **persistent storage and a run registry API** for tracking execution history without changing execution semantics.

---

## Test Metrics

### Backend
- **Total Tests:** 180 passing, 12 skipped (optional dependencies)
- **Coverage:** 82% line coverage (exceeding 70% gate by 12%)
- **Test Markers:** unit, integration, ungar, training, tunix
- **Duration:** ~21.8s

### Frontend
- **Total Tests:** 21 passing
- **Coverage:** 77% line coverage (exceeding 60% gate)
- **Duration:** ~2.2s

### E2E
- **Total Tests:** 5 tests (Playwright)
- **Status:** All passing

---

## Current Architecture

### Backend Structure
```
backend/tunix_rt_backend/
├── app.py (847 lines, thin controllers)
├── db/
│   ├── base.py
│   └── models/
│       ├── trace.py
│       └── score.py
├── services/
│   ├── traces_batch.py
│   ├── datasets_export.py
│   ├── datasets_builder.py
│   ├── ungar_generator.py
│   ├── tunix_export.py (M12)
│   └── tunix_execution.py (M13)
├── integrations/
│   ├── ungar/
│   │   ├── availability.py
│   │   └── high_card_duel.py
│   └── tunix/
│       ├── availability.py (M13 - real checks)
│       └── manifest.py (M12)
├── schemas/
│   ├── trace.py
│   ├── dataset.py
│   ├── score.py
│   ├── ungar.py
│   └── tunix.py (M12/M13)
└── helpers/
    ├── datasets.py
    └── traces.py
```

### Tunix Integration (M13 State)
- **Design:** Optional, gated execution
- **Runtime Required:** No (graceful degradation with 501)
- **Endpoints:** 4 (status, export, manifest, run)
- **Execution Modes:** dry-run (validation only), local (subprocess)
- **Persistence:** None (ephemeral results only)

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

### Tunix (M12/M13 - Execution Ready)
- `GET /api/tunix/status` - Check Tunix integration status
- `POST /api/tunix/sft/export` - Export traces in Tunix SFT format (JSONL)
- `POST /api/tunix/sft/manifest` - Generate training manifest (YAML)
- `POST /api/tunix/run` - Execute Tunix training run (M13)

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
| Backend Tests | ✅ PASS | 180/180 |
| Backend Coverage | ✅ PASS | 82% (≥70%) |
| Frontend Tests | ✅ PASS | 21/21 |
| Frontend Coverage | ✅ PASS | 77% (≥60%) |
| E2E Tests | ✅ PASS | 5/5 |
| Linting (ruff) | ✅ PASS | 0 errors |
| Type Checking (mypy) | ✅ PASS | 0 errors |
| Security (pip-audit) | ✅ PASS | 0 vulnerabilities |

---

## CI/CD Configuration

### Workflows
- **ci.yml** - Main CI pipeline (path-filtered)
- **ungar-integration.yml** - Optional UNGAR tests (non-blocking, manual)
- **tunix-integration.yml** - Optional Tunix tests (non-blocking, manual)

### CI Jobs (Main Pipeline)
1. **changes** - Path filtering
2. **backend** - Ruff, mypy, pytest, coverage
3. **frontend** - Vitest, build
4. **e2e** - Playwright tests
5. **security-backend** - pip-audit
6. **security-secrets** - gitleaks

---

## Optional Dependencies (Current)

### UNGAR (backend[ungar])
- **Purpose:** High Card Duel trace generator
- **Pattern:** Lazy imports, 501 responses when unavailable
- **Tests:** 6 optional tests marked `@pytest.mark.ungar`
- **Status:** M7 complete

### Training (backend[training])
- **Purpose:** JAX/Flax training smoke tests
- **Pattern:** Subprocess execution, dry-run validation
- **Tests:** 2 optional tests marked `@pytest.mark.training`
- **Status:** M11 complete

### Tunix (M13 State)
- **Purpose:** Training execution (dry-run + local modes)
- **Pattern:** Subprocess execution, graceful degradation
- **Tests:** 20 default tests + 2 optional runtime tests marked `@pytest.mark.tunix`
- **Status:** M13 complete (execution only, no persistence)

---

## Database Schema (Pre-M14)

### Current Tables
```sql
-- traces table (M02)
CREATE TABLE traces (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    trace_version VARCHAR(64),
    payload JSON NOT NULL
);
CREATE INDEX ix_traces_created_at ON traces(created_at);

-- scores table (M04)
CREATE TABLE scores (
    id SERIAL PRIMARY KEY,
    trace_id UUID NOT NULL REFERENCES traces(id),
    scorer VARCHAR(64) NOT NULL,
    score FLOAT NOT NULL,
    rationale TEXT,
    scored_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX ix_scores_trace_id ON scores(trace_id);
```

**Note:** No `tunix_runs` table exists yet. M14 will add persistent run storage.

---

## Known Issues (Pre-M14)

**None blocking.** All gates green.

**M13 Limitations (By Design):**
- No run persistence (results are ephemeral)
- No run history API
- No run detail retrieval
- No filtering/querying of past runs
- Frontend only shows latest run result

**Deferred items:**
- npm audit (4 moderate dev dependencies) - deferred
- Vite 7 / Vitest 4 upgrade - requires dedicated testing

---

## M14 Scope (Planned)

### What M14 Will Add
1. **tunix_runs database table** - UUID primary key, indexed timestamps
2. **Alembic migration** - Schema upgrade/downgrade support
3. **Run persistence** - Create record immediately, update on completion
4. **GET /api/tunix/runs endpoint** - Paginated list with filtering
5. **GET /api/tunix/runs/{run_id} endpoint** - Full run details
6. **Frontend Run History panel** - Collapsible section with manual refresh
7. **Stdout/stderr storage** - Truncated to 10KB per field
8. **Graceful DB failures** - Log but don't fail user requests

### What M14 Will NOT Do
- Add run deletion/retry/cancellation - deferred
- Add streaming logs - deferred
- Add auto-refresh or websockets - deferred
- Add run metadata mutation - deferred
- Change execution semantics - execution unchanged from M13
- Add background job processing - deferred

### Design Constraints
- **No execution changes** - M13 execution logic remains identical
- **Graceful degradation** - DB write failures don't break runs
- **Default CI must pass** - No new dependencies
- **Maintain coverage** - 82% backend, 77% frontend (no regression)
- **Use dry-run for tests** - No Tunix runtime dependency

---

## Acceptance Criteria for M14

| Criterion | Target |
|-----------|--------|
| Default CI green | ✅ Required |
| Backend tests passing | ≥180 (baseline) |
| Backend coverage | ≥82% (maintain) |
| Frontend tests passing | ≥21 (baseline) |
| Migration works | Up/down without data loss |
| List endpoint works | Pagination + filtering (status, dataset_key, mode) |
| Detail endpoint works | Full run details by UUID |
| UI renders history | Collapsible panel with table |
| UI handles empty state | "No runs found" message |
| UI displays details | Expandable rows with stdout/stderr |
| DB failures graceful | Log error, return execution result |

---

## Performance Baseline

**Batch Import (1000 traces):** ~1.2s  
**Dataset Export (100 traces, tunix_sft):** ~103ms  
**Tunix Export (100 traces):** ~200ms (M12)  
**Manifest Generation:** <50ms (M12)  
**Tunix Dry-Run:** ~500ms (M13)  
**Database Pool:** 5 connections, 10 overflow

---

## Files to Monitor for Changes

### Backend (Expected Changes)
- `backend/tunix_rt_backend/db/models/tunix_run.py` - NEW model
- `backend/tunix_rt_backend/db/models/__init__.py` - Export TunixRun
- `backend/alembic/versions/XXXXX_add_tunix_runs_table.py` - NEW migration
- `backend/tunix_rt_backend/services/tunix_execution.py` - Add persistence
- `backend/tunix_rt_backend/schemas/tunix.py` - Add list/detail schemas
- `backend/tunix_rt_backend/app.py` - Add list/detail endpoints
- `backend/tests/test_tunix_registry.py` - NEW test file

### Frontend (Expected Changes)
- `frontend/src/App.tsx` - Add Run History panel
- `frontend/src/api/client.ts` - Add list/detail functions
- `frontend/src/App.test.tsx` - Add history tests

### Documentation (Expected Changes)
- `docs/M14_RUN_REGISTRY.md` - NEW complete guide
- `docs/M14_SUMMARY.md` - NEW milestone summary
- `tunix-rt.md` - Update with M14 endpoints + schema
- `README.md` - Add run history examples

---

## Commit History (Recent M13)

```
<M13 commits - execution phase>
docs(m13): add M13_SUMMARY.md
docs(m13): add M13_TUNIX_EXECUTION.md guide
test(m13): add 7 Tunix execution frontend tests
feat(m13): add Tunix execution UI (dry-run + local)
test(m13): add 22 Tunix execution backend tests
feat(m13): add POST /api/tunix/run endpoint
feat(m13): add TunixExecutionService (dry-run + local)
feat(m13): add execution request/response schemas
feat(m13): add real Tunix availability checks
chore(m13): add backend[tunix] optional extra
chore(m13): add baseline documentation
</commit_history>

---

## Baseline Verification

**Command to reproduce:**
```powershell
# Backend tests
cd backend
pytest --cov=tunix_rt_backend --cov-report=term -v

# Frontend tests
cd ../frontend
npm run test

# E2E tests
cd ../e2e
npx playwright test
```

**Expected Results:**
- Backend: 180 passed, 12 skipped, 82% coverage
- Frontend: 21 passed, 77% coverage
- E2E: 5 passed

---

## M13 → M14 Transition

### M13 Delivered
✅ Real Tunix availability checks  
✅ TunixExecutionService (dry-run + local)  
✅ POST /api/tunix/run endpoint  
✅ Subprocess execution with output capture  
✅ Frontend execution UI  
✅ 22 execution tests (20 default + 2 optional)  
✅ 82% backend coverage  

### M14 Will Extend
🔄 Add persistent run storage (tunix_runs table)  
🔄 Add Alembic migration for schema  
🔄 Persist runs immediately, update on completion  
🔄 Add GET /api/tunix/runs (list with pagination/filtering)  
🔄 Add GET /api/tunix/runs/{run_id} (detail view)  
🔄 Add frontend Run History panel  
🔄 Graceful DB failure handling  
🔄 Maintain default CI green (no new dependencies)  

---

**Baseline Status:** ✅ **VERIFIED**  
**Ready for M14:** Yes  
**Next Step:** Begin Phase 1 - Database schema + migration
