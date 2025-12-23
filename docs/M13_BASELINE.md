# M13 Baseline State

**Date:** December 22, 2025  
**Baseline Milestone:** M12 Complete ✅  
**Previous Milestone:** M12 - Tunix Integration Skeleton (Mock-First)

---

## Executive Summary

This baseline captures the state of tunix-rt **before M13 (Tunix Runtime Execution)** begins. M12 established **mock-first Tunix integration** as an artifact-based bridge (JSONL exports + YAML manifests) without requiring Tunix runtime. M13 will add **optional, gated execution** following the UNGAR pattern exactly.

---

## Test Metrics

### Backend
- **Total Tests:** 160 passing, 10 skipped (optional dependencies)
- **Coverage:** 92% line coverage (exceeding 70% gate by 22%)
- **Test Markers:** unit, integration, ungar, training, tunix (M12 added)
- **Duration:** ~13.2s

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
├── app.py (741 lines, thin controllers)
├── services/
│   ├── traces_batch.py
│   ├── datasets_export.py
│   ├── datasets_builder.py
│   ├── ungar_generator.py
│   └── tunix_export.py (M12)
├── integrations/
│   ├── ungar/
│   │   ├── availability.py
│   │   └── high_card_duel.py
│   └── tunix/ (M12)
│       ├── availability.py (mock-first)
│       └── manifest.py
├── schemas/
│   ├── trace.py
│   ├── dataset.py
│   ├── score.py
│   ├── ungar.py
│   └── tunix.py (M12)
└── helpers/
    ├── datasets.py
    └── traces.py
```

### Tunix Integration (M12 State)
- **Design:** Mock-first, artifact-based
- **Runtime Required:** No (always returns False)
- **Endpoints:** 3 (status, export, manifest)
- **Export Format:** Reuses tunix_sft from M09
- **Manifest Generation:** YAML configs for SFT training

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

### Tunix (M12 - Artifact-Based)
- `GET /api/tunix/status` - Check Tunix integration status
- `POST /api/tunix/sft/export` - Export traces in Tunix SFT format (JSONL)
- `POST /api/tunix/sft/manifest` - Generate training manifest (YAML)

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
| Backend Tests | ✅ PASS | 160/160 |
| Backend Coverage | ✅ PASS | 92% (≥70%) |
| Frontend Tests | ✅ PASS | 21/21 |
| Frontend Coverage | ✅ PASS | 77% (≥60%) |
| E2E Tests | ✅ PASS | 5/5 |
| Linting (ruff) | ✅ PASS | 0 errors |
| Type Checking (mypy) | ✅ PASS | 0 errors (35 files) |
| Security (pip-audit) | ✅ PASS | 0 vulnerabilities |

---

## CI/CD Configuration

### Workflows
- **ci.yml** - Main CI pipeline (path-filtered)
- **ungar-integration.yml** - Optional UNGAR tests (non-blocking, manual)
- **tunix-integration.yml** - Optional Tunix tests (non-blocking, manual) [M12]

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

### Tunix (M12 State)
- **Purpose:** Artifact generation (JSONL + manifests)
- **Pattern:** Mock-first, no runtime dependency
- **Tests:** 14 default tests (no Tunix required)
- **Status:** M12 complete (artifact-based only)

---

## Known Issues (Pre-M13)

**None blocking.** All gates green.

**M12 Limitations (By Design):**
- No Tunix runtime integration (mock-first approach)
- No training execution capability
- `tunix_available()` always returns False
- Manifests are best-effort (no CLI validation)

**Deferred items:**
- npm audit (4 moderate dev dependencies) - deferred
- Vite 7 / Vitest 4 upgrade - requires dedicated testing

---

## M13 Scope (Planned)

### What M13 Will Add
1. **Real `tunix_available()` checks** - Detect Tunix CLI + imports
2. **TunixExecutionService** - Dry-run + local execution modes
3. **POST /api/tunix/run endpoint** - Execute training runs
4. **backend[tunix] optional extra** - Follow UNGAR pattern
5. **Subprocess execution** - Capture stdout/stderr/exit code
6. **Frontend execution UI** - "Run with Tunix (Local)" button
7. **Optional tests** - `@pytest.mark.tunix` for local execution
8. **CI workflow** - tunix-runtime.yml (manual, dry-run only)

### What M13 Will NOT Do
- Add database persistence (training_runs table) - deferred to M14
- Parse training results/checkpoints - deferred to M14
- Add evaluation metrics - deferred to M14
- Add TPU orchestration - future milestone
- Add async/background execution - future milestone
- Add streaming logs - future milestone

### Design Constraints
- **Default CI must pass** without Tunix installed
- **Execution must be opt-in** and fail gracefully (501)
- **No TPU assumptions** (CPU/GPU only)
- **No coupling** in core code paths
- **Follow UNGAR pattern exactly** (lazy imports, optional extra)

---

## Acceptance Criteria for M13

| Criterion | Target |
|-----------|--------|
| Default CI green | ✅ Required |
| Backend tests passing | ≥160 (baseline) |
| Backend coverage | ≥92% (maintain) |
| Frontend tests passing | ≥21 (baseline) |
| Dry-run mode works | Validates manifest + dataset |
| Local mode works | Executes Tunix CLI, captures logs |
| 501 response when unavailable | Graceful degradation |
| Optional tests pass | With Tunix installed |
| CI workflow added | tunix-runtime.yml (manual) |

---

## Performance Baseline

**Batch Import (1000 traces):** ~1.2s  
**Dataset Export (100 traces, tunix_sft):** ~103ms  
**Tunix Export (100 traces):** ~200ms (M12)  
**Manifest Generation:** <50ms (M12)  
**Database Pool:** 5 connections, 10 overflow

---

## Files to Monitor for Changes

### Backend (Expected Changes)
- `backend/tunix_rt_backend/integrations/tunix/availability.py` - Real Tunix checks
- `backend/tunix_rt_backend/services/tunix_execution.py` - NEW service
- `backend/tunix_rt_backend/schemas/tunix.py` - Add run schemas
- `backend/tunix_rt_backend/app.py` - Add /api/tunix/run endpoint
- `backend/tests/test_tunix_execution.py` - NEW test file
- `backend/pyproject.toml` - Add backend[tunix] optional extra

### Frontend (Expected Changes)
- `frontend/src/App.tsx` - Add execution button + results display
- `frontend/src/api/client.ts` - Add executeTunixRun function
- `frontend/src/App.test.tsx` - Add execution tests

### CI (Expected Changes)
- `.github/workflows/tunix-runtime.yml` - NEW workflow (manual dispatch)

### Documentation (Expected Changes)
- `docs/M13_TUNIX_EXECUTION.md` - NEW complete guide
- `docs/M13_SUMMARY.md` - NEW milestone summary
- `tunix-rt.md` - Update with M13 endpoints
- `README.md` - Add execution examples

---

## Commit History (Recent M12)

```
<M12 commits - artifact generation phase>
docs(m12): add M12_SUMMARY.md
docs(m12): add M12_TUNIX_INTEGRATION.md guide
ci(m12): add tunix-integration.yml workflow
test(m12): add 5 Tunix frontend tests
feat(m12): add Tunix frontend panel
test(m12): add 14 Tunix integration tests
feat(m12): add Tunix API endpoints
feat(m12): add Tunix manifest generation (YAML)
feat(m12): add Tunix export service (reuse tunix_sft)
feat(m12): add Tunix request/response schemas
feat(m12): add Tunix availability shim (mock-first)
chore(m12): add baseline documentation
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
- Backend: 160 passed, 10 skipped, 92% coverage
- Frontend: 21 passed, 77% coverage
- E2E: 5 passed

---

## M12 → M13 Transition

### M12 Delivered
✅ Artifact-based Tunix integration (no runtime)  
✅ JSONL export (tunix_sft format)  
✅ YAML manifest generation  
✅ 3 API endpoints (status, export, manifest)  
✅ Frontend panel for exports  
✅ 14 default tests (no Tunix required)  
✅ 92% backend coverage  

### M13 Will Extend
🔄 Transform mock-first → optional runtime  
🔄 Add execution capability (dry-run + local)  
🔄 Add backend[tunix] optional extra  
🔄 Add optional tests for real execution  
🔄 Add CI workflow for runtime verification  
🔄 Maintain default CI green (no Tunix)  

---

**Baseline Status:** ✅ **VERIFIED**  
**Ready for M13:** Yes  
**Next Step:** Begin Phase 1 - Real Tunix availability checks + schemas
