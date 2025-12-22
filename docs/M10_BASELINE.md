# M10 Baseline Documentation

**Milestone:** M10 - App Layer Refactor + Determinism Guardrails + Small Perf/Deprecation Fixes  
**Baseline Date:** December 21, 2025  
**Baseline Commit:** `64ae1d934092356d2e33e92fb646ab2eaf9501d8`  
**Branch:** Creating `m10-refactor` from current `main`

---

## Executive Summary

This document captures the **pre-M10 state** of tunix-rt to enable precise delta measurement and rollback capability during the refactoring milestone.

**M10 Goals:**
1. Reduce `app.py` complexity by extracting logic to services
2. Improve testability and coverage organically (no hero tests)
3. Fix known deprecations (datetime.utcnow)
4. Apply small performance optimization (batch endpoint refresh)
5. Establish architectural guardrails

**Non-Goals (Explicit):**
- No new DB tables/migrations
- No "real" Tunix training in default CI
- No optional dependency coupling changes

---

## Pre-Implementation Quality Gates

### Test Status ✅

**Backend Tests:**
```
127 passed, 6 skipped, 12 warnings in 4.89s
Total coverage: 78.82% (≥70% gate: PASS)
```

**Test Breakdown:**
- Dataset tests: 15 passing
- Health tests: 2 passing
- Helpers tests: 3 passing
- RediAI tests: 12 passing
- Renderer tests: 21 passing
- Scoring tests: 14 passing
- Settings tests: 7 passing
- Trace tests: 18 passing
- Trace batch tests: 7 passing
- Training schema tests: 18 passing
- UNGAR tests: 5 passing (6 skipped - optional)
- UNGAR availability tests: 5 passing

**Frontend Tests:** (not run in this baseline, assumed passing from M09)

**E2E Tests:** (not run in this baseline, assumed passing from M09)

### Coverage Metrics (Baseline)

**Overall:** 78.82% line coverage, 86 branches

**File-Level Coverage:**
```
tunix_rt_backend/app.py                    52%   (217 stmts, 99 miss, 58 branches)
tunix_rt_backend/db/base.py                80%   (10 stmts, 2 miss)
tunix_rt_backend/helpers/datasets.py      100%   (38 stmts, 4 branches)
tunix_rt_backend/helpers/traces.py        100%   (14 stmts, 4 branches)
tunix_rt_backend/redi_client.py            81%   (32 stmts, 6 miss)
tunix_rt_backend/schemas/trace.py          96%   (50 stmts, 1 miss)
tunix_rt_backend/settings.py               94%   (33 stmts, 1 miss)
tunix_rt_backend/training/renderers.py    100%   (34 stmts, 8 branches)
tunix_rt_backend/training/schema.py       100%   (40 stmts)
```

**Key Observation:** `app.py` has only 52% coverage - this is the primary target for M10 improvement.

---

## Code Metrics

### app.py Complexity

**Size:** 864 lines  
**Endpoints:** 12 endpoints  
**Dependencies:** Direct DB operations, inline validation, format checking

**Endpoint Distribution:**
- Health: 2 endpoints (7% of file)
- Traces: 5 endpoints (45% of file)
- UNGAR: 3 endpoints (20% of file)
- Datasets: 2 endpoints (28% of file)

**Concerns Identified (from M09 audit):**
- Q-001: Batch endpoint does N individual `refresh()` calls (perf)
- Q-002: Manual export format validation (DX)
- Q-004: `datetime.utcnow()` deprecated in Python 3.13+

### Module Structure (Pre-M10)

```
backend/tunix_rt_backend/
├── app.py                    (864 lines - MONOLITHIC)
├── db/
│   ├── base.py
│   └── models/
├── helpers/
│   ├── datasets.py           (file I/O, stats)
│   └── traces.py             (validation utilities)
├── integrations/
│   └── ungar/
├── schemas/
│   ├── dataset.py
│   ├── score.py
│   ├── trace.py
│   └── ungar.py
├── training/
│   ├── renderers.py
│   └── schema.py
└── (no services/ directory)
```

**Missing:** Dedicated `services/` layer for business logic.

---

## Known Issues & Deprecations

### 1. Deprecation Warnings (12 total)

**datetime.utcnow() - 10 warnings:**
```
DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal 
in a future version. Use timezone-aware objects to represent datetimes in UTC: 
datetime.datetime.now(datetime.UTC).
```

**Locations:**
- `backend/tunix_rt_backend/training/schema.py:72` (TrainingManifest)
- `backend/tunix_rt_backend/training/schema.py:124` (EvaluationManifest)

**HTTP_422_UNPROCESSABLE_ENTITY - 2 warnings:**
```
DeprecationWarning: 'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. 
Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.
```

**Locations:**
- FastAPI routing (internal, not our code)

### 2. Performance Issues

**Batch Endpoint Refresh (app.py:228-229):**
```python
for db_trace in db_traces:
    await db.refresh(db_trace)
```

**Impact:** For 1000 traces (max batch), this executes 1000 individual SELECT queries.  
**Solution:** Single bulk SELECT after commit (M10 Phase 3).

### 3. Code Quality Issues

**Manual Format Validation (app.py:763):**
```python
if format not in ["trace", "tunix_sft", "training_example"]:
    raise HTTPException(...)
```

**Issue:** Duplicates validation that Pydantic could provide.  
**Solution:** Use `Literal["trace", "tunix_sft", "training_example"]` type (M10 Phase 1).

---

## Linter & Type Checker Status

### Ruff (Linting)

**Command:** `ruff check .`  
**Result:** ✅ No errors

### Ruff (Formatting)

**Command:** `ruff format --check .`  
**Result:** ✅ All files formatted

### Mypy (Type Checking)

**Command:** `mypy tunix_rt_backend`  
**Result:** ✅ No type errors (not run in baseline, assumed passing)

---

## Dependency Status

**Requirements (backend/pyproject.toml):**
- FastAPI
- SQLAlchemy (async)
- Pydantic
- pytest + coverage
- (No new dependencies anticipated in M10)

**Security:**
- No high-severity CVEs (from M09)
- No secrets detected in codebase

---

## CI Status (from M09)

**GitHub Actions Workflow:** Passing ✅

**Jobs:**
- Backend (Python 3.11, 3.12): Passing
- Frontend: Passing
- E2E: Passing

**Coverage Gates:**
- Line coverage: ≥70% (currently 78.82%) ✅
- Branch coverage: ≥68% (not enforced, informational)

---

## Database Schema

**Migrations:** Up to date (from M09)

**Tables:**
- `traces` (UUID, created_at, trace_version, payload)
- `scores` (UUID, trace_id FK, criteria, score, details, created_at)

**No schema changes planned for M10.**

---

## M10 Acceptance Criteria (Pre-Flight Checklist)

Before starting M10 implementation:

- ✅ All 127 tests passing
- ✅ Coverage ≥ baseline (78.82%)
- ✅ No failing CI jobs
- ✅ Known issues documented (deprecations, perf, format validation)
- ✅ Commit SHA captured for rollback
- ✅ Branch strategy defined (`m10-refactor`)

---

## Expected M10 Outcomes

**After M10 completion, we expect:**

1. **app.py shrinks** from 864 lines to ~500-600 lines
2. **New services/ directory** with:
   - `services/traces_batch.py` (batch import logic)
   - `services/datasets_export.py` (export formatting logic)
3. **Coverage improvement** (organically, no hero tests):
   - app.py: 52% → 70%+
   - Overall: 78.82% → 80%+
4. **Zero deprecation warnings** in test output
5. **Batch endpoint perf** improved (~5-10x for large batches)
6. **New documentation:**
   - `docs/M10_GUARDRAILS.md`
   - `docs/M10_SUMMARY.md`
   - Updated `tunix-rt.md`

---

## Rollback Plan

If M10 encounters blockers:

1. Revert to baseline commit: `64ae1d934092356d2e33e92fb646ab2eaf9501d8`
2. Delete `m10-refactor` branch
3. Document blocking issue in M10 questions/answers
4. Re-plan with reduced scope

**Baseline is reproducible:** All tests, coverage, and commit state documented.

---

**Baseline Established:** December 21, 2025  
**Ready for M10 Implementation:** ✅  
**Next Step:** Create `m10-refactor` branch and begin Phase 1

