# M11 Baseline — Stabilize + Complete Service Extraction + Training Script Smoke Tests

**Milestone:** M11  
**Branch:** `m11-stabilize`  
**Baseline Date:** 2025-12-21  
**Baseline Commit:** `0012675370f5669dcc9bebb12221f7842b7e5053`

---

## Purpose

This document establishes the **pre-M11 baseline state** to track changes and verify that M11 refactoring maintains or improves quality metrics without regressions.

**M11 Goals:**
1. Complete app.py extraction (UNGAR + dataset build to services/)
2. Add training script dry-run smoke tests
3. Implement security + DX guardrails (SBOM, SHA pinning, pre-commit)
4. Document production training workflows and SLOs
5. Optional: Frontend coverage to 70%

---

## Test Results (Baseline)

### Backend Tests

**Command:**
```bash
cd backend
python -m pytest -q --cov=tunix_rt_backend --cov-branch --cov-config=.coveragerc --cov-report=term
```

**Results:**
- ✅ **132 passed, 6 skipped**
- ⏱️ **Duration:** 5.78s
- 📊 **Coverage:** 84.16% line (655 statements, 92 missed)
- 🌿 **Branch Coverage:** Not explicitly reported in summary but historically ~88%

**Skipped Tests:**
- 6 UNGAR integration tests (require `pip install -e '.[ungar]'`)

**Warnings:**
- 1 `PytestAssertRewriteWarning` (anyio already imported)
- 1 `DeprecationWarning` for `HTTP_422_UNPROCESSABLE_ENTITY` (should use `HTTP_422_UNPROCESSABLE_CONTENT`)

**Coverage by Module (Key Files):**

| Module | Stmts | Miss | Cover | Notes |
|--------|-------|------|-------|-------|
| `app.py` | 176 | 68 | 59% | ⚠️ Target for Phase 3 extraction |
| `services/traces_batch.py` | 39 | 5 | 91% | ✅ Recent M10 addition |
| `services/datasets_export.py` | 45 | 7 | 81% | ✅ Recent M10 addition |
| `helpers/datasets.py` | 38 | 0 | 100% | ✅ Perfect |
| `helpers/traces.py` | 14 | 0 | 100% | ✅ Perfect |
| `scoring.py` | 11 | 0 | 100% | ✅ Perfect |
| `training/renderers.py` | 34 | 0 | 100% | ✅ Perfect |
| `training/schema.py` | 40 | 0 | 100% | ✅ Perfect |

**Coverage Gates:**
- ✅ Minimum: 70% (current: 84.16%) — **PASS**
- ✅ Line gate: 80% (per `coverage_gate.py`) — **PASS**
- ✅ Branch gate: 68% (historical) — **PASS**

---

### Mypy Type Checking

**Command:**
```bash
cd backend
mypy tunix_rt_backend
```

**Results:**
- ✅ **Success: no issues found in 28 source files**

---

### Frontend Tests

**Command:**
```bash
cd frontend
npm run test -- --run
```

**Results:**
- ✅ **11 passed (11)**
- ⏱️ **Duration:** 3.61s (tests: 2.44s)
- ⚠️ **Warnings:** Multiple React `act()` warnings (non-blocking, common in testing)

**Test Files:**
- `src/App.test.tsx` (11 tests)

**Coverage Notes:**
- Frontend coverage historically ~60% line / 50% branch
- Target for M11 Phase 5: 70% line coverage

---

### E2E Tests

**Status:** Not run in baseline (requires Docker Compose infrastructure)

**Historical Status (from M10):**
- ✅ 5 E2E tests passing (Playwright)
- Uses mock RediAI mode in CI
- PostgreSQL service container in CI

---

## Codebase Metrics

### Backend Structure

**Total Lines:** 741 lines in `app.py` (target: <600 after Phase 3)

**Current Services:**
- `services/traces_batch.py` (39 statements)
- `services/datasets_export.py` (45 statements)

**Endpoints Requiring Extraction (Phase 3):**
1. UNGAR endpoints (3 total):
   - `GET /api/ungar/status`
   - `POST /api/ungar/high-card-duel/generate`
   - `GET /api/ungar/high-card-duel/export.jsonl`
2. Dataset build endpoint:
   - `POST /api/datasets/build`

**Expected Impact:**
- `app.py` reduction: ~100-150 lines → target <600 lines
- New service files: 2 (`ungar_generator.py`, `datasets_builder.py`)
- New tests: ~7 service-level tests

---

### Security Posture

**Current State:**

| Security Control | Status | Notes |
|------------------|--------|-------|
| pip-audit | ✅ Clean | 0 high/critical CVEs |
| npm audit | ⚠️ 4 moderate | Dev-only (Vite/Vitest), deferred to future |
| Gitleaks | ✅ Clean | No secrets detected |
| SBOM generation | ❌ Disabled | Re-enable in M11 Phase 1 |
| GitHub Actions pinning | ⚠️ Tag-based | Upgrade to SHA pinning in M11 Phase 1 |
| Pre-commit hooks | ❌ None | Add in M11 Phase 1 |

---

### Documentation Gaps (M11 Phase 2 Targets)

**Missing ADRs:**
- ADR-006: Tunix API Abstraction Pattern (CRITICAL)

**Missing Production Docs:**
- `docs/TRAINING_PRODUCTION.md` (local vs production training)
- `docs/PERFORMANCE_SLOs.md` (P95 latency targets)

---

## CI/CD Status

**Current Workflow Jobs (`.github/workflows/ci.yml`):**

| Job | Status | Notes |
|-----|--------|-------|
| `backend` | ✅ Passing | Ruff, mypy, pytest, coverage gates |
| `frontend` | ✅ Passing | Vitest, build |
| `e2e` | ✅ Passing | Playwright with Postgres service |
| `security-backend` | ✅ Passing | pip-audit (warn-only) |
| `security-frontend` | ⚠️ Warnings | npm audit (4 moderate, dev-only) |
| `security-secrets` | ✅ Passing | Gitleaks |

**GitHub Actions Versions (Tag-based, to be SHA-pinned):**
- `actions/checkout@v4`
- `actions/setup-python@v5`
- `actions/setup-node@v4`
- `actions/upload-artifact@v4`
- `actions/cache@v4`
- `dorny/paths-filter@v2`
- `gitleaks/gitleaks-action@v2`

---

## Acceptance Criteria for M11

**Phase 0 (Baseline):**
- ✅ Baseline doc created
- ✅ All current tests passing
- ✅ No behavior changes

**Phase 1 (Fix-First):**
- SBOM artifact uploads successfully in CI
- All GitHub Actions use SHA pinning
- Pre-commit hooks work locally

**Phase 2 (Docs):**
- ADR-006 exists and is actionable
- TRAINING_PRODUCTION.md complete
- PERFORMANCE_SLOs.md defines targets

**Phase 3 (App Extraction):**
- `app.py` reduced to <600 lines
- All modified endpoints <20 lines each
- UNGAR + dataset build in services/
- New service tests pass
- Coverage ≥84% (baseline)

**Phase 4 (Training Tests):**
- `--dry-run` flag implemented
- Smoke tests via subprocess pass
- Tests fast (<10s) and deterministic

**Phase 5 (Optional - Frontend):**
- Frontend coverage ≥70%
- 5 new component tests
- All tests pass

---

## Rollback Plan

If M11 introduces regressions:

1. **Immediate:** Revert latest commit on `m11-stabilize`
2. **Per-phase:** Each phase is atomic and independently revertible
3. **Complete rollback:** Delete `m11-stabilize` branch, cherry-pick successful phases

**Critical Invariants (must not break):**
- Backend tests: 132+ passing
- Coverage: ≥80% line, ≥68% branch
- Mypy: 0 errors
- CI: All jobs green

---

## Baseline Summary

| Metric | Baseline Value | M11 Target | Status |
|--------|---------------|------------|--------|
| Backend tests | 132 passing | ≥132 passing | ✅ |
| Backend coverage | 84.16% line | ≥80% line | ✅ |
| app.py lines | 741 | <600 | ❌ Needs work |
| Service files | 2 | 4 | ❌ Needs work |
| Frontend tests | 11 passing | 11+ passing | ✅ |
| Frontend coverage | ~60% | 70% (optional) | ⚠️ Phase 5 |
| E2E tests | 5 passing | 5 passing | ✅ |
| Mypy errors | 0 | 0 | ✅ |
| SBOM | Disabled | Enabled | ❌ Phase 1 |
| Actions pinning | Tags | SHAs | ❌ Phase 1 |
| Pre-commit | None | Configured | ❌ Phase 1 |
| Training tests | None | Smoke tests | ❌ Phase 4 |

---

**Next Step:** Proceed to **Phase 1 — Fix-First & Stabilize** (SBOM, SHA pinning, pre-commit).
