# M08 Baseline Document

**Date:** December 21, 2025  
**Purpose:** Establish M07 stability baseline before beginning M08 work  
**Commit:** `89bcddf69f80d2243c304409a8b90d867955792d` - docs(M07): Add milestone closeout document

---

## Test Counts (Pre-M08)

### Backend (pytest)
- **Total Tests:** 72
- **Status:** All passing
- **Test Files:**
  - `test_health.py` - 2 tests
  - `test_helpers.py` - 3 tests
  - `test_redi_health.py` - 15 tests
  - `test_scoring.py` - 12 tests
  - `test_settings.py` - 7 tests
  - `test_traces.py` - 17 tests
  - `test_ungar.py` - 11 tests (6 optional UNGAR tests skipped in default CI)
  - `test_ungar_availability.py` - 5 tests

### Frontend (Vitest)
- **Total Tests:** 11
- **Status:** All passing
- **Test Files:**
  - `App.test.tsx` - 11 tests

### E2E (Playwright)
- **Total Tests:** 6
- **Status:** All passing
- **Test Files:**
  - `smoke.spec.ts`:
    - Smoke Tests (4 tests)
    - Trace Upload and Retrieval (1 test)
    - Trace Comparison and Evaluation (1 test)

---

## Coverage Numbers (Pre-M08)

### Backend Coverage (Core-only measurement)
- **Statements:** 363
- **Covered:** 305 (84.04%)
- **Branches:** 54 measured
- **Missing:** 38 branches
- **Coverage Gate:** ≥70% (passing)
- **Config:** `.coveragerc` with omit patterns for optional UNGAR generator

### Coverage Strategy
- **Core Coverage:** Measures all code except truly-optional modules (UNGAR generator)
- **Omit Patterns:** `tunix_rt_backend/integrations/ungar/high_card_duel.py`
- **Full Coverage:** Separate `.coveragerc.full` config for optional UNGAR workflow

---

## CI Status (Pre-M08)

### Main CI Workflow (`.github/workflows/ci.yml`)
- **Backend (Python 3.11):** ✅ Passing
- **Backend (Python 3.12):** ✅ Passing
- **Frontend:** ✅ Passing
- **E2E:** ✅ Passing

### Optional UNGAR Workflow (`.github/workflows/ungar-integration.yml`)
- **Status:** Non-blocking (manual dispatch + nightly)
- **UNGAR Tests:** 6 tests (requires `backend[ungar]` installed)

---

## Database Schema (Pre-M08)

### Tables
1. **`traces`**
   - Columns: `id` (UUID), `created_at` (timestamptz), `trace_version` (varchar), `payload` (JSON)
   - Indexes: Primary key on `id`, index on `created_at`
   - Relationships: One-to-many with `scores`

2. **`scores`**
   - Columns: `id` (UUID), `trace_id` (UUID FK), `criteria` (varchar), `score` (float), `details` (JSON), `created_at` (timestamptz)
   - Indexes: Primary key on `id`, index on `trace_id`, index on `criteria`
   - Relationships: Many-to-one with `traces` (CASCADE delete)

### Migrations
- **Current Head:** `f3cc010ca8a6` (add_scores_table)
- **Total Migrations:** 3
  - `001_create_traces_table.py` (grandfathered ID)
  - `f8f1393630e4_add_traces_created_at_index.py`
  - `f3cc010ca8a6_add_scores_table.py`

---

## API Endpoints (Pre-M08)

### Core Endpoints
- `GET /api/health` - Application health
- `GET /api/redi/health` - RediAI integration health

### Trace Endpoints
- `POST /api/traces` - Create trace
- `GET /api/traces/{id}` - Get trace by ID
- `GET /api/traces` - List traces (paginated)

### Scoring Endpoints
- `POST /api/traces/{id}/score` - Score a trace
- `GET /api/traces/compare` - Compare two traces

### UNGAR Endpoints (Optional)
- `GET /api/ungar/status` - Check UNGAR availability
- `POST /api/ungar/high-card-duel/generate` - Generate traces
- `GET /api/ungar/high-card-duel/export.jsonl` - Export JSONL

---

## Dependencies (Pre-M08)

### Backend Production Dependencies
- FastAPI ≥0.104.0
- Uvicorn ≥0.24.0
- httpx ≥0.25.0
- Pydantic ≥2.4.0
- SQLAlchemy[asyncio] ≥2.0.0
- Alembic ≥1.12.0
- asyncpg ≥0.29.0

### Backend Optional Dependencies
- **`[dev]`:** pytest, pytest-cov, pytest-asyncio, ruff, mypy, aiosqlite
- **`[ungar]`:** ungar @ git+...@0e29e104aa1b13542b193515e3895ee87122c1cb

### Frontend Dependencies
- React 18
- TypeScript 5
- Vite 5
- Vitest 1

---

## Known Issues & Technical Debt (Pre-M08)

### From M07 Audit
1. **Type ignore comments lack explanation** (low priority)
   - Files: `availability.py`, `high_card_duel.py`
   - Fix: Add inline comments explaining why type checking is disabled

2. **No E2E test for UNGAR panel** (low priority)
   - UNGAR frontend panel covered in unit tests but not E2E
   - Fix: Add Playwright smoke test for UNGAR section

3. **Quick start missing from UNGAR docs** (low priority)
   - Documentation comprehensive but lacks copy/paste walkthrough
   - Fix: Add "Quick Start Happy Path" section

4. **Defensive fallbacks lack logging** (low priority)
   - `_format_card()` and similar functions return `"??"` silently
   - Fix: Add `logger.warning()` before fallback returns

5. **Python-level JSON filtering in export** (performance, future concern)
   - Export fetches 10× limit then filters in Python
   - Acceptable at current scale; document optimization trigger

---

## Exit Criteria (M08 Baseline Verification)

✅ **All tests passing:** 72 backend + 11 frontend + 6 E2E = 89 total  
✅ **Coverage maintained:** 84% ≥ 70% gate  
✅ **CI green:** Main workflow passing on all jobs  
✅ **No regressions:** All M07 functionality intact  
✅ **Documentation complete:** This baseline document committed

---

## Next Steps

M08 will proceed with:
1. **Phase 0:** Verify this baseline (✅ complete with this document)
2. **Phase 1:** Close M07 "paper cuts" + add UNGAR E2E test
3. **Phase 2:** Dataset manifests + build/export endpoints
4. **Phase 3:** Tunix SFT prompt renderer
5. **Phase 4:** Training smoke harness (optional dependency)
6. **Phase 5:** Final E2E coverage for datasets

---

**Baseline Established:** ✅ M07 is stable and ready for M08 work  
**Baseline Commit:** `89bcddf69f80d2243c304409a8b90d867955792d`
