# M03 Clarifying Questions

## Project Familiarization Complete ✅

I've analyzed the tunix-rt codebase and have a good understanding of:
- **Current state**: M2 complete with trace storage/retrieval, PostgreSQL + Alembic, 92% line/90% branch coverage
- **Architecture**: FastAPI backend, Vite+React frontend, Playwright E2E, Docker Compose with PostgreSQL
- **M03 Goals**: Hardening the trace subsystem (DB pool config, created_at index, frontend tests, coverage artifacts, DX improvements)

## Confirmed Issues

1. **Frontend coverage artifacts**: ✅ Confirmed - Running `npm run test -- --coverage` locally does NOT generate a coverage directory, even though vite.config.ts has coverage settings defined
2. **Migration revision ID**: ✅ Confirmed - Current migration uses hardcoded '001'
3. **Trace UI lacks frontend tests**: ✅ Confirmed - Only App.test.tsx exists with health endpoint tests, no trace-specific tests
4. **DB pool settings**: ✅ Confirmed - Settings exist in settings.py but are NOT applied in db/base.py's create_async_engine call

## Clarifying Questions

### 1. Alembic Auto-Generated IDs
For future migrations (deliverable A.3), should I:
- **Option A**: Just document the policy in tunix-rt.md with a note like "Future migrations: use `alembic revision -m "msg"` without manual revision ID override"
- **Option B**: Create a template/snippet showing the correct command
- **Option C**: Add a Makefile validation target that checks migration files for auto-generated UUIDs
- **Preference?** The plan says "no tooling enforcement required yet", so I'm leaning toward Option A.

### 2. Created_at Index Migration Naming
For the new migration (deliverable A.2):
- Should I use the command `alembic revision -m "add created_at index"` (auto-generated UUID)?
- Or should I use `alembic revision --autogenerate -m "add created_at index"` (let Alembic detect the change)?
- **My recommendation**: Manual revision (not autogenerate) since we're only adding an index, not changing the model. Autogenerate might not detect index changes reliably.

### 3. Frontend Coverage Configuration
I confirmed locally that `npm run test -- --coverage` does NOT generate a coverage directory. Looking at vite.config.ts, the coverage settings are there but vitest isn't creating output. Should I:
- **Option A**: Check if vitest needs an explicit `coverage.enabled = true` in the config
- **Option B**: Verify the vite/vitest version compatibility
- **Option C**: Just fix whatever's broken and document the solution
- **Preference?** Option C - I'll diagnose and fix it.

### 4. Frontend Test Scope
For deliverable B.4 (2-3 trace UI tests), should the tests:
- Mock the fetch calls to `/api/traces` endpoints (like existing tests do)?
- Test button states (disabled/enabled)?
- Test error rendering?
- **My plan**: Add 3 tests covering the scenarios in the plan (Load Example, Upload success, Fetch success) with proper mocking, following the existing test patterns.

### 5. README Curl Examples Location
For deliverable C.6, should the curl examples be:
- **Option A**: Added to the existing "API Endpoints" section in README.md with inline examples
- **Option B**: Added as a new "Quick API Examples" section before or after "API Endpoints"
- **Option C**: Added to tunix-rt.md instead of README.md
- **Preference?** Option A - inline examples in the existing API Endpoints section for easy reference.

### 6. DB Troubleshooting Section Location
Should the "DB Troubleshooting" section be:
- **Option A**: Added to README.md near the Docker Compose section
- **Option B**: Added as a subsection under "Troubleshooting" in README.md
- **Option C**: Added to tunix-rt.md
- **Preference?** Option B - extend the existing Troubleshooting section in README.md.

### 7. Coverage Threshold Confirmation
The plan says "do not lower coverage thresholds". Current thresholds are:
- **Backend**: 80% line, 68% branch (enforced via coverage_gate.py)
- **Frontend**: 60% line, 50% branch (in vite.config.ts)
- Confirm these should remain unchanged even if new trace tests change the percentages slightly?

### 8. Testing Strategy for New Migration
For the created_at index migration (deliverable A.2):
- Should I verify the migration works by:
  1. Running `alembic upgrade head` on SQLite (like CI does)
  2. Running on local PostgreSQL via docker compose
  3. Checking the index exists via SQL query
- **My plan**: Do all 3 locally before committing.

## Ready to Proceed?

Once you've answered these questions (or told me to use my best judgment), I'll:
1. Create a comprehensive todo list based on the M03 plan phases
2. Execute each phase systematically
3. Run all verification checks from the plan before marking M03 complete

**Estimated timeline**: This is a small, focused milestone. I expect ~6-8 major deliverables with testing/verification. Should be completable in one session.
