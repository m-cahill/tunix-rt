You are implementing **M14** for the tunix-rt project.

Context:
- M13 is complete and approved.
- Tunix execution (dry-run + local) works.
- Runs are currently ephemeral and returned only in the HTTP response.
- The audit explicitly calls out result persistence as the next milestone.

Goal of M14:
Introduce **persistent Tunix run storage** and a **run registry API**, without changing execution semantics.

---

## Scope (M14)

### 1. Database Schema

Add a new table: `tunix_runs`

Required fields:
- run_id (UUID, primary key)
- dataset_key (string, indexed)
- model_id (string)
- mode (`dry_run` | `local`)
- status (`pending` | `running` | `completed` | `failed` | `timeout`)
- exit_code (nullable int)
- started_at (datetime, UTC)
- completed_at (nullable datetime)
- duration_seconds (nullable float)
- stdout (text, truncated)
- stderr (text, truncated)
- created_at (datetime, UTC)

Requirements:
- Alembic migration with downgrade
- No breaking changes to existing tables

---

### 2. Service Layer Changes

In `TunixExecutionService`:
- Create a run record immediately when execution starts
- Update status + outputs on completion or failure
- Ensure failures (timeouts, subprocess errors) are persisted

Do NOT:
- Introduce async/background execution yet
- Change subprocess behavior
- Change response schema

---

### 3. API Endpoints

Add:
- `GET /api/tunix/runs`
  - Paginated
  - Filterable by:
    - status
    - dataset_key
    - mode
- `GET /api/tunix/runs/{run_id}`
  - Returns full run details

Requirements:
- Proper Pydantic response models
- 404 if run_id not found
- No authentication changes

---

### 4. Frontend

Add a minimal **Run History** panel:
- List recent runs
- Status badge (pending/running/completed/failed)
- Click to view stdout/stderr
- No live updates yet (polling is OK or static fetch)

UX constraints:
- Keep UI simple
- No charts
- No streaming

---

### 5. Tests

Backend:
- Test run record creation
- Test run status updates
- Test list + detail endpoints
- No Tunix runtime required

Frontend:
- Test rendering of run history
- Test error handling (404, empty list)

CI:
- No new required workflows
- Existing CI must remain green

---

### 6. Documentation

Add:
- `docs/M14_RUN_REGISTRY.md`
- Update README with:
  - “Run Persistence (M14)” section
- Add `M14_BASELINE.md` and `M14_SUMMARY.md`

---

## Explicit Non-Goals (Do Not Implement)

- Background workers
- Streaming logs
- Async subprocess execution
- Comparison or evaluation logic
- Run deletion UI

---

## Acceptance Criteria

- Runs persist across server restarts
- Run history visible in UI
- CI fully green
- Coverage does not regress below gates
- Clear upgrade path to async execution in M15

Proceed step-by-step. Keep changes isolated and auditable.
