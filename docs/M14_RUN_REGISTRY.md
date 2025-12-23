# M14: Tunix Run Registry Guide

**Date:** December 22, 2025  
**Milestone:** M14 (Tunix Run Registry - Phase 3)  
**Status:** ✅ Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Database Schema](#database-schema)
4. [Migration Guide](#migration-guide)
5. [Service Layer Changes](#service-layer-changes)
6. [API Reference](#api-reference)
7. [Frontend Integration](#frontend-integration)
8. [Testing Strategy](#testing-strategy)
9. [Error Handling](#error-handling)
10. [Performance Considerations](#performance-considerations)
11. [Future Enhancements](#future-enhancements)

---

## Executive Summary

M14 introduces **persistent storage and a run registry API** for Tunix training runs. This milestone adds database persistence and query capabilities while maintaining execution semantics from M13.

### Key Capabilities

1. **Persistent Storage**: All Tunix runs (dry-run and local) are saved to PostgreSQL
2. **Run Registry API**: Query, filter, and paginate historical runs
3. **Run Details**: View complete execution metadata including stdout/stderr
4. **Frontend History**: Collapsible panel with manual refresh
5. **Graceful Degradation**: DB failures don't break execution

### Design Principles

- **No execution changes**: M13 execution logic remains identical
- **Immediate persistence**: Run records created before execution starts
- **Graceful DB failures**: Log errors, don't fail user requests
- **No new dependencies**: Uses existing PostgreSQL + Alembic
- **Default CI passes**: All tests use dry-run mode (no Tunix required)

### Out of Scope for M14

- Run deletion/retry/cancellation - deferred
- Streaming logs - deferred
- Auto-refresh or websockets - deferred
- Run metadata mutation (status changes) - deferred
- Background job processing - deferred
- Advanced analytics/metrics - deferred

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Run History Panel (Collapsible)                       │  │
│  │  - List runs (paginated)                              │  │
│  │  - Filter by status/dataset/mode                      │  │
│  │  - View details (expandable rows)                     │  │
│  │  - Manual refresh button                              │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ GET /api/tunix/runs                                   │  │
│  │   - Pagination (limit, offset)                        │  │
│  │   - Filtering (status, dataset_key, mode)            │  │
│  │   - Returns TunixRunListResponse                      │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │ GET /api/tunix/runs/{run_id}                         │  │
│  │   - Returns full TunixRunResponse (with logs)        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ TunixExecutionService (Modified)                      │  │
│  │  1. Create run record (status="running")              │  │
│  │  2. Execute training (M13 logic unchanged)            │  │
│  │  3. Update run record with results                    │  │
│  │  4. Graceful DB failure handling (log, don't fail)    │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │ SQLAlchemy ORM
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  PostgreSQL Database                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ tunix_runs table                                      │  │
│  │  - run_id (UUID, PK)                                  │  │
│  │  - dataset_key (indexed)                              │  │
│  │  - model_id, mode, status                             │  │
│  │  - exit_code, started_at (indexed), completed_at     │  │
│  │  - duration_seconds                                    │  │
│  │  - stdout, stderr (truncated 10KB each)              │  │
│  │  - created_at                                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### `tunix_runs` Table

```sql
CREATE TABLE tunix_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_key VARCHAR(256) NOT NULL,
    model_id VARCHAR(256) NOT NULL,
    mode VARCHAR(64) NOT NULL,  -- 'dry-run' | 'local'
    status VARCHAR(64) NOT NULL,  -- 'pending' | 'running' | 'completed' | 'failed' | 'timeout'
    exit_code INTEGER,  -- NULL for dry-run or timeout
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,  -- NULL only if crash
    duration_seconds FLOAT,  -- Calculated if completed_at exists
    stdout TEXT NOT NULL DEFAULT '',  -- Truncated to 10KB
    stderr TEXT NOT NULL DEFAULT '',  -- Truncated to 10KB
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX ix_tunix_runs_dataset_key ON tunix_runs(dataset_key);
CREATE INDEX ix_tunix_runs_started_at ON tunix_runs(started_at);
```

### Field Descriptions

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `run_id` | UUID | NO | Unique identifier (primary key) |
| `dataset_key` | VARCHAR(256) | NO | Dataset identifier (indexed for filtering) |
| `model_id` | VARCHAR(256) | NO | Hugging Face model ID |
| `mode` | VARCHAR(64) | NO | Execution mode: `dry-run` or `local` |
| `status` | VARCHAR(64) | NO | Run status (see state machine below) |
| `exit_code` | INTEGER | YES | Process exit code (NULL for dry-run/timeout) |
| `started_at` | TIMESTAMPTZ | NO | Execution start time (UTC, indexed) |
| `completed_at` | TIMESTAMPTZ | YES | Execution completion time (UTC) |
| `duration_seconds` | FLOAT | YES | Computed duration (always set if completed) |
| `stdout` | TEXT | NO | Standard output (truncated to 10KB) |
| `stderr` | TEXT | NO | Standard error (truncated to 10KB) |
| `created_at` | TIMESTAMPTZ | NO | Record creation time (UTC) |

### Status State Machine

```
pending ──► running ──► completed
                │
                ├──────► failed
                │
                └──────► timeout
```

**Note:** In M14, runs are created with `status="running"` directly. The `pending` state is reserved for future background job processing.

---

## Migration Guide

### Creating the Migration

The migration was generated with:

```bash
cd backend
alembic revision -m "add_tunix_runs_table"
```

### Migration File Structure

```python
# backend/alembic/versions/4bf76cdb97da_add_tunix_runs_table.py

"""add_tunix_runs_table

Revision ID: 4bf76cdb97da
Revises: f3cc010ca8a6
Create Date: 2025-12-22 14:30:00.000000
"""

def upgrade() -> None:
    """Create tunix_runs table with indexes."""
    op.create_table(
        "tunix_runs",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("dataset_key", sa.String(length=256), nullable=False),
        sa.Column("model_id", sa.String(length=256), nullable=False),
        sa.Column("mode", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("stdout", sa.Text(), nullable=False),
        sa.Column("stderr", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_tunix_runs_dataset_key", "tunix_runs", ["dataset_key"])
    op.create_index("ix_tunix_runs_started_at", "tunix_runs", ["started_at"])

def downgrade() -> None:
    """Drop tunix_runs table and indexes."""
    op.drop_index("ix_tunix_runs_started_at", table_name="tunix_runs")
    op.drop_index("ix_tunix_runs_dataset_key", table_name="tunix_runs")
    op.drop_table("tunix_runs")
```

### Running Migrations

```bash
# Apply migration
cd backend
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Check current version
alembic current

# View migration history
alembic history
```

---

## Service Layer Changes

### TunixExecutionService Modifications

The `execute_tunix_run` function was modified to add persistence:

```python
async def execute_tunix_run(request: TunixRunRequest, db: AsyncSession) -> TunixRunResponse:
    """
    Execute a Tunix training run with persistent storage.
    
    Persistence Flow:
    1. Create TunixRun record with status="running" (immediately)
    2. Execute training (M13 logic unchanged)
    3. Update TunixRun record with results (status, logs, exit code)
    4. If DB write fails: log error, return execution result anyway
    
    Args:
        request: TunixRunRequest with dataset_key, model_id, hyperparameters
        db: AsyncSession for dataset validation and run persistence
    
    Returns:
        TunixRunResponse with run metadata (persisted to DB)
    """
    # Step 1: Create run record immediately
    run = TunixRun(
        dataset_key=request.dataset_key,
        model_id=request.model_id,
        mode="dry-run" if request.dry_run else "local",
        status="running",
        started_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    
    try:
        db.add(run)
        await db.commit()
        await db.refresh(run)
    except Exception as e:
        logger.error(f"Failed to create run record: {e}")
        await db.rollback()
        # Continue execution even if DB write fails
    
    # Step 2: Execute training (M13 logic)
    if request.dry_run:
        result = await _execute_dry_run(request, db)
    else:
        result = await _execute_local(request, db)
    
    # Step 3: Update run record with results
    try:
        run.status = result.status
        run.exit_code = result.exit_code
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = result.duration_seconds
        run.stdout = result.stdout[:10240]  # Truncate to 10KB
        run.stderr = result.stderr[:10240]  # Truncate to 10KB
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to update run record: {e}")
        await db.rollback()
        # Return execution result even if DB write fails
    
    return result
```

### Graceful DB Failure Handling

**Philosophy:** Database persistence is **secondary** to execution. If a DB write fails:

1. Log the error (with full context)
2. Continue execution normally
3. Return the execution result to the user
4. Do NOT raise an exception or fail the request

**Why?** User-facing operations should not be blocked by infrastructure failures. The primary value is executing the training run and returning results.

---

## API Reference

### List Tunix Runs

**Endpoint:** `GET /api/tunix/runs`

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | 20 | Number of runs to return (max: 100) |
| `offset` | int | No | 0 | Number of runs to skip |
| `status` | string | No | - | Filter by status (`completed`, `failed`, `running`, etc.) |
| `dataset_key` | string | No | - | Filter by dataset key |
| `mode` | string | No | - | Filter by mode (`dry-run`, `local`) |

**Example Request:**

```bash
# List all runs (first page)
curl http://localhost:8000/api/tunix/runs

# List completed runs for specific dataset
curl "http://localhost:8000/api/tunix/runs?status=completed&dataset_key=test-v1"

# Paginate with larger page size
curl "http://localhost:8000/api/tunix/runs?limit=50&offset=50"
```

**Response:** `200 OK`

```json
{
  "data": [
    {
      "run_id": "123e4567-e89b-12d3-a456-426614174000",
      "dataset_key": "test-v1",
      "model_id": "google/gemma-2b-it",
      "mode": "dry-run",
      "status": "completed",
      "started_at": "2025-12-22T14:30:00Z",
      "duration_seconds": 5.2
    },
    {
      "run_id": "223e4567-e89b-12d3-a456-426614174001",
      "dataset_key": "test-v2",
      "model_id": "meta-llama/Llama-2-7b",
      "mode": "local",
      "status": "failed",
      "started_at": "2025-12-22T15:00:00Z",
      "duration_seconds": 120.5
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "next_offset": null
  }
}
```

**Error Responses:**

- `422 Unprocessable Entity`: Invalid pagination parameters (limit > 100, negative offset)

---

### Get Run Details

**Endpoint:** `GET /api/tunix/runs/{run_id}`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run_id` | UUID | Yes | Run identifier |

**Example Request:**

```bash
curl http://localhost:8000/api/tunix/runs/123e4567-e89b-12d3-a456-426614174000
```

**Response:** `200 OK`

```json
{
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "mode": "dry-run",
  "dataset_key": "test-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./datasets/test-v1",
  "exit_code": 0,
  "stdout": "Dry-run validation successful\nDataset: test-v1 (100 examples)\nModel: google/gemma-2b-it\n",
  "stderr": "",
  "duration_seconds": 5.2,
  "started_at": "2025-12-22T14:30:00Z",
  "completed_at": "2025-12-22T14:30:05Z",
  "message": "Dry-run completed successfully"
}
```

**Error Responses:**

- `404 Not Found`: Run ID does not exist
- `422 Unprocessable Entity`: Invalid UUID format

---

## Frontend Integration

### Run History Panel

The frontend adds a collapsible "Run History (M14)" section under the existing Tunix panel:

```tsx
// App.tsx (simplified)

const [showRunHistory, setShowRunHistory] = useState(false);
const [tunixRuns, setTunixRuns] = useState<TunixRunListItem[]>([]);
const [selectedRunDetail, setSelectedRunDetail] = useState<TunixRunResponse | null>(null);

const handleFetchTunixRuns = async () => {
  try {
    const response = await listTunixRuns();
    setTunixRuns(response.data);
  } catch (error) {
    setTunixRunsError(`Failed to fetch runs: ${error}`);
  }
};

const handleViewRunDetail = async (runId: string) => {
  if (selectedRunDetail?.run_id === runId) {
    setSelectedRunDetail(null); // Toggle off
  } else {
    const detail = await getTunixRunDetail(runId);
    setSelectedRunDetail(detail);
  }
};
```

### UI Components

**Collapsible Section:**
- Toggle button: `▶ Run History (M14)` / `▼ Run History (M14)`
- Manual refresh button
- Fetch runs on expand (first time only)

**Run List Table:**

| Column | Description |
|--------|-------------|
| Run ID | First 8 chars of UUID |
| Dataset | `dataset_key` |
| Model | `model_id` (truncated) |
| Mode | `dry-run` or `local` |
| Status | Badge with color coding |
| Started At | ISO timestamp |
| Duration | Seconds (1 decimal place) |
| Actions | "View" button to expand details |

**Detail View (Expandable Row):**
- Status, Exit Code, Duration
- Stdout (monospace, preformatted)
- Stderr (monospace, preformatted)
- Timestamps (started_at, completed_at)

---

## Testing Strategy

### Backend Tests

All M14 tests use **dry-run mode** (no Tunix dependency):

```python
# tests/test_tunix_registry.py

def test_run_persists_to_database(client, test_db, test_dataset):
    """Verify a successful run is saved to the database."""
    response = client.post(
        "/api/tunix/run",
        json={
            "dataset_key": test_dataset,
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )
    assert response.status_code == 202
    
    # Verify database persistence
    run_id = response.json()["run_id"]
    detail_response = client.get(f"/api/tunix/runs/{run_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "completed"

def test_list_runs_with_runs(client, test_db, test_dataset):
    """Verify list endpoint returns multiple runs."""
    # Create 3 runs
    for i in range(3):
        client.post("/api/tunix/run", json={
            "dataset_key": test_dataset,
            "model_id": f"model-{i}",
            "dry_run": True,
        })
    
    # List runs
    response = client.get("/api/tunix/runs")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 3

def test_list_runs_filter_by_status(client, test_db, test_dataset):
    """Verify filtering by status works correctly."""
    # (Test implementation with multiple runs of different statuses)
```

**Coverage:**
- Run persistence (create, update)
- List endpoint (pagination, filtering)
- Detail endpoint (by UUID)
- Error cases (404, 422, invalid params)

### Frontend Tests

```typescript
// App.test.tsx

it('displays Run History section collapsed by default', async () => {
  // Verify toggle button shows "▶ Run History"
  // Verify content is not visible
});

it('expands Run History and fetches runs when clicked', async () => {
  // Mock listTunixRuns response
  // Click toggle button
  // Verify content is visible
  // Verify runs are displayed in table
});

it('displays run details when View button is clicked', async () => {
  // Mock run list and detail responses
  // Click View button
  // Verify stdout/stderr are displayed
  // Click again to hide
});
```

---

## Error Handling

### Database Failures

**Scenario:** PostgreSQL connection fails or write operation times out.

**Behavior:**
1. Log error with full context (`logger.error(...)`)
2. Continue execution normally
3. Return result to user
4. Do NOT raise exception

**Example:**

```python
try:
    db.add(run)
    await db.commit()
except Exception as e:
    logger.error(f"Failed to create run record: {e}")
    await db.rollback()
    # Execution continues
```

### API Error Codes

| Status Code | Scenario | Response |
|-------------|----------|----------|
| 200 OK | List/detail success | JSON with data |
| 404 Not Found | Run ID not found | `{"detail": "Run not found"}` |
| 422 Unprocessable Entity | Invalid UUID or params | Pydantic validation error |
| 500 Internal Server Error | Unexpected error | Generic error message |

---

## Performance Considerations

### Indexing Strategy

Two indexes optimize query performance:

1. **`ix_tunix_runs_dataset_key`**: Speeds up filtering by dataset
2. **`ix_tunix_runs_started_at`**: Speeds up time-based sorting/filtering

### Stdout/Stderr Truncation

**Limit:** 10KB (10,240 bytes) per field

**Rationale:**
- Typical training logs are < 5KB
- Prevents DB bloat from verbose outputs
- Frontend can display without lag

**Implementation:**

```python
run.stdout = result.stdout[:10240]  # Truncate to 10KB
run.stderr = result.stderr[:10240]
```

### Pagination Limits

- **Default:** 20 runs per page
- **Maximum:** 100 runs per page (enforced with validation)

**Why?** Prevent slow queries and large response payloads.

---

## Future Enhancements

### M15+ Candidates (Out of Scope for M14)

1. **Run Deletion:** `DELETE /api/tunix/runs/{run_id}`
2. **Run Retry:** `POST /api/tunix/runs/{run_id}/retry`
3. **Status Mutations:** Manual status updates (e.g., mark as "cancelled")
4. **Streaming Logs:** Websocket for real-time log streaming
5. **Auto-Refresh:** Frontend polling or SSE for live updates
6. **Advanced Filtering:** Date ranges, model ID filters, full-text search
7. **Run Metadata:** Add `tags`, `notes`, `user_id` fields
8. **Checkpoints:** Parse and store training checkpoints
9. **Metrics:** Extract loss/accuracy from logs
10. **Background Jobs:** Async execution with Celery/RQ

---

## Complete Example Workflow

### 1. Execute a Tunix Run (Dry-Run)

```bash
curl -X POST http://localhost:8000/api/tunix/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "test-v1",
    "model_id": "google/gemma-2b-it",
    "dry_run": true,
    "hyperparameters": {
      "learning_rate": 2e-5,
      "num_epochs": 3,
      "batch_size": 8
    }
  }'
```

**Response:**

```json
{
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "mode": "dry-run",
  "dataset_key": "test-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./datasets/test-v1",
  "exit_code": 0,
  "stdout": "Dry-run validation successful\n",
  "stderr": "",
  "duration_seconds": 5.2,
  "started_at": "2025-12-22T14:30:00Z",
  "completed_at": "2025-12-22T14:30:05Z",
  "message": "Dry-run completed successfully"
}
```

### 2. List All Runs

```bash
curl http://localhost:8000/api/tunix/runs
```

### 3. Filter Runs by Dataset

```bash
curl "http://localhost:8000/api/tunix/runs?dataset_key=test-v1"
```

### 4. Get Run Details

```bash
curl http://localhost:8000/api/tunix/runs/123e4567-e89b-12d3-a456-426614174000
```

### 5. Frontend: View Run History

1. Click "▶ Run History (M14)" to expand
2. Runs are fetched and displayed in table
3. Click "View" on a run to see stdout/stderr
4. Click "Refresh" to fetch latest runs

---

## Troubleshooting

### Issue: Migration Fails with "relation already exists"

**Solution:** Check if you've already run the migration:

```bash
cd backend
alembic current
# If already at latest, downgrade and re-run:
alembic downgrade -1
alembic upgrade head
```

### Issue: Run List Returns Empty Array

**Cause:** No runs have been executed yet.

**Solution:** Execute a dry-run via UI or API to create a run record.

### Issue: Run Detail Returns 404

**Cause:** Invalid or non-existent `run_id`.

**Solution:** Verify the UUID format and check the list endpoint for valid IDs.

### Issue: Frontend Shows "Failed to fetch runs"

**Cause:** Backend is down or CORS issue.

**Solution:**
1. Check backend logs: `docker compose logs backend`
2. Verify backend is running: `curl http://localhost:8000/api/health`
3. Check browser console for CORS errors

---

## Summary

M14 successfully adds **persistent run storage and a registry API** to tunix-rt:

✅ **Database Schema:** `tunix_runs` table with UUID primary key  
✅ **Alembic Migration:** Reversible schema changes  
✅ **Service Layer:** Immediate persistence with graceful DB failures  
✅ **API Endpoints:** List (paginated, filterable) and detail (full logs)  
✅ **Frontend UI:** Collapsible Run History panel with manual refresh  
✅ **Testing:** 12 new backend tests + 7 frontend tests (dry-run only)  
✅ **Documentation:** Comprehensive guide and baseline docs  

**Next Steps:** M15 candidates include run deletion, retry, streaming logs, and advanced analytics.
