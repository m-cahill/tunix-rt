# Tunix RT - Reasoning-Trace Framework

**Milestone M18 Complete** ✅  
**Coverage:** 82% Backend Line, 77% Frontend Line | **Security:** SHA-Pinned CI, SBOM Enabled, Pre-commit Hooks | **Architecture:** Real Judge + Regression Gates + Pagination | **Tests:** 209 backend + 28 frontend tests

## Overview

Tunix RT is a full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon. The system provides health monitoring, RediAI integration, and a foundation for trace-quality optimization workflows.

**M1 Enhancements:** Enterprise-grade testing (90% branch coverage), security scanning (pip-audit, npm audit, gitleaks), configuration validation, TTL caching, and developer experience tools.

**M2 Enhancements:** Database integration (async SQLAlchemy + PostgreSQL), Alembic migrations, trace CRUD API (create/retrieve/list), frontend trace UI, comprehensive validation, and payload size limits.

**M3 Enhancements:** Trace system hardening - DB connection pool settings applied, created_at index for list performance, frontend trace UI unit tests (8 total), frontend coverage artifact generation confirmed, Alembic auto-ID migration policy documented, curl API examples, and DB troubleshooting guide.

**M14 Enhancements:** Tunix Run Registry - persistent storage with `tunix_runs` table (UUID PK, indexed columns), Alembic migration, immediate run persistence (create with status="running", update on completion), graceful DB failure handling, `GET /api/tunix/runs` with pagination/filtering, `GET /api/tunix/runs/{run_id}` for details, frontend Run History panel (collapsible, manual refresh), stdout/stderr truncation (10KB), 12 new backend tests + 7 frontend tests (all dry-run, no Tunix dependency).

**M15 Enhancements:** Async Execution Engine - `POST /api/tunix/run?mode=async` for non-blocking enqueue, dedicated worker process (`worker.py`) using Postgres `SKIP LOCKED` for robust job claiming, status polling endpoint, frontend "Run Async" toggle with auto-refresh, Prometheus metrics (`/metrics`) for run counts/duration/latency, `config` column migration for deferred execution parameters.

**M16 Enhancements:** Operational UX - Real-time log streaming via SSE (`/api/tunix/runs/{id}/logs`), Run cancellation (`POST /cancel`) with worker termination, Artifacts/Checkpoints management (`/artifacts` list + download), Hardening (pinned dependencies, optimized trace batch insertion), `tunix_run_log_chunks` table for streaming persistence.

**M17 Enhancements:** Evaluation & Quality Loop - `tunix_run_evaluations` table for persisting scores, "Mock Judge" for deterministic evaluation, Auto-trigger on run completion (async/sync), Leaderboard UI (`/leaderboard`), Manual re-evaluation API (`POST /evaluate`), Dry-run exclusion.

**M18 Enhancements:** Judge Abstraction & Regression - `GemmaJudge` implementation (using RediAI), `regression_baselines` table and endpoints (`POST /api/regression/baselines`, `/check`), Leaderboard pagination (API + UI), Pluggable Judge interface.

## System Architecture

### Components

1. **Backend (FastAPI)**
   - Health monitoring endpoints
   - RediAI integration with mock/real modes
   - Trace storage & retrieval (M2)
   - Async SQLAlchemy + PostgreSQL database (M2)
   - Alembic migrations (M2)
   - Dependency injection for testability
   - 80% line / 68% branch coverage gates

2. **Frontend (Vite + React + TypeScript)**
   - Real-time health status display with 30s auto-refresh (M1)
   - RediAI integration monitoring
   - Trace upload and retrieval UI (M2)
   - Typed API client for type-safe backend calls (M1)
   - Responsive UI with status indicators
   - 60% line / 50% branch coverage gates (M2)

3. **E2E Tests (Playwright)**
   - Smoke tests for critical paths
   - Mock RediAI mode for CI
   - Real RediAI mode for local development

4. **Infrastructure**
   - Docker Compose with PostgreSQL
   - GitHub Actions CI with path filtering
   - Automated testing and deployment checks

## API Endpoints

### Health Endpoints

#### `GET /api/health`

**Description:** Check tunix-rt application health

**Response:**
```json
{
  "status": "healthy"
}
```

**Status Codes:**
- `200 OK`: Application is healthy

---

#### `GET /api/redi/health`

**Description:** Check RediAI integration health (with TTL caching)

**Response (Healthy):**
```json
{
  "status": "healthy"
}
```

**Response (Down):**
```json
{
  "status": "down",
  "error": "HTTP 404"
}
```

**Status Codes:**
- `200 OK`: Request succeeded (check `status` field for actual health)

**Behavior:**
- **Mock Mode (`REDIAI_MODE=mock`)**: Always returns `{"status": "healthy"}`
- **Real Mode (`REDIAI_MODE=real`)**: Probes actual RediAI instance at `REDIAI_BASE_URL`

**Caching (M1):**
- Responses cached for 30 seconds (configurable via `REDIAI_HEALTH_CACHE_TTL_SECONDS`)
- Cache hit: <1ms response time
- Cache miss: ~10-50ms (makes HTTP request to RediAI)
- Reduces load on RediAI during UI polling

**Error Details:**
- `HTTP <code>`: Non-2xx response from RediAI
- `Timeout after 5s`: Request timeout
- `Connection refused`: Cannot connect to RediAI instance

---

### Trace Endpoints (M2)

#### `POST /api/traces`

**Description:** Create a new reasoning trace

**Request Body:**
```json
{
  "trace_version": "1.0",
  "prompt": "What is 27 × 19?",
  "final_answer": "513",
  "steps": [
    {"i": 0, "type": "parse", "content": "Parse the multiplication task"},
    {"i": 1, "type": "compute", "content": "Break down: 27 × 19 = 27 × (20 - 1)"},
    {"i": 2, "type": "result", "content": "Final: 513"}
  ],
  "meta": {"source": "example"}
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-20T10:30:00Z",
  "trace_version": "1.0"
}
```

**Status Codes:**
- `201 Created`: Trace created successfully
- `413 Payload Too Large`: Exceeds `TRACE_MAX_BYTES` (default 1MB)
- `422 Unprocessable Entity`: Validation error (invalid schema, duplicate step indices, etc.)

**Validation:**
- `trace_version`: Required, max 64 chars
- `prompt`: Required, 1-50000 chars
- `final_answer`: Required, 1-50000 chars
- `steps`: Required, 1-1000 items, unique indices
- Each step: `i` (non-negative), `type` (1-64 chars), `content` (1-20000 chars)

---

#### `GET /api/traces/{id}`

**Description:** Get a trace by ID

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-20T10:30:00Z",
  "trace_version": "1.0",
  "payload": {
    "trace_version": "1.0",
    "prompt": "What is 27 × 19?",
    "final_answer": "513",
    "steps": [...],
    "meta": {...}
  }
}
```

**Status Codes:**
- `200 OK`: Trace found
- `404 Not Found`: Trace with given ID doesn't exist

---

#### `GET /api/traces?limit=20&offset=0`

**Description:** List traces with pagination

**Query Parameters:**
- `limit` (optional): Max items to return (1-100, default 20)
- `offset` (optional): Pagination offset (default 0)

**Response:**
```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2025-12-20T10:30:00Z",
      "trace_version": "1.0"
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "next_offset": 20
  }
}
```

**Status Codes:**
- `200 OK`: Success
- `422 Unprocessable Entity`: Invalid limit (>100) or offset (<0)

**Note:** List endpoint returns trace metadata only (no full payload). Use `GET /api/traces/{id}` to retrieve full trace.

---

### Scoring Endpoints (M5)

#### `POST /api/traces/{id}/score`

**Description:** Score a trace using specified criteria

**Request Body:**
```json
{
  "criteria": "baseline"
}
```

**Response:**
```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "score": 67.5,
  "details": {
    "step_count": 5,
    "avg_step_length": 342.5,
    "total_chars": 1712,
    "step_score": 25.0,
    "length_score": 34.25,
    "criteria": "baseline",
    "scored_at": "2025-12-21T10:30:00Z"
  }
}
```

**Status Codes:**
- `201 Created`: Score computed and stored successfully
- `404 Not Found`: Trace with given ID doesn't exist

**Scoring Logic (Baseline):**
- **Score range:** 0-100
- **Step score (0-50):** Rewards having 1-10 steps (ideal range)
- **Length score (0-50):** Rewards average step length of 100-500 chars (ideal range)
- **Formula:** `step_score = min(step_count / 10, 1.0) * 50`  
  `length_score = min(avg_step_length / 500, 1.0) * 50`

---

#### `GET /api/traces/compare?base=ID1&other=ID2`

**Description:** Compare two traces side-by-side with scores

**Query Parameters:**
- `base` (required): UUID of the base trace
- `other` (required): UUID of the other trace

**Response:**
```json
{
  "base": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-12-21T10:30:00Z",
    "score": 25.5,
    "trace_version": "1.0",
    "payload": {
      "trace_version": "1.0",
      "prompt": "Simple task",
      "final_answer": "Simple answer",
      "steps": [
        {"i": 0, "type": "think", "content": "Short reasoning"}
      ]
    }
  },
  "other": {
    "id": "660f9500-f39c-52e5-b827-557766551111",
    "created_at": "2025-12-21T10:35:00Z",
    "score": 75.8,
    "trace_version": "1.0",
    "payload": {
      "trace_version": "1.0",
      "prompt": "Complex task",
      "final_answer": "Detailed answer",
      "steps": [
        {"i": 0, "type": "analyze", "content": "Deep analysis"},
        {"i": 1, "type": "compute", "content": "Complex computation"},
        {"i": 2, "type": "verify", "content": "Verification step"}
      ]
    }
  }
}
```

**Status Codes:**
- `200 OK`: Comparison successful
- `404 Not Found`: One or both traces don't exist

**Note:** Scores are computed on-the-fly using the baseline scorer for each comparison request.

---

### Tunix Integration Endpoints (M12/M13/M16)

**M12 Design:** Mock-first, artifact-based integration (no Tunix runtime required)  
**M13 Enhancement:** Optional runtime execution with graceful degradation  
**M16 Enhancement:** Live logs via SSE, cancellation, artifacts

#### `GET /api/tunix/status`

**Description:** Check Tunix integration status

**Response (M13 with Tunix installed):**
```json
{
  "available": true,
  "version": "0.1.0",
  "runtime_required": true,
  "message": "Tunix runtime is available for execution."
}
```

**Response (M13 without Tunix):**
```json
{
  "available": false,
  "version": null,
  "runtime_required": false,
  "message": "Tunix runtime not available. Install with `pip install -e '.[tunix]'` for local execution."
}
```

**Status Codes:**
- `200 OK`: Status retrieved successfully

**M13 Note:** `runtime_required` is now `true` if Tunix is available, enabling local execution mode.

---

#### `POST /api/tunix/sft/export`

**Description:** Export traces in Tunix SFT format (JSONL). Reuses the `tunix_sft` export format from M09 (Gemma chat templates with reasoning steps).

**Request Body:**
```json
{
  "dataset_key": "my_dataset-v1",
  "trace_ids": null,
  "limit": 100
}
```

**Parameters:**
- `dataset_key` (optional): Dataset identifier to export
- `trace_ids` (optional): Array of specific trace IDs to export
- `limit` (optional, default 100): Maximum traces to export

**Note:** Either `dataset_key` OR `trace_ids` must be provided.

**Response:** `application/x-ndjson` (JSONL content)

**Status Codes:**
- `200 OK`: Export successful (returns JSONL)
- `400 Bad Request`: Neither dataset_key nor trace_ids provided
- `404 Not Found`: Dataset not found

**Example:**
```bash
curl -X POST http://localhost:8000/api/tunix/sft/export \
  -H "Content-Type: application/json" \
  -d '{"dataset_key": "ungar_hcd-v1"}' > export.jsonl
```

---

#### `POST /api/tunix/sft/manifest`

**Description:** Generate a Tunix SFT training run manifest (YAML config)

**Request Body:**
```json
{
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./output/run_001",
  "learning_rate": 2e-5,
  "num_epochs": 3,
  "batch_size": 8,
  "max_seq_length": 2048
}
```

**Parameters:**
- `dataset_key` (required): Dataset identifier
- `model_id` (required): Model identifier (e.g., "google/gemma-2b-it")
- `output_dir` (required): Output directory for training artifacts
- `learning_rate` (optional, default 2e-5): Learning rate
- `num_epochs` (optional, default 3): Number of epochs
- `batch_size` (optional, default 8): Batch size
- `max_seq_length` (optional, default 2048): Maximum sequence length

**Response:**
```json
{
  "manifest_yaml": "version: \"1.0\"\nrunner: tunix\n...",
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "format": "tunix_sft",
  "message": "Manifest generated. Save as YAML and execute with Tunix CLI."
}
```

**Status Codes:**
- `201 Created`: Manifest generated successfully
- `404 Not Found`: Dataset not found

**Example:**
```bash
curl -X POST http://localhost:8000/api/tunix/sft/manifest \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "ungar_hcd-v1",
    "model_id": "google/gemma-2b-it",
    "output_dir": "./output/run_001"
  }' | jq -r '.manifest_yaml' > config.yaml
```

**See also:** `docs/M12_TUNIX_INTEGRATION.md` for complete documentation.

---

#### `POST /api/tunix/run` (M13)

**Description:** Execute a Tunix training run (dry-run or local mode)

**Request Body:**
```json
{
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./output/run_001",
  "dry_run": true,
  "learning_rate": 2e-5,
  "num_epochs": 3,
  "batch_size": 8,
  "max_seq_length": 2048
}
```

**Parameters:**
- `dataset_key` (required): Dataset identifier
- `model_id` (required): Model identifier
- `output_dir` (optional, auto-generated if not provided): Output directory
- `dry_run` (optional, default `true`): If true, validate only; if false, execute training
- `learning_rate` (optional, default 2e-5): Learning rate
- `num_epochs` (optional, default 3): Number of epochs
- `batch_size` (optional, default 8): Batch size
- `max_seq_length` (optional, default 2048): Maximum sequence length

**Response (202 Accepted):**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "mode": "dry-run",
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./output/run_001",
  "exit_code": 0,
  "stdout": "Dry-run validation passed",
  "stderr": "",
  "duration_seconds": 0.05,
  "started_at": "2025-12-22T10:00:00.000Z",
  "completed_at": "2025-12-22T10:00:00.050Z",
  "message": "Dry-run completed successfully. Configuration is valid."
}
```

**Status Codes:**
- `202 Accepted`: Run initiated/completed successfully
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Dataset not found
- `501 Not Implemented`: Tunix not available and `dry_run=false`

**Execution Modes:**
- **Dry-run mode (`dry_run=true`, default):** Validates configuration without executing
- **Local mode (`dry_run=false`):** Executes training locally (requires Tunix installation)

**Example (Dry-run):**
```bash
curl -X POST http://localhost:8000/api/tunix/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_key": "ungar_hcd-v1", "model_id": "google/gemma-2b-it", "dry_run": true}'
```

**Example (Local execution):**
```bash
# Requires: pip install -e "backend[tunix]"
curl -X POST http://localhost:8000/api/tunix/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_key": "ungar_hcd-v1", "model_id": "google/gemma-2b-it", "dry_run": false, "num_epochs": 1}'
```

**See also:** `docs/M13_TUNIX_EXECUTION.md` for complete execution guide.

---

#### `GET /api/tunix/runs` (M14)

**Description:** List Tunix training runs with pagination and filtering

**Query Parameters:**
- `limit` (optional, default 20, max 100): Number of runs to return
- `offset` (optional, default 0): Number of runs to skip
- `status` (optional): Filter by status (`completed`, `failed`, `running`, `timeout`, `pending`, `cancelled`)
- `dataset_key` (optional): Filter by dataset key
- `mode` (optional): Filter by mode (`dry-run`, `local`)

**Response (200 OK):**
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
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "next_offset": null
  }
}
```

**Status Codes:**
- `200 OK`: Request succeeded
- `422 Unprocessable Entity`: Invalid pagination parameters (e.g., limit > 100)

**Example (List all runs):**
```bash
curl http://localhost:8000/api/tunix/runs
```

**Example (Filter by dataset and status):**
```bash
curl "http://localhost:8000/api/tunix/runs?dataset_key=test-v1&status=completed"
```

**Example (Paginate with larger page size):**
```bash
curl "http://localhost:8000/api/tunix/runs?limit=50&offset=50"
```

---

#### `GET /api/tunix/runs/{run_id}` (M14)

**Description:** Get full details for a specific Tunix run

**Path Parameters:**
- `run_id` (required): Run identifier (UUID)

**Response (200 OK):**
```json
{
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "mode": "dry-run",
  "dataset_key": "test-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./datasets/test-v1",
  "exit_code": 0,
  "stdout": "Dry-run validation successful\nDataset: test-v1 (100 examples)\n",
  "stderr": "",
  "duration_seconds": 5.2,
  "started_at": "2025-12-22T14:30:00Z",
  "completed_at": "2025-12-22T14:30:05Z",
  "message": "Dry-run completed successfully"
}
```

**Status Codes:**
- `200 OK`: Run found
- `404 Not Found`: Run ID does not exist
- `422 Unprocessable Entity`: Invalid UUID format

**Example:**
```bash
curl http://localhost:8000/api/tunix/runs/123e4567-e89b-12d3-a456-426614174000
```

**See also:** `docs/M14_RUN_REGISTRY.md` for complete run registry guide.

---

### M16 Log Streaming & Artifacts

#### `GET /api/tunix/runs/{id}/logs`

**Description:** Stream real-time logs via Server-Sent Events (SSE).

**Parameters:**
- `since_seq` (int): Start streaming from this sequence number (resume).

**Response:** `text/event-stream` with events `log`, `status`, `heartbeat`.

#### `POST /api/tunix/runs/{id}/cancel`

**Description:** Cancel a pending or running job.

#### `GET /api/tunix/runs/{id}/artifacts`

**Description:** List output files (checkpoints, configs).

#### `GET /api/tunix/runs/{id}/artifacts/{filename}/download`

**Description:** Download a specific artifact file.

---

### M17 Evaluation Endpoints (Updated M18)

#### `GET /api/tunix/evaluations`

**Description:** Get leaderboard data (ranked list of evaluated runs).

**Query Parameters (M18):**
- `limit` (default 50): Max items.
- `offset` (default 0): Pagination offset.

**Response:**
```json
{
  "data": [
    {
      "run_id": "...",
      "model_id": "google/gemma-2b-it",
      "score": 85.5,
      "verdict": "pass"
    }
  ],
  "pagination": { "limit": 50, "offset": 0, "next_offset": 50 }
}
```

#### `GET /api/tunix/runs/{id}/evaluation`

**Description:** Get detailed evaluation results for a run.

#### `POST /api/tunix/runs/{id}/evaluate`

**Description:** Manually trigger evaluation for a completed run (skips dry-runs). Supports `judge_override`.

---

### M18 Regression Endpoints

#### `POST /api/regression/baselines`

**Description:** Create or update a regression baseline.

**Request Body:**
```json
{
  "name": "gemma-v1-baseline",
  "run_id": "...",
  "metric": "score"
}
```

#### `POST /api/regression/check`

**Description:** Check run against baseline.

**Request Body:**
```json
{
  "run_id": "...",
  "baseline_name": "gemma-v1-baseline"
}
```

**Response:**
```json
{
  "verdict": "pass",
  "delta": 5.0,
  "delta_percent": 6.2
}
```

---

## Database Schema

### M2 Schema

**Table: `traces`**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY | Trace unique identifier |
| `created_at` | TIMESTAMPTZ | NOT NULL | Creation timestamp (UTC) |
| `trace_version` | VARCHAR(64) | NOT NULL | Trace format version |
| `payload` | JSON/JSONB | NOT NULL | Full trace data (ReasoningTrace) |

**Indexes:**
- Primary key on `id` (automatic)
- Index `ix_traces_created_at` on `created_at` for list pagination performance (M3)

### M5 Schema

**Table: `scores`**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY | Score unique identifier |
| `trace_id` | UUID | FK → traces.id (CASCADE) | Associated trace |
| `criteria` | VARCHAR(64) | NOT NULL | Scoring criteria (e.g., 'baseline') |
| `score` | FLOAT | NOT NULL | Numeric score value (0-100 for baseline) |
| `details` | JSON | NULLABLE | Detailed scoring breakdown |
| `created_at` | TIMESTAMPTZ | NOT NULL | Score creation timestamp (UTC) |

**Indexes:**
- Primary key on `id` (automatic)
- Index `ix_scores_trace_id` on `trace_id` for trace-based lookups
- Index `ix_scores_criteria` on `criteria` for criteria-based filtering

**Relationships:**
- `scores.trace_id` → `traces.id` (CASCADE on delete)

**Migrations:**
- Managed by Alembic (async mode)
- Migration files in `backend/alembic/versions/`
- Run migrations: `make db-upgrade` or `alembic upgrade head`

**Migration Policy (M3):**
- **DO NOT** manually set revision IDs; use Alembic-generated UUIDs
- Create new migrations: `alembic revision -m "description"`
- Alembic will auto-generate a unique revision ID (e.g., `f8f1393630e4`, `f3cc010ca8a6`)
- Existing migration `001` is grandfathered; all future migrations use auto-generated IDs

**M5 Migrations:**
- `f3cc010ca8a6_add_scores_table.py` - Creates scores table with FK to traces

### M14 Schema

**Table: `tunix_runs`**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `run_id` | UUID | PRIMARY KEY | Run unique identifier |
| `dataset_key` | VARCHAR(256) | NOT NULL | Dataset identifier |
| `model_id` | VARCHAR(256) | NOT NULL | Hugging Face model identifier |
| `mode` | VARCHAR(64) | NOT NULL | Execution mode (`dry-run` or `local`) |
| `status` | VARCHAR(64) | NOT NULL | Run status (`pending`, `running`, `completed`, `failed`, `timeout`, `cancelled`) |
| `exit_code` | INTEGER | NULLABLE | Process exit code (NULL for dry-run/timeout) |
| `started_at` | TIMESTAMPTZ | NOT NULL | Execution start time (UTC) |
| `completed_at` | TIMESTAMPTZ | NULLABLE | Execution completion time (UTC, NULL only if crash) |
| `duration_seconds` | FLOAT | NULLABLE | Execution duration (calculated if completed) |
| `stdout` | TEXT | NOT NULL DEFAULT '' | Standard output (truncated to 10KB) |
| `stderr` | TEXT | NOT NULL DEFAULT '' | Standard error (truncated to 10KB) |
| `created_at` | TIMESTAMPTZ | NOT NULL | Record creation time (UTC) |

**Indexes:**
- Primary key on `run_id` (automatic)
- Index `ix_tunix_runs_dataset_key` on `dataset_key` for dataset filtering
- Index `ix_tunix_runs_started_at` on `started_at` for time-based queries

**Status State Machine:**
```
pending ──► running ──► completed
                │
                ├──────► failed
                │
                ├──────► timeout
                │
                └──────► cancelled
```

**Note:** In M14, runs are created with `status="running"` directly. The `pending` state is reserved for future background job processing (M15+).

**M14 Migrations:**
- `4bf76cdb97da_add_tunix_runs_table.py` - Creates tunix_runs table with indexes

### M16 Schema

**Table: `tunix_run_log_chunks`**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `run_id` | UUID | FK → tunix_runs.run_id (CASCADE) | Associated run |
| `seq` | INTEGER | NOT NULL | Monotonic sequence number per run |
| `stream` | VARCHAR(16) | NOT NULL | Stream name ('stdout' or 'stderr') |
| `chunk` | TEXT | NOT NULL | Log text content |
| `created_at` | TIMESTAMPTZ | NOT NULL | Creation timestamp (UTC) |

**Indexes:**
- `ix_tunix_run_log_chunks_run_id_seq` on `(run_id, seq)` for efficient streaming.

### M17 Schema (New)

**Table: `tunix_run_evaluations`**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY | Evaluation unique ID |
| `run_id` | UUID | FK → tunix_runs.run_id | Associated run |
| `score` | FLOAT | NOT NULL | Primary aggregate score (0-100) |
| `verdict` | VARCHAR(32) | NOT NULL | 'pass' or 'fail' |
| `details` | JSON | NOT NULL | Full evaluation metrics and details |
| `created_at` | TIMESTAMPTZ | NOT NULL | Creation timestamp |

**Indexes:**
- `ix_tunix_run_evaluations_run_id`: Fast lookup by run.
- `ix_tunix_run_evaluations_score`: Leaderboard sorting.

### M18 Schema (New)

**Table: `regression_baselines`**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PRIMARY KEY | Baseline unique ID |
| `name` | VARCHAR | UNIQUE, NOT NULL | Baseline name |
| `run_id` | UUID | FK → tunix_runs.run_id | Baseline run reference |
| `metric` | VARCHAR | NOT NULL | Metric to compare (e.g. 'score') |
| `created_at` | TIMESTAMPTZ | NOT NULL | Creation timestamp |

## Configuration

### Environment Variables

| Variable | Default | Description | Validation |
|----------|---------|-------------|------------|
| `BACKEND_PORT` | `8000` | FastAPI server port | Must be 1-65535 |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string | None |
| `DB_POOL_SIZE` | `5` | DB connection pool size (M2) | Must be 1-50 |
| `DB_MAX_OVERFLOW` | `10` | DB pool max overflow (M2) | Must be 0-50 |
| `DB_POOL_TIMEOUT` | `30` | DB pool timeout seconds (M2) | Must be 1-300 |
| `TRACE_MAX_BYTES` | `1048576` | Max trace payload size (1MB) (M2) | Must be 1024-10485760 |
| `FRONTEND_PORT` | `5173` | Vite dev server port | None |
| `REDIAI_MODE` | `mock` | RediAI integration mode | Must be "mock" or "real" |
| `REDIAI_BASE_URL` | `http://localhost:8080` | RediAI instance URL (real mode) | Must be valid HTTP/HTTPS URL |
| `REDIAI_HEALTH_PATH` | `/health` | RediAI health endpoint path | Must start with "/" (M2) |
| `REDIAI_HEALTH_CACHE_TTL_SECONDS` | `30` | Cache TTL for health checks (M1) | Must be 0-300 |

**Configuration Validation (M1):**
- All settings are validated on application startup using Pydantic
- Invalid configuration causes immediate failure with descriptive error messages
- See `backend/tunix_rt_backend/settings.py` for validation logic

**Tuning DB Pool Settings (Production - M3):**
Default settings (pool_size=5, max_overflow=10) support ~50 concurrent requests.
- **Increase pool_size:** For CPU-bound workloads (set to # of cores)
- **Increase max_overflow:** To handle request spikes (2x expected concurrent users)
- **Decrease:** For memory-constrained or low-concurrency environments
- **Monitor:** Watch for "QueuePool limit exceeded" errors in logs

### RediAI Integration Modes

**Mock Mode:**
- No external RediAI instance required
- Returns deterministic health responses
- Used in CI and automated testing

**Real Mode:**
- Connects to running RediAI instance
- Enables end-to-end integration testing
- Requires `REDIAI_BASE_URL` configuration

## Local Development

### Backend Setup

```bash
cd backend
python -m pip install -e ".[dev]"

# Run migrations
make db-upgrade  # or: alembic upgrade head

# Run linting and tests
ruff check .
ruff format --check .
mypy tunix_rt_backend
pytest --cov=tunix_rt_backend --cov-branch --cov-fail-under=70

# Start server
uvicorn tunix_rt_backend.app:app --reload --port 8000
```

### Database Migrations (M2)

```bash
# Apply migrations
make db-upgrade

# Create new migration after model changes
make db-revision msg="description"

# Rollback last migration
make db-downgrade

# View migration history
make db-history
```

### Frontend Setup

```bash
cd frontend
npm ci
npm run test
npm run build

# Start dev server
npm run dev
```

### E2E Tests

**Quick Start (M4+):**
```bash
# Run E2E with full infrastructure setup
make e2e

# Stop infrastructure when done
make e2e-down
```

**Manual:**
```bash
cd e2e
npm ci

# Mock mode (no RediAI required)
REDIAI_MODE=mock npx playwright test

# Real mode (requires RediAI)
REDIAI_MODE=real REDIAI_BASE_URL=http://localhost:8080 npx playwright test
```

**M4 Changes:**
- All servers bind to `127.0.0.1` (IPv4) to avoid IPv6 connection issues
- CI includes Postgres service container + automated migrations
- Playwright config supports environment variables for port configuration
- `make e2e` handles full lifecycle (DB setup → migrations → tests)

### Docker Compose

```bash
# Start services
docker compose up -d

# Check health
curl http://localhost:8000/api/health
curl http://localhost:8000/api/redi/health

# Stop services
docker compose down
```

#### Accessing Host RediAI from Docker

When running backend in Docker and RediAI on the host:

```bash
# Set in .env or docker-compose.yml
REDIAI_BASE_URL=http://host.docker.internal:8080
```

**Linux users:** Add to `docker-compose.yml`:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline uses conditional jobs based on changed files:

1. **changes**: Always runs, detects which components changed
2. **backend**: Runs if `backend/` or `.github/workflows/` changed
   - Ruff linting and formatting
   - mypy type checking
   - pytest with 70% coverage gate
3. **frontend**: Runs if `frontend/` or `.github/workflows/` changed
   - Vitest unit tests
   - Production build
4. **e2e**: Runs if any code or workflow changed
   - Playwright tests with mock RediAI
   - Full integration testing

### Path Filtering

Uses `dorny/paths-filter@v2` to avoid merge-blocking issues with required checks. When only documentation changes, jobs skip cleanly without blocking PRs.

**Important CI Invariants (M3 Hardening):**
- **Concrete SHAs Required**: paths-filter uses event-aware commit SHAs, never symbolic refs like `HEAD`
  - Pull requests: `github.event.pull_request.base.sha` and `github.event.pull_request.head.sha`
  - Push events: `github.event.before` and `github.sha`
- **Full History Checkout**: `changes` job uses `fetch-depth: 0` to ensure diff computation works correctly
- **Validation Step**: CI fails fast with clear error if base/ref SHAs are empty (prevents silent misconfigurations)
- **Why This Matters**: Symbolic refs can point to incorrect commits or not exist, causing nondeterministic CI failures. Event-aware SHAs guarantee reproducible builds.

## Testing Strategy

### Backend Tests (pytest)

- **Unit tests**: `tests/test_health.py`, `tests/test_redi_health.py`
- **Dependency injection**: Tests use `app.dependency_overrides` for deterministic RediAI responses
- **Coverage gate**: 70% minimum (enforced in CI)
- **Test markers**: `unit`, `integration`

**Example:**
```bash
cd backend
pytest -v
pytest --cov=tunix_rt_backend --cov-report=term --cov-fail-under=70
```

### Frontend Tests (Vitest)

- **Unit tests**: `src/App.test.tsx`
- **Mock fetch**: No external dependencies
- **Component testing**: React Testing Library

**Example:**
```bash
cd frontend
npm run test
```

### E2E Tests (Playwright)

- **Smoke tests**: `tests/smoke.spec.ts`
- **Mock mode**: CI runs with no external dependencies
- **Real mode**: Local testing with actual RediAI

**Example:**
```bash
cd e2e
REDIAI_MODE=mock npx playwright test
REDIAI_MODE=real npx playwright test --headed
```

## Project Structure

```
tunix-rt/
├── backend/                     # FastAPI backend
│   ├── tunix_rt_backend/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI routes (thin controllers, <600 lines)
│   │   ├── services/           # Business logic layer (M10, M11)
│   │   │   ├── traces_batch.py         # Batch trace operations
│   │   │   ├── datasets_export.py      # Dataset export formatting
│   │   │   ├── datasets_builder.py     # Dataset manifest creation (M11)
│   │   │   ├── ungar_generator.py      # UNGAR trace generation (M11)
│   │   │   ├── tunix_execution.py      # Run execution & cancellation (M16)
│   │   │   ├── evaluation.py           # Evaluation & Scoring (M17/M18)
│   │   │   ├── regression.py           # Regression testing (M18)
│   │   │   ├── judges.py               # Judge implementations (M18)
│   │   ├── helpers/            # Utilities
│   │   │   ├── datasets.py
│   │   │   └── traces.py
│   │   ├── integrations/       # External service integrations
│   │   │   └── ungar/          # Optional UNGAR integration
│   │   ├── redi_client.py      # RediAI client (real + mock)
│   │   └── settings.py         # Environment configuration
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_services.py            # Service layer tests (M10)
│   │   ├── test_services_ungar.py      # UNGAR service tests (M11)
│   │   ├── test_services_datasets.py   # Dataset service tests (M11)
│   │   ├── test_redi_health.py
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                    # Vite + React + TypeScript
│   ├── src/
│   │   ├── App.tsx             # Main component
│   │   ├── App.test.tsx        # Unit tests
│   │   ├── main.tsx            # Entry point
│   │   ├── index.css           # Styles
│   │   ├── components/         # React Components (M16)
│   │   │   ├── LiveLogs.tsx    # Live log streaming
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── e2e/                        # Playwright E2E tests
│   ├── tests/
│   │   └── smoke.spec.ts
│   ├── package.json
│   └── playwright.config.ts
├── .github/workflows/
│   └── ci.yml                  # CI pipeline (SHA-pinned actions)
├── docker-compose.yml          # Postgres + backend
├── LICENSE                     # Apache-2.0
└── README.md                   # User documentation
```

## Development Guidelines

### Conventional Commits

All commits follow Conventional Commits format:

```bash
feat(backend): add new endpoint
fix(frontend): resolve rendering issue
test(e2e): add smoke test
chore(ci): update workflow
docs: update README
```

### Code Quality

**Backend:**
- Ruff for linting and formatting
- mypy for type checking (strict mode)
- pytest for testing (70% coverage minimum)

**Frontend:**
- TypeScript strict mode
- Vitest for unit tests
- React Testing Library best practices

**E2E:**
- Playwright for browser automation
- Mock mode for CI reliability
- Real mode for integration validation

## Troubleshooting

### RediAI Connection Issues

**Problem:** `/api/redi/health` returns `{"status": "down"}`

**Solutions:**
1. Verify RediAI is running: `curl http://localhost:8080/health`
2. Check `REDIAI_BASE_URL` in environment variables
3. Switch to mock mode for testing: `REDIAI_MODE=mock`

### Docker Compose Issues

**Problem:** Backend can't reach RediAI on host

**Solutions:**
1. Set `REDIAI_BASE_URL=http://host.docker.internal:8080`
2. On Linux, add `extra_hosts` to docker-compose.yml (see documentation above)

### CI Failures

**Problem:** E2E tests fail in CI

**Cause:** Usually environment mismatch (CI uses mock mode)

**Solution:** Ensure `REDIAI_MODE=mock` is set in CI workflow

## M1 Features (Hardening & Guardrails)

**M1 Complete** ✅ - Enterprise-grade hardening without scope expansion

### Testing & Coverage
- **Branch coverage enforcement**: 90% achieved (≥68% gate)
- **Line coverage**: 92.39% (≥80% gate)
- **Custom coverage gate**: `backend/tools/coverage_gate.py`
- **21 comprehensive tests** (200% increase from M0)

### Security Baseline
- **Automated scanning**: pip-audit, npm audit, gitleaks
- **SBOM generation**: CycloneDX format (backend)
- **Dependabot**: Weekly updates for pip, npm, GitHub Actions
- **Configuration validation**: Pydantic validators with fail-fast
- **Documentation**: `SECURITY_NOTES.md` tracks vulnerabilities

### Features
- **TTL Cache**: `/api/redi/health` cached for 30s (configurable)
- **Frontend Polling**: Status updates every 30s automatically
- **Typed API Client**: TypeScript interfaces for type-safe API calls
- **Better Diagnostics**: Detailed HTTP error messages (codes, timeouts, connection)

### Developer Experience
- **Makefile**: `make install`, `make test`, `make docker-up`, etc.
- **PowerShell scripts**: `.\scripts\dev.ps1` for Windows users
- **Cross-platform**: Works on Mac, Linux, and Windows
- **Architecture Decision Records**: 3 ADRs documenting key decisions

### CI Enhancements
- **Security jobs**: 3 new jobs (pip-audit, npm audit, gitleaks)
- **Git-based operations**: No GitHub API dependencies (fork-safe)
- **Coverage enforcement**: Automated dual-threshold gates
- **Conditional execution**: Fast feedback with path filtering

## Completed Milestones

### M1: Hardening & Guardrails ✅
- Enterprise-grade testing (92% line, 90% branch coverage)
- Security scanning (pip-audit, npm audit, gitleaks)
- Configuration validation with Pydantic
- TTL caching for RediAI health
- Frontend polling and typed API client
- Developer experience tools (Makefile, scripts, ADRs)

### M2: Trace Storage & Retrieval ✅
- Async SQLAlchemy + PostgreSQL database integration
- Alembic migrations for schema management
- Trace model with UUID, timestamps, JSON payload
- Three trace endpoints: POST (create), GET by ID, GET list (paginated)
- Payload size validation (1MB default limit)
- Comprehensive backend tests (17 new tests)
- Frontend trace UI with upload/fetch functionality
- E2E test for complete trace flow
- Frontend coverage measurement (60% lines, 50% branches)

### M3: Trace System Hardening ✅
- DB connection pool settings (pool_size, max_overflow, pool_timeout) applied to create_async_engine
- created_at index migration (f8f1393630e4) for improved list query performance
- Index `ix_traces_created_at` tested on SQLite (CI parity) and verified via SQL
- Frontend trace UI unit tests: 8 tests total (Load Example, Upload success, Fetch success)
- Frontend coverage artifacts confirmed generating (coverage/coverage-final.json)
- Alembic auto-generated UUID migration policy documented (no manual revision IDs)
- Curl examples added to README for all trace endpoints
- DB troubleshooting guide added to README (docker compose, psql, alembic commands)
- All tests passing: Backend 92% line/90% branch, Frontend 8/8 tests

## Completed Milestones

### M4: E2E Infrastructure Hardening ✅
- IPv4 (127.0.0.1) standardization across all services
- Postgres service container in CI with healthcheck
- Automated migrations before E2E tests
- Cross-platform Playwright webServer configuration
- Reduced retries from 2 to 1 (no flakiness masking)
- Local `make e2e` target for full lifecycle testing
- All 5 E2E tests passing (including trace upload/fetch)
- CORS support for both localhost and 127.0.0.1

### M5: Evaluation & Comparison Loop (Phase 1) ✅
- Baseline scoring logic (0-100 range based on step count + average length)
- `POST /api/traces/{id}/score` endpoint with 201 Created response
- `GET /api/traces/compare` endpoint for side-by-side comparison
- Scores table with FK to traces (cascade delete)
- Frontend comparison UI with side-by-side layout
- 12 backend tests for scoring logic and endpoints (all passing)
- 3 new frontend tests for comparison UI (11 total)
- E2E test for complete comparison flow with two distinct traces
- API documentation in README.md and tunix-rt.md

### M6: Validation Refactor & CI Stability Hardening ✅
- Validation helper extraction (`get_trace_or_404`) with optional label parameter
- Centralized validation logic (eliminated 3 instances of duplication)
- Removed synthetic branch flags (coverage workarounds)
- Coverage improvements: 90% line (+1%), 88% branch (+9%)
- E2E selector hardening with data-testid convention (`sys:*`, `trace:*`, `compare:*`)
- All text-based selectors replaced with getByTestId (no global text matching)
- 3 new helper unit tests (100% coverage)
- Comprehensive guardrails documentation (validation + selector rules)
- Frontend tests updated to use new naming convention (11/11 passing)
- Backend: 56 tests passing | Frontend: 11 tests passing

### M7: UNGAR Integration Bridge ✅
- **Optional dependency**: UNGAR installable via `backend[ungar]` extra
- **Pinned version**: Commit `0e29e104aa1b13542b193515e3895ee87122c1cb` for reproducibility
- **High Card Duel generator**: Converts game episodes to reasoning traces
- **JSONL export**: Tunix-friendly format with `prompts`, `trace_steps`, `final_answer`
- **Three new endpoints**:
  - `GET /api/ungar/status` - Check availability
  - `POST /api/ungar/high-card-duel/generate` - Generate traces
  - `GET /api/ungar/high-card-duel/export.jsonl` - Export JSONL
- **Frontend panel**: Minimal UNGAR UI with status display and generator
- **Testing**:
  - Default tests (3): Verify 501 responses without UNGAR
  - Optional tests (6): Full integration validation with UNGAR installed
  - All 59 backend tests passing, 11 frontend tests passing
- **CI**: Optional workflow (`.github/workflows/ungar-integration.yml`) for manual/nightly runs
- **Documentation**: Complete integration guide in `docs/M07_UNGAR_INTEGRATION.md`
- **Guardrails**: Core runtime never requires UNGAR; graceful degradation with 501 responses
- **Coverage maintained**: 90% line, 88% branch

## Completed Milestones

### M8: Dataset & Training Bridge v1 ✅
- Dataset manifest system (file-based, versioned)
- Dataset build endpoint (latest/random strategies)
- Dataset export endpoint (trace + tunix_sft formats)
- Tunix SFT prompt renderer (Gemma chat template)
- Training smoke harness (optional backend[training] dependency)
- M7 hardening (type comments, logging, E2E, quick start)
- 23 new tests (13 dataset + 9 renderer + 1 E2E)
- Documentation: M08_BASELINE.md, M08_SUMMARY.md, updated README

### M9: Reproducible Training Loop v1 (SFT) ✅
- TrainingExample schema and manifests (5 schemas total)
- Enhanced Gemma IT formatting helpers
- Three dataset export formats (trace, tunix_sft, training_example)
- Batch trace import endpoint (POST /api/traces/batch)
- Training infrastructure (train_sft_tunix.py, eval scripts)
- Static evaluation set (25 diverse examples)
- Evaluation loop with delta reporting
- 39 new tests (127 total), 80% line coverage
- 5 comprehensive docs (BASELINE, DATASET_FORMAT, TRAINING_QUICKSTART, EVAL_LOOP, SUMMARY)
- ADR-005: Coverage Gates Strategy

### M10: App Layer Refactor + Determinism Guardrails ✅
- Service layer architecture (services/ directory)
- Thin controller pattern (app.py reduced by 14%)
- Typed export format validation (ExportFormat Literal)
- Batch endpoint optimization (~10x perf improvement)
- Timezone-aware UTC datetimes (zero deprecation warnings)
- 5 new service layer tests (132 total), 84% line coverage (+5.34%)
- 7 architectural guardrails documented
- 3 comprehensive docs (BASELINE, GUARDRAILS, SUMMARY)

### M11: Stabilize + Complete Service Extraction + Training Script Smoke Tests ✅
- **Complete app extraction:** UNGAR + dataset build moved to services/
- **app.py thin controller:** 741 → 588 lines (21% reduction, <600 line target achieved)
- **4 total services:** traces_batch, datasets_export, datasets_builder, ungar_generator
- **Security hardening:** SHA-pinned GitHub Actions, SBOM re-enabled, pre-commit hooks
- **Training infrastructure:** Dry-run smoke tests via subprocess (7 tests, JAX-gated)
- **Frontend coverage boost:** 60% → 77% line coverage (16 tests, +5 component tests)
- **Production documentation:** ADR-006, TRAINING_PRODUCTION.md, PERFORMANCE_SLOs.md
- **Test growth:** 146 backend tests (+14), 16 frontend tests (+5), 10 total skipped (optional deps)
- **Comprehensive M11 documentation:** BASELINE.md, SUMMARY.md

### M12: Tunix Integration Skeleton + Run Manifest Pipeline (Phase 1) ✅
- **Mock-first integration:** No Tunix runtime dependency required
- **Artifact-based approach:** Generates JSONL exports + YAML training manifests
- **3 new endpoints:** /api/tunix/status, /sft/export, /sft/manifest
- **Reuses tunix_sft format:** Leverages M09 Gemma chat template export format
- **YAML manifest generation:** Complete training configs with hyperparameters
- **Frontend Tunix panel:** Status display + export/manifest generation UI
- **Test growth:** 160 backend tests (+14), 21 frontend tests (+5)
- **Coverage improvement:** 92% backend line (+8%), 77% frontend line (maintained)
- **Optional CI workflow:** tunix-integration.yml (non-blocking, nightly)
- **Complete documentation:** M12_BASELINE.md, M12_TUNIX_INTEGRATION.md

### M13: Tunix Runtime Execution (Phase 2) ✅
- **Optional, gated execution:** Requires `backend[tunix]` installation for local runs
- **Dry-run mode (default):** Validates configuration without executing
- **Local execution mode:** Runs `tunix train` via subprocess, captures output
- **Graceful degradation:** Returns 501 if Tunix unavailable and `dry_run=false`
- **No TPU assumptions:** CPU/GPU only for M13
- **New endpoint:** POST /api/tunix/run (dry_run param, 202 response)
- **Execution metadata:** run_id, status, timestamps, stdout/stderr, exit_code
- **Frontend UI:** "Run with Tunix" buttons with results display (collapsible logs)
- **CI workflow:** Separate `tunix-runtime.yml` (manual, never blocks merge)
- **Test growth:** 168 backend tests (+8), 25 frontend tests (+4)
- **Coverage maintained:** 92% backend line, 77% frontend line
- **Complete documentation:** M13_BASELINE.md, M13_TUNIX_EXECUTION.md

### M14: Tunix Run Registry (Phase 3) ✅
- **Persistent storage:** `tunix_runs` table with UUID primary key, indexed columns
- **Alembic migration:** Reversible schema changes (upgrade + downgrade)
- **Immediate persistence:** Create run record with status="running", update on completion
- **Graceful DB failures:** Log errors, don't fail user requests (execution is primary)
- **No execution changes:** M13 execution logic remains identical (black box)
- **New endpoints:** GET /api/tunix/runs (pagination + filtering), GET /api/tunix/runs/{run_id}
- **Filtering:** By status, dataset_key, mode (AND logic)
- **Stdout/stderr truncation:** 10KB per field (prevents DB bloat)
- **Frontend Run History panel:** Collapsible section with manual refresh, expandable details
- **Test growth:** 192 backend tests (+12 dry-run), 28 frontend tests (+7 mocked)
- **Coverage maintained:** 82% backend line, 77% frontend line
- **Complete documentation:** M14_BASELINE.md, M14_RUN_REGISTRY.md, M14_SUMMARY.md

### M15: Async Execution & Run Registry (Phase 4) ✅
- **Async API:** `POST /api/tunix/run?mode=async` returns `200 OK` + `status="pending"` immediately.
- **Worker Process:** `worker.py` consumes pending runs using `SKIP LOCKED` for atomic claiming.
- **Status Endpoint:** `GET /api/tunix/runs/{id}/status` for lightweight polling.
- **Observability:** `/metrics` endpoint exposing Prometheus counters and histograms.
- **Frontend Updates:** "Run Async" toggle, auto-polling for pending runs, status badges.
- **Schema:** Added `config` JSON column to `tunix_runs` for persisting run parameters.
- **Refactoring:** Decoupled execution logic from API request lifecycle.
- **Tests:** Backend unit tests for async flow + worker logic; E2E test for async UI flow.
- **Documentation:** Updated README with worker info; new PERFORMANCE_BASELINE.md.

### M16: Operational UX & Hardening (Phase 5) ✅
- **Live Logs:** Real-time streaming via SSE (`/api/tunix/runs/{id}/logs`).
- **Cancellation:** Stop running/pending jobs (`POST /cancel`).
- **Artifacts:** List and download run outputs.
- **Hardening:** Pinned dependencies, optimized batch insert.
- **Schema:** `tunix_run_log_chunks` table.

### M17: Evaluation & Model Quality Loop (Phase 6) ✅
- **Evaluation Engine:** Deterministic "Mock Judge" service.
- **Database:** `tunix_run_evaluations` table for metrics and verdicts.
- **Auto-Trigger:** Runs automatically after successful completion (skips dry-runs).
- **Leaderboard:** New UI page for ranking runs by score.
- **Endpoints:** `GET /evaluation`, `GET /leaderboard`, `POST /evaluate`.

### M18: Judge Abstraction & Regression (Phase 7) ✅
- **Real Judge:** Pluggable `Judge` interface with `GemmaJudge` (via RediAI).
- **Regression Gates:** New service and DB table for named baselines.
- **Pagination:** Leaderboard API and UI now support pagination.
- **Credibility:** Evaluation decoupled from service logic.

## Next Steps (M19+)

1. **M19**: Hyperparameter Tuning (Ray Tune)
2. **M20**: Model Registry & Deployment

## Architecture Decisions

Key architectural decisions are documented in Architecture Decision Records (ADRs):

- **ADR-001**: Mock/Real Mode RediAI Integration Pattern
- **ADR-002**: CI Conditional Jobs Strategy with Path Filtering
- **ADR-003**: Coverage Strategy (Line + Branch Thresholds)
- **ADR-004**: Optional Code Coverage Strategy
- **ADR-005**: Coverage Gates for Optional and Expanding Runtime Code (M09)
- **ADR-006**: Tunix API Abstraction Pattern (M11)

See `docs/adr/` for full details.

## License

Apache-2.0

---

**Last Updated:** M18 Complete  
**Version:** 0.10.0  
**Coverage:** Backend 82% Line, Frontend 77% Line  
**Security:** SHA-Pinned CI + SBOM + Pre-commit Hooks  
**Architecture:** Tunix Async + Evaluation Loop + Real Judges  
**Tests:** 237 total (209 backend + 28 frontend)
