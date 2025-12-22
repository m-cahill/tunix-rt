# Tunix RT - Reasoning-Trace Framework

**Milestone M8 Complete** ✅  
**Coverage:** 84% Line (core), 89% Branch | **Security:** Baseline Operational | **Database:** PostgreSQL + Alembic | **Features:** Dataset Pipelines, Tunix SFT Rendering, Training Bridge & UNGAR Integration

## Overview

Tunix RT is a full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon. The system provides health monitoring, RediAI integration, and a foundation for trace-quality optimization workflows.

**M1 Enhancements:** Enterprise-grade testing (90% branch coverage), security scanning (pip-audit, npm audit, gitleaks), configuration validation, TTL caching, and developer experience tools.

**M2 Enhancements:** Database integration (async SQLAlchemy + PostgreSQL), Alembic migrations, trace CRUD API (create/retrieve/list), frontend trace UI, comprehensive validation, and payload size limits.

**M3 Enhancements:** Trace system hardening - DB connection pool settings applied, created_at index for list performance, frontend trace UI unit tests (8 total), frontend coverage artifact generation confirmed, Alembic auto-ID migration policy documented, curl API examples, and DB troubleshooting guide.

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
│   │   ├── app.py              # FastAPI routes
│   │   ├── redi_client.py      # RediAI client (real + mock)
│   │   └── settings.py         # Environment configuration
│   ├── tests/
│   │   ├── test_health.py
│   │   └── test_redi_health.py
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                    # Vite + React + TypeScript
│   ├── src/
│   │   ├── App.tsx             # Main component
│   │   ├── App.test.tsx        # Unit tests
│   │   ├── main.tsx            # Entry point
│   │   └── index.css           # Styles
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── e2e/                        # Playwright E2E tests
│   ├── tests/
│   │   └── smoke.spec.ts
│   ├── package.json
│   └── playwright.config.ts
├── .github/workflows/
│   └── ci.yml                  # CI pipeline
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

## Next Steps (M8+)

1. **M8**: Multi-game UNGAR support (Mini Spades, Gin Rummy)
2. **M9**: Tunix SFT training workflow integration
3. **M10**: Richer trace schemas with reasoning explanations
4. **M11**: Production deployment (Netlify + Render)

## Architecture Decisions

Key architectural decisions are documented in Architecture Decision Records (ADRs):

- **ADR-001**: Mock/Real Mode RediAI Integration Pattern
- **ADR-002**: CI Conditional Jobs Strategy with Path Filtering
- **ADR-003**: Coverage Strategy (Line + Branch Thresholds)

See `docs/adr/` for full details.

## License

Apache-2.0

---

**Last Updated:** M1 Complete  
**Version:** 0.2.0  
**Coverage:** 92% Line, 90% Branch  
**Security:** Baseline Operational
