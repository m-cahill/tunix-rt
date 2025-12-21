# tunix-rt

**Tunix Reasoning-Trace Framework for AI-Native Development**

**Status:** M5 Complete ✅ | Coverage: 89% Line, 79% Branch | Features: Trace Evaluation & Comparison | Database: PostgreSQL + Alembic

A full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+  
- Docker & Docker Compose
- RediAI running locally (for real mode integration)
- Make (optional, for Makefile commands)

### Quick Commands (M1+)

**Using Makefile (Mac/Linux/WSL):**
```bash
make install    # Install all dependencies
make test       # Run all tests with coverage
make lint       # Run linters and type checking
make docker-up  # Start Docker services
```

**Using PowerShell (Windows):**
```powershell
.\scripts\dev.ps1 install
.\scripts\dev.ps1 test
.\scripts\dev.ps1 lint
.\scripts\dev.ps1 docker-up
```

### Backend Setup (Manual)

```bash
cd backend
python -m pip install -e ".[dev]"

# Run linting and tests
ruff check .
ruff format --check .
mypy tunix_rt_backend
pytest --cov=tunix_rt_backend --cov-branch
python tools/coverage_gate.py  # Enforce line ≥80%, branch ≥68%

# Start development server
uvicorn tunix_rt_backend.app:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm ci
npm run test
npm run build

# Start development server
npm run dev
```

### E2E Tests

**Quick Start (Recommended - M4+):**

```bash
# Run E2E with full infrastructure (one command)
make e2e

# Stop infrastructure when done
make e2e-down
```

This will:
1. Start Postgres in Docker
2. Run database migrations
3. Run Playwright tests (automatically starts backend + frontend)
4. Leave Postgres running for iteration

**Manual Setup:**

```bash
cd e2e
npm ci

# Run with mock RediAI (no external dependencies)
REDIAI_MODE=mock npx playwright test

# Run with real RediAI (requires RediAI running)
REDIAI_MODE=real REDIAI_BASE_URL=http://localhost:8080 npx playwright test
```

**Notes:**
- E2E tests now use `127.0.0.1` instead of `localhost` to avoid IPv6 connection issues
- If you have a local Postgres on port 5432, temporarily change `docker-compose.yml` to use `5433:5432` and set `DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/postgres`

### Docker Compose

```bash
# Start postgres + backend
docker compose up -d

# Check health
curl http://localhost:8000/api/health
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (see configuration below):

```bash
# Backend
BACKEND_PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30

# Trace Configuration
TRACE_MAX_BYTES=1048576  # 1MB default

# Frontend (dev)
FRONTEND_PORT=5173

# RediAI Integration
REDIAI_MODE=mock  # or "real" (validated: must be "mock" or "real")
REDIAI_BASE_URL=http://localhost:8080  # (validated: must be valid HTTP/HTTPS URL)
REDIAI_HEALTH_PATH=/health  # (validated: must start with /)
REDIAI_HEALTH_CACHE_TTL_SECONDS=30  # Cache TTL for health checks (0-300, default: 30)
```

**M1 Configuration Validation:**
- All settings are validated on startup using Pydantic
- Invalid configuration causes immediate failure with clear error messages
- See `backend/tunix_rt_backend/settings.py` for validation logic

### RediAI Integration Modes

**Mock Mode (CI/Testing):**
- Set `REDIAI_MODE=mock`
- No external RediAI instance required
- Returns deterministic health responses
- Used in CI and E2E tests

**Real Mode (Local Development):**
- Set `REDIAI_MODE=real`
- Set `REDIAI_BASE_URL=http://localhost:8080` (or your RediAI URL)
- Requires RediAI running locally
- Enables end-to-end integration testing

### Docker Compose + Host RediAI

When running the backend in Docker and RediAI on your host machine:

```bash
# In docker-compose.yml or .env
REDIAI_BASE_URL=http://host.docker.internal:8080
```

**Linux Note:** You may need to add to `docker-compose.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

## Database Migrations

**M2+** tunix-rt uses Alembic for database migrations.

### Running Migrations

```bash
# Using Makefile
make db-upgrade      # Apply all pending migrations

# Manual
cd backend
alembic upgrade head

# Create new migration (after model changes)
make db-revision msg="description"
```

### Migration Commands

- `make db-upgrade` - Apply all pending migrations
- `make db-downgrade` - Rollback last migration
- `make db-current` - Show current database version
- `make db-history` - Show migration history
- `make db-revision msg="..."` - Create new migration

## API Endpoints

### Health Endpoints

- `GET /api/health` - tunix-rt application health
  - Response: `{"status": "healthy"}`
  
- `GET /api/redi/health` - RediAI integration health
  - Mock mode: `{"status": "healthy"}`
  - Real mode (healthy): `{"status": "healthy"}`
  - Real mode (down): `{"status": "down", "error": "..."}`

### Trace Endpoints (M2+)

**Create a new trace:**

```bash
curl -X POST http://localhost:8000/api/traces \
  -H "Content-Type: application/json" \
  -d '{
    "trace_version": "1.0",
    "prompt": "What is 27 × 19?",
    "final_answer": "513",
    "steps": [
      {"i": 0, "type": "parse", "content": "Parse the multiplication task"},
      {"i": 1, "type": "compute", "content": "27 × 19 = 513"}
    ],
    "meta": {"source": "example"}
  }'
```

Response: `{"id": "550e8400-...", "created_at": "2025-12-21T...", "trace_version": "1.0"}`

**Get a trace by ID:**

```bash
curl http://localhost:8000/api/traces/550e8400-e29b-41d4-a716-446655440000
```

Response: Full trace with payload

**List traces (paginated):**

```bash
# Get first 20 traces
curl http://localhost:8000/api/traces

# Get next page with custom limit
curl "http://localhost:8000/api/traces?limit=10&offset=20"
```

Response: `{"data": [...], "pagination": {"limit": 20, "offset": 0, "next_offset": 20}}`

**Score a trace:**

```bash
# Score a trace using baseline criteria
curl -X POST http://localhost:8000/api/traces/550e8400-e29b-41d4-a716-446655440000/score \
  -H "Content-Type: application/json" \
  -d '{"criteria": "baseline"}'
```

Response: 
```json
{
  "trace_id": "550e8400-...",
  "score": 67.5,
  "details": {
    "step_count": 5,
    "avg_step_length": 342.5,
    "step_score": 25.0,
    "length_score": 34.25,
    "criteria": "baseline"
  }
}
```

**Compare two traces:**

```bash
# Compare two traces side-by-side with scores
curl "http://localhost:8000/api/traces/compare?base=550e8400-e29b-41d4-a716-446655440000&other=660f9500-f39c-52e5-b827-557766551111"
```

Response: Both traces with full payloads and computed scores

### UNGAR Generator (Optional - M7)

**Check UNGAR availability:**

```bash
curl http://localhost:8000/api/ungar/status
```

**Generate High Card Duel traces:**

```bash
curl -X POST http://localhost:8000/api/ungar/high-card-duel/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 5, "seed": 42, "persist": true}'
```

**Export traces as JSONL:**

```bash
curl "http://localhost:8000/api/ungar/high-card-duel/export.jsonl?limit=10"
```

**Installation:** To use UNGAR features, install with the optional extra:

```bash
cd backend
pip install -e ".[dev,ungar]"
```

See [docs/M07_UNGAR_INTEGRATION.md](docs/M07_UNGAR_INTEGRATION.md) for complete documentation.

## Project Structure

```
tunix-rt/
├── backend/                 # FastAPI backend
│   ├── tunix_rt_backend/    # Main package
│   │   ├── app.py           # FastAPI app with routes
│   │   ├── redi_client.py   # RediAI client (real + mock)
│   │   └── settings.py      # Environment configuration
│   ├── tests/               # Backend tests
│   └── pyproject.toml       # Python dependencies
├── frontend/                # Vite + React + TypeScript
│   ├── src/
│   │   ├── App.tsx          # Main app component
│   │   └── main.tsx         # Entry point
│   ├── tests/               # Frontend unit tests
│   └── package.json
├── e2e/                     # Playwright E2E tests
│   ├── tests/
│   │   └── smoke.spec.ts    # Smoke test
│   └── playwright.config.ts
├── docker-compose.yml       # Postgres + backend services
└── .github/workflows/       # CI pipeline
    └── ci.yml
```

## Development

### Run All Quality Checks

**Backend:**
```bash
cd backend
ruff check .
ruff format --check .
mypy tunix_rt_backend
pytest --cov=tunix_rt_backend --cov-report=term --cov-fail-under=70
```

**Frontend:**
```bash
cd frontend
npm run test
npm run build
```

**E2E:**
```bash
cd e2e
REDIAI_MODE=mock npx playwright test
```

### CI Pipeline

The CI pipeline uses conditional jobs based on changed files:

- **Backend changes** → Run backend linting, type checking, tests
- **Frontend changes** → Run frontend tests and build
- **E2E changes** → Run Playwright tests (with mock RediAI)
- **README-only changes** → Skip all jobs cleanly

Uses `dorny/paths-filter` to avoid merge-blocking issues with required checks.

## Testing Strategy

### Backend Tests

- **Unit tests**: `tests/test_health.py`, `tests/test_redi_health.py`
- **Dependency injection**: Tests use `app.dependency_overrides` for deterministic RediAI responses
- **Coverage gate**: 70% minimum (enforced in CI)

### Frontend Tests

- **Unit tests**: Vitest + React Testing Library
- **Coverage gates**: 60% line, 50% branch (enforced in CI)
- **Mock fetch**: No external dependencies
- **Component testing**: Trace UI (Load/Upload/Fetch) + Health monitoring

### E2E Tests

- **Smoke test**: Load page, assert "API: healthy" visible
- **Mock mode**: CI runs with `REDIAI_MODE=mock` (no RediAI required)
- **Real mode**: Local testing with actual RediAI instance

## Troubleshooting

### Database Issues

**Check PostgreSQL is running:**

```bash
docker compose ps
```

Expected output: `tunix-rt-postgres-1` should show status "Up"

**Verify database connection:**

```bash
# From host machine
psql postgresql://postgres:postgres@localhost:5432/postgres -c "SELECT 1;"

# Or using Docker
docker exec -it tunix-rt-postgres-1 psql -U postgres -c "SELECT 1;"
```

**Run migrations:**

```bash
cd backend
alembic upgrade head
```

If migrations fail, check:
- PostgreSQL is running (`docker compose ps`)
- `DATABASE_URL` environment variable is correct
- Database user has sufficient permissions

**Check current migration version:**

```bash
cd backend
alembic current
```

**View migration history:**

```bash
cd backend
alembic history --verbose
```

### RediAI Connection Issues

If `/api/redi/health` returns `{"status": "down"}`:

1. Verify RediAI is running:
   ```bash
   curl http://localhost:8080/health
   ```

2. Check `REDIAI_BASE_URL` environment variable matches your RediAI instance

3. For testing, switch to mock mode:
   ```bash
   export REDIAI_MODE=mock
   ```

### Docker Compose Issues

**Backend can't reach RediAI on host:**

Set `REDIAI_BASE_URL=http://host.docker.internal:8080` in your `.env` file.

On Linux, add to `docker-compose.yml`:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

**Port conflicts:**

If port 5432 or 8000 is already in use:
```bash
# Check what's using the port
netstat -ano | findstr :5432   # Windows
lsof -i :5432                   # Mac/Linux

# Stop conflicting containers
docker ps
docker stop <container_name>
```

## License

Apache-2.0

## Contributing

Follow Conventional Commits format:

```bash
feat(backend): add new endpoint
fix(frontend): resolve rendering issue
test(e2e): add smoke test
chore(ci): update workflow
docs: update README
```
