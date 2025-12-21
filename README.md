# tunix-rt

**Tunix Reasoning-Trace Framework for AI-Native Development**

**Status:** M1 Complete ✅ | Coverage: 92% Line, 90% Branch | Security: Baseline Operational

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

```bash
cd e2e
npm ci

# Run with mock RediAI (no external dependencies)
REDIAI_MODE=mock npx playwright test

# Run with real RediAI (requires RediAI running)
REDIAI_MODE=real REDIAI_BASE_URL=http://localhost:8080 npx playwright test
```

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

- `POST /api/traces` - Create a new reasoning trace
  - Request body: ReasoningTrace JSON
  - Response: `{"id": "uuid", "created_at": "...", "trace_version": "..."}`
  - Status: 201 Created, 413 Payload Too Large, 422 Validation Error
  
- `GET /api/traces/{id}` - Get a trace by ID
  - Response: Full trace with payload
  - Status: 200 OK, 404 Not Found
  
- `GET /api/traces?limit=20&offset=0` - List traces (paginated)
  - Response: `{"data": [...], "pagination": {...}}`
  - Params: `limit` (1-100, default 20), `offset` (default 0)
  - Status: 200 OK, 422 Invalid Parameters

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
- **Mock fetch**: No external dependencies
- **Component testing**: Verify UI updates based on health responses

### E2E Tests

- **Smoke test**: Load page, assert "API: healthy" visible
- **Mock mode**: CI runs with `REDIAI_MODE=mock` (no RediAI required)
- **Real mode**: Local testing with actual RediAI instance

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
