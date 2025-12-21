# Tunix RT - Reasoning-Trace Framework

**Milestone M0 Complete** ✅

## Overview

Tunix RT is a full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon. The system provides health monitoring, RediAI integration, and a foundation for trace-quality optimization workflows.

## System Architecture

### Components

1. **Backend (FastAPI)**
   - Health monitoring endpoints
   - RediAI integration with mock/real modes
   - Dependency injection for testability
   - 70% test coverage minimum

2. **Frontend (Vite + React + TypeScript)**
   - Real-time health status display
   - RediAI integration monitoring
   - Responsive UI with status indicators

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

**Description:** Check RediAI integration health

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
  "error": "Connection refused"
}
```

**Status Codes:**
- `200 OK`: Request succeeded (check `status` field for actual health)

**Behavior:**
- **Mock Mode (`REDIAI_MODE=mock`)**: Always returns `{"status": "healthy"}`
- **Real Mode (`REDIAI_MODE=real`)**: Probes actual RediAI instance at `REDIAI_BASE_URL`

## Database Schema

### M0 Status
No database tables in M0 - health endpoints only. Database integration planned for M1.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_PORT` | `8000` | FastAPI server port |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `FRONTEND_PORT` | `5173` | Vite dev server port |
| `REDIAI_MODE` | `mock` | RediAI integration mode (`mock` or `real`) |
| `REDIAI_BASE_URL` | `http://localhost:8080` | RediAI instance URL (real mode) |
| `REDIAI_HEALTH_PATH` | `/health` | RediAI health endpoint path |

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

# Run linting and tests
ruff check .
ruff format --check .
mypy tunix_rt_backend
pytest --cov=tunix_rt_backend --cov-fail-under=70

# Start server
uvicorn tunix_rt_backend.app:app --reload --port 8000
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

```bash
cd e2e
npm ci

# Mock mode (no RediAI required)
REDIAI_MODE=mock npx playwright test

# Real mode (requires RediAI)
REDIAI_MODE=real REDIAI_BASE_URL=http://localhost:8080 npx playwright test
```

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

## Next Steps (M1+)

M0 provides the foundation. Future milestones will add:

1. **M1**: Database models for trace storage
2. **M2**: Trace upload and retrieval endpoints
3. **M3**: RediAI workflow registry integration
4. **M4**: Trace quality metrics and optimization
5. **M5**: Deployment to Netlify (frontend) and Render (backend)

## License

Apache-2.0

---

**Last Updated:** M0 Complete
**Version:** 0.1.0
