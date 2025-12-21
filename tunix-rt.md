# Tunix RT - Reasoning-Trace Framework

**Milestone M1 Complete** ✅  
**Coverage:** 92% Line, 90% Branch | **Security:** Baseline Operational

## Overview

Tunix RT is a full-stack application for managing reasoning traces and integrating with the RediAI framework for the Tunix Hackathon. The system provides health monitoring, RediAI integration, and a foundation for trace-quality optimization workflows.

**M1 Enhancements:** Enterprise-grade testing (90% branch coverage), security scanning (pip-audit, npm audit, gitleaks), configuration validation, TTL caching, and developer experience tools.

## System Architecture

### Components

1. **Backend (FastAPI)**
   - Health monitoring endpoints
   - RediAI integration with mock/real modes
   - Dependency injection for testability
   - 70% test coverage minimum

2. **Frontend (Vite + React + TypeScript)**
   - Real-time health status display with 30s auto-refresh (M1)
   - RediAI integration monitoring
   - Typed API client for type-safe backend calls (M1)
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

## Database Schema

### M0 Status
No database tables in M0 - health endpoints only. Database integration planned for M1.

## Configuration

### Environment Variables

| Variable | Default | Description | Validation (M1) |
|----------|---------|-------------|-----------------|
| `BACKEND_PORT` | `8000` | FastAPI server port | Must be 1-65535 |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string | None |
| `FRONTEND_PORT` | `5173` | Vite dev server port | None |
| `REDIAI_MODE` | `mock` | RediAI integration mode | Must be "mock" or "real" |
| `REDIAI_BASE_URL` | `http://localhost:8080` | RediAI instance URL (real mode) | Must be valid HTTP/HTTPS URL |
| `REDIAI_HEALTH_PATH` | `/health` | RediAI health endpoint path | None |
| `REDIAI_HEALTH_CACHE_TTL_SECONDS` | `30` | Cache TTL for health checks (M1) | Must be 0-300 |

**Configuration Validation (M1):**
- All settings are validated on application startup using Pydantic
- Invalid configuration causes immediate failure with descriptive error messages
- See `backend/tunix_rt_backend/settings.py` for validation logic

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

## Next Steps (M2+)

M1 provides enterprise-grade hardening. Future milestones will add:

1. **M2**: Database models and trace storage (Alembic migrations)
2. **M3**: Trace upload and retrieval endpoints
3. **M4**: RediAI workflow registry integration
4. **M5**: Trace quality metrics and optimization
5. **M6**: Deployment to Netlify (frontend) and Render (backend)

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
