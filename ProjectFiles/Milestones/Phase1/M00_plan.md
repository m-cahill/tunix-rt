You are implementing Milestone M0 for repo m-cahill/tunix-rt.

Context / goals:
- Repo is currently EMPTY.
- M0 must be a bare-minimum FULL STACK monorepo that is still enterprise-lean: tested, CI-gated, fast.
- RediAI is already running locally OUTSIDE this repo. We want integration EARLY, but CI must never require RediAI.
- License: Apache-2.0.
- Stay small: only a health probe integration for RediAI in M0 (no job submission / trace upload yet).

Primary outcomes by end of M0:
1) Backend (FastAPI) serves:
   - GET /api/health -> {"status":"healthy"}
   - GET /api/redi/health -> probes RediAI via injected client and returns:
       - {"status":"healthy"} if reachable
       - {"status":"down","error":"..."} if unreachable
   These must have deterministic unit tests (no real network).
2) Frontend (Vite + React + TS) renders a page that fetches:
   - /api/health and shows "API: healthy"
   - /api/redi/health and shows "RediAI: healthy" or "RediAI: down"
3) E2E (Playwright) has one smoke test that loads the page and asserts "API: healthy".
   - In CI: backend runs with REDIAI_MODE=mock so RediAI is deterministic and not required.
   - Locally: allow REDIAI_MODE=real + REDIAI_BASE_URL to hit the real running RediAI.
4) docker-compose.yml provides postgres + backend (web optional), with healthchecks and depends_on service_healthy.
   - Support calling host RediAI from container via host.docker.internal (document in README).
5) CI:
   - Separate jobs for backend, frontend, e2e.
   - Use caching for Python/pip and Node/npm.
   - Use monorepo path filtering so we don’t waste CI time.
   - IMPORTANT GUARDRAIL: avoid the “required checks + path filters” merge-block issue by using a single workflow with a paths-filter job that conditionally runs other jobs while still producing stable check results (use dorny/paths-filter).

Repository layout (M0):
tunix-rt/
  LICENSE (Apache-2.0)
  README.md
  .gitignore
  .editorconfig
  .env.example

  backend/
    pyproject.toml
    ruff.toml (or ruff config in pyproject)
    mypy.ini
    tunix_rt_backend/
      __init__.py
      app.py
      redi_client.py
      settings.py (optional, minimal env parsing)
    tests/
      test_health.py
      test_redi_health.py

  frontend/
    package.json
    package-lock.json (use npm)
    vite.config.ts
    tsconfig.json
    src/
      main.tsx
      App.tsx
    tests/
      App.test.tsx

  e2e/
    package.json (or reuse frontend deps; choose the smallest maintainable option)
    playwright.config.ts
    tests/
      smoke.spec.ts

  docker-compose.yml

  .github/workflows/
    ci.yml

Implementation phases (keep commits small; Conventional Commits):
------------------------------------------------------------

PHASE M0.1 — Bootstrap repo (push-ready)
- Add Apache-2.0 LICENSE.
- Add .gitignore, .editorconfig.
- Add README with:
  - Quickstart (backend, frontend, e2e).
  - RediAI integration notes (mock vs real).
  - Compose notes (host.docker.internal).
- Add .env.example with:
  - BACKEND_PORT=8000
  - FRONTEND_PORT=5173 (dev)
  - REDIAI_MODE=mock|real (default mock)
  - REDIAI_BASE_URL=http://localhost:<your-port>
  - REDIAI_HEALTH_PATH=/health (or whatever)
- Verification: repo clean, pushed.

PHASE M0.2 — Backend (FastAPI) + RediAI boundary + deterministic tests
Backend requirements:
- Use FastAPI with dependency injection for the Redi client.
- Implement tunix_rt_backend/redi_client.py:
  - RediClient with method health() using httpx (or requests). Keep minimal.
  - Must raise/return structured error on connection issues.
- Implement dependency provider get_redi_client() in app.py (or deps.py):
  - If REDIAI_MODE=mock: return a MockRediClient that always reports healthy (or configurable).
  - If REDIAI_MODE=real: return RediClient(base_url, health_path).
- app.py:
  - GET /api/health
  - GET /api/redi/health (calls injected client)
- Tests:
  - test_health.py: TestClient GET /api/health -> 200 + exact JSON.
  - test_redi_health.py:
    - Use FastAPI app.dependency_overrides to inject:
      (a) fake client returning healthy -> assert {"status":"healthy"}
      (b) fake client raising exception -> assert {"status":"down", "error": ...}
    - Guardrail: ensure dependency_overrides are cleared after tests to avoid leakage.
- Tooling (backend):
  - ruff (lint + format check)
  - mypy strict (scope ONLY to tunix_rt_backend)
  - pytest + coverage gate: --cov-fail-under=70 (M0), keep tiny tests but real.
- Verification (local):
  cd backend
  python -m pip install -e ".[dev]"
  ruff check .
  ruff format --check .
  mypy tunix_rt_backend
  pytest -q

PHASE M0.3 — Frontend (Vite + React + TS) + unit tests
Frontend requirements:
- Vite React TS app.
- App.tsx:
  - On load, fetch /api/health and /api/redi/health.
  - Render two lines:
    "API: healthy" (or "API: down")
    "RediAI: healthy" (or "RediAI: down")
  - Keep UI minimal.
- vite.config.ts:
  - Dev proxy /api -> http://localhost:8000 (or BACKEND_PORT).
- Tests (Vitest + React Testing Library):
  - Mock fetch for both endpoints.
  - Assert DOM updates accordingly.
- Verification (local):
  cd frontend
  npm ci
  npm run test
  npm run build

PHASE M0.4 — E2E smoke (Playwright) with deterministic CI behavior
E2E requirements:
- Use Playwright test runner.
- playwright.config.ts:
  - Use webServer to start backend and frontend before tests.
  - In CI, start backend with REDIAI_MODE=mock to avoid needing RediAI at all.
  - Locally, allow developer to set REDIAI_MODE=real and REDIAI_BASE_URL to hit their running RediAI.
- smoke.spec.ts:
  - Load "/" and assert "API: healthy" is visible.
  - Optionally assert RediAI line exists (but do NOT require real RediAI in CI).
- Verification (local):
  - With mock:
    REDIAI_MODE=mock run playwright
  - With real:
    REDIAI_MODE=real REDIAI_BASE_URL=http://localhost:<port> run playwright

PHASE M0.5 — docker-compose (postgres + backend) and docs
- docker-compose.yml:
  - postgres service with pg_isready healthcheck
  - backend service built from backend/ (or use python image + pip install)
  - depends_on postgres: condition: service_healthy
  - backend healthcheck hits /api/health
  - Document how backend container reaches host RediAI:
    - REDIAI_BASE_URL=http://host.docker.internal:<port>
    - Note: on Linux, may need extra_hosts: ["host.docker.internal:host-gateway"] (document, keep optional)
- Verification:
  docker compose up -d
  curl localhost:<backend-port>/api/health

PHASE M0.6 — CI (single workflow with conditional jobs + caching + guardrails)
- Create .github/workflows/ci.yml as ONE workflow that always runs and produces stable checks.
- Use dorny/paths-filter to detect changes in:
  - backend/**
  - frontend/**
  - e2e/**
  - .github/workflows/**
- Jobs:
  1) changes: runs paths-filter, outputs booleans.
  2) backend: runs only if backend changed (or workflow changed). Uses:
     - actions/setup-python (pip cache)
     - run ruff/mypy/pytest+coverage
  3) frontend: runs only if frontend changed (or workflow changed). Uses:
     - actions/setup-node@v4 with cache: npm
     - npm ci, npm test, npm run build
  4) e2e: runs only if backend/frontend/e2e changed (or workflow changed):
     - install deps
     - start with REDIAI_MODE=mock
     - run Playwright
- Ensure that when a section didn’t change, the job is cleanly skipped (but the workflow still reports required checks consistently).
- Keep runtime fast; fail hard on lint/test.
- Verification:
  - Open a PR touching backend only -> frontend job skipped, backend job runs, e2e runs.
  - Open a PR touching README only -> only “changes” job runs and workflow completes cleanly.

General rules:
- Keep code minimal and readable.
- No extra features beyond health endpoints, basic UI, and tests.
- Use Conventional Commits per phase.
- After all phases: update README with exact local commands for:
  - Backend dev server
  - Frontend dev server
  - E2E
  - Compose
  - Real vs mock RediAI integration

End-state verification checklist:
- Backend: all gates pass locally and in CI (ruff/mypy/pytest+coverage).
- Frontend: tests + build pass locally and in CI.
- E2E: Playwright smoke passes in CI with REDIAI_MODE=mock.
- README: copy-paste runnable quickstart.
- Repo pushed to GitHub.
