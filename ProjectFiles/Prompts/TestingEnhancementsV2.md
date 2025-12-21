Title: Full‑Stack Reliability & CI‑Gated Deploys (Python backend + React/Vite frontend)

Scope & Stack (fixed)

- Backend: Python 3.11+ with pytest, coverage.py, Hypothesis; optional mutmut
- Frontend: React + Vite + Vitest + Testing Library + Playwright (+ axe)
- CI: GitHub Actions (Linux runners)
- Deploy: Netlify (frontend), Render (backend)
- Infra helpers: Docker/Compose (integration tests), PostgreSQL service container
- Observability: JSON logs + OpenTelemetry with trace_id/span_id in logs
- Coverage targets (hard gates): line ≥ 85%, branch ≥ 85% (backend & frontend)

Repo layout (expected)

- /backend  (Python package, tests under /backend/tests)
- /frontend (React/Vite app, tests under /frontend)
- /ops      (docker-compose.ci.yml, schemathesis.yaml if used)
- CI files in .github/workflows/

Deliverables (all must be committed in PR)

1) Tests
   - Backend: unit + integration (pytest), property-based tests (Hypothesis).
   - Frontend: unit (Vitest + Testing Library), E2E (Playwright) incl. basic a11y checks (axe).
   - API contract tests: choose Schemathesis (OpenAPI-based) OR Pact (CDC); wire into CI.
2) Coverage & quality
   - Backend: pytest-cov with branch coverage; fail under 85.
   - Frontend: Vitest coverage thresholds (lines, functions, statements, branches ≥ 85).
   - Optional, behind flag: mutation testing (mutmut for backend; StrykerJS for FE/Vitest).
3) Observability
   - Structured JSON logging with secret/PII redaction filter.
   - OTel logging integration so every request log contains trace_id & span_id.
   - Caplog tests proving redaction and presence of trace/span IDs.
4) CI/CD (GitHub Actions)
   - Separate jobs: backend tests, frontend tests, integration tests (Docker), contract tests.
   - Upload artifacts: JUnit XML, coverage XML/HTML, Playwright trace/HTML report.
   - Concurrency guard (per-branch) to prevent overlapping deploys.
   - On PR: deploy frontend to Netlify **Deploy Preview** via CLI; run Playwright E2E against preview URL; gate merge on E2E pass.
   - On main: deploy backend to Render via Deploy Hook; deploy frontend to Netlify production.
5) Docs & scripts
   - TESTING_SUMMARY.md: baseline, notable gaps, how to run tests locally, where reports live.
   - Makefile or scripts to run full local suite and CI-parity commands.

Acceptance Criteria (Definition of Done)

- All test jobs pass in CI. Backend & frontend each meet line+branch ≥ 85% and gate the build.
- E2E Playwright suite passes against Netlify Deploy Preview (PRs).
- Contract tests pass (Schemathesis or Pact).
- JSON logs include trace_id + span_id; redaction tests pass.
- CI stores JUnit + coverage + Playwright artifacts.
- On main, successful gated deploys: Render (backend) + Netlify production (frontend).

Implementation Steps (execute in order)
A) Baseline

1. Add pytest + plugins; Hypothesis; pytest-xdist; pytest-timeout; pytest-randomly; (optional) pytest-rerunfailures for known flakes.
2. Add Vitest + RTL + Playwright (+ @axe-core/playwright) in /frontend.
3. Add .coveragerc with branch=True and report settings; set --cov-fail-under=85 in pytest config.
4. Set vitest.config.ts coverage.thresholds.* = 85; enable perFile thresholds.
   B) Backend tests (breadth → depth)
5. Unit tests for public functions/classes (@pytest.mark.parametrize).
6. Property-based tests for core pure logic (Hypothesis).
7. Integration tests with PostgreSQL (service container or docker compose); mark as `integration`.
   C) Frontend tests
8. Vitest unit tests using Testing Library’s getByRole-first approach.
9. Playwright E2E: happy-path flows; add axe scan to each critical page.
   D) Observability
10. Add JSON logging (python-json-logger or structlog) with redaction filter; include request ids.
11. OTel logging integration to inject trace_id/span_id; assertions via caplog.
    E) Contract testing (pick one or both)
- Schemathesis: point to ops/schemathesis.yaml (OpenAPI); run against local container in CI.
- Pact (FE consumer vs BE provider): generate pact files in FE job; verify in BE job.
  F) CI/CD wiring (see workflow below)
- Jobs: backend, frontend, integration, contracts, netlify_preview, e2e, deploy_backend, deploy_frontend.
- Artifacts uploaded for every job; concurrency group per ref to avoid overlapping deploys.
  G) Optional (flagged)
- Mutation testing (mutmut backend; StrykerJS + Vitest runner frontend). Store reports.
