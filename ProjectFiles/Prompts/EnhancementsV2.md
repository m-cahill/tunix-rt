Great — I’ve taken your **Enhancements Prompt** and extended it so it’s **AI-agent-ready**, fully aligned with the testing prompt style we just refined. The goal here is to make enhancements unambiguous, enforceable in CI/CD, and structured so an AI can implement them in a fresh repo with minimal human oversight.

Here’s the improved version:

* * *

Cursor AI Enhancement Prompt (Full-Stack, AI-Agent Ready)
=========================================================

**Title:** End-to-End Enhancements for Reliability, Auditability, and AI Developer Support

**Scope:**

* Backend: Python 3.11+ (FastAPI, pytest, SQLAlchemy, etc.)

* Frontend: React + Vite + TypeScript (Vitest, Playwright)

* CI/CD: GitHub Actions, deploy to **Netlify (frontend)** + **Render (backend)**

* Infrastructure: Docker/Compose available

* Observability: JSON logs + OpenTelemetry (trace_id, span_id, request correlation)

* Coverage Gate: 85% line + branch minimum (backend + frontend)

* * *

Objectives
----------

Upgrade the project to **enterprise-grade robustness** by adding **testing, observability, static analysis, security, documentation, and audit tooling**. All enhancements must be **AI-friendly**, meaning:

* Modularized config files (`.coveragerc`, `pyproject.toml`, `vitest.config.ts`, `.eslintrc.js`, `.pre-commit-config.yaml`).

* Pre-commit hooks and CI enforce quality gates automatically.

* Clear, structured logs and reports for downstream AI parsing.

* Documentation and metadata auto-generated and version-controlled.

* * *

Deliverables
------------

### 1. **Comprehensive Testing**

* Backend: pytest unit + integration tests, Hypothesis for property-based tests, contract tests (Schemathesis or Pact).

* Frontend: Vitest unit tests, Playwright E2E tests against Netlify preview deploys.

* Coverage: enforce ≥85% line + branch coverage, fail CI below threshold.

* Reports: store JUnit XML + coverage XML/HTML + Playwright traces.

### 2. **Static Analysis & Code Quality**

* Python: Black, Ruff, isort, MyPy, Radon (complexity), Bandit (security).

* Frontend: ESLint, Prettier, TypeScript strict mode, npm audit.

* Pre-commit hooks run all formatters + linters; fail fast on violations.

### 3. **Debugging & Observability**

* Add JSON logging with secret/PII redaction filter.

* Integrate OpenTelemetry: propagate `trace_id` + `span_id` through API → DB → logs.

* Caplog tests to confirm trace context and redaction.

* Expose `/metrics` endpoint (Prometheus format).

### 4. **Security & Dependency Hygiene**

* Python: Safety or pip-audit for dependency scanning.

* Node: `npm audit --production`.

* CI job blocks merges on high-severity issues.

* Add Dependabot or Renovate for auto dependency updates.

### 5. **Documentation & Audit Trails**

* Auto-generate API docs (FastAPI’s OpenAPI + Sphinx autodoc).

* Frontend: Storybook or equivalent for UI components.

* Docstring enforcement with pydocstyle.

* Changelog generation (Conventional Commits + semantic-release).

* CI auto-builds HTML docs + publishes to Netlify (docs subsite).

### 6. **CI/CD Integration**

* GitHub Actions workflow with jobs: `lint`, `backend-tests`, `frontend-tests`, `contracts`, `preview-deploy`, `e2e`, `deploy`.

* PRs: Preview deploys to Netlify, Playwright tests against preview URL.

* Main: Deploy to Render (backend) + Netlify (frontend).

* Concurrency guard: cancel in-progress runs on new commits to same branch.

* Artifact upload for all test reports.

### 7. **AI Developer Support Enhancements**

* Structured failure outputs: test + lint logs in machine-readable JSON.

* Version-controlled baseline metrics (`TESTING_SUMMARY.md`, `QUALITY_SUMMARY.md`).

* Explicit config stubs (`.coveragerc`, `vitest.config.ts`, `.pre-commit-config.yaml`) committed.

* Small, modular repos: backend/, frontend/, ops/ with Dockerized integration tests.

* Pre-wired scripts in `Makefile` (e.g., `make test`, `make lint`, `make e2e`).

* * *

Acceptance Criteria (must be green in CI)
-----------------------------------------

* All tests pass with ≥85% line + branch coverage.

* Pre-commit + CI lint/type/security checks pass.

* Netlify preview deploys run successfully on PRs; Playwright E2E suite passes.

* JSON logs include trace_id/span_id and redact secrets.

* Docs + changelog auto-generated and deployed.

* CI artifacts include JUnit, coverage, complexity, lint, and security reports.

* * *

Phase Plan (E2E-Testable Milestones)
------------------------------------

1. **Phase 0 – Baseline**: Add pre-commit + lint/type checks + CI skeleton.

2. **Phase 1 – Backend tests + coverage gates**: pytest, Hypothesis, contracts, 85% enforced.

3. **Phase 2 – Frontend tests + coverage gates**: Vitest, Playwright PR previews, axe a11y checks.

4. **Phase 3 – Observability**: JSON logs, OTel trace propagation, Prometheus metrics, caplog tests.

5. **Phase 4 – Security & audit**: Bandit, Safety, npm audit, Dependabot.

6. **Phase 5 – Docs & changelog**: Sphinx, Storybook, semantic-release, Netlify docs deploy.

7. **Phase 6 – Final CI/CD hardening**: full workflow with concurrency, artifact uploads, structured reports.

* * *

This prompt is **tight, modular, and AI-ready**: an agent can drop it into a new repo and iteratively implement it phase by phase, with each milestone fully verifiable end-to-end. It also aligns with your testing prompt theme.

* * *

Would you like me to also create a **ready-to-use repo skeleton** (backend/, frontend/, ops/, config files, CI workflow) that matches this prompt, so you can hand it straight to Cursor AI for implementation?
