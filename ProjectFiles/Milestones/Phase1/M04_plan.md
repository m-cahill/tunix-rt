Implement Milestone M4 for m-cahill/tunix-rt.

CONTEXT
- M3 is closed; CI is green.
- M3 audit flagged one medium-severity, pre-existing issue:
  E2E trace upload test failing in CI with `connect ECONNREFUSED ::1:8000`.
  This is not an M3 regression, but it blocks trustworthy E2E for future work.
  Source: M03_audit notes the failing E2E infrastructure issue and the ECONNREFUSED evidence. (See attached audit/summary.)

M4 NORTH STAR
Make E2E deterministic and green by ensuring:
- backend + db + frontend are started in CI,
- migrations are applied,
- Playwright waits for real readiness,
- localhost/IPv6 pitfalls are eliminated.

NO SCOPE CREEP
- Do NOT add new product endpoints.
- Do NOT refactor app architecture.
- Do NOT change coverage thresholds.
- Do NOT relax CI gates.
- Only touch e2e setup/config + minimal docs/targets.

====================================================
DELIVERABLES (Definition of Done)
====================================================

A) Fix E2E infrastructure so trace upload test passes
1) Standardize E2E base URLs to avoid IPv6 ::1 issues
- In Playwright config, use baseURL pointing to IPv4:
  - http://127.0.0.1:<frontend_port>
- Ensure any backend URL used by the app/tests is also 127.0.0.1 (not localhost).
Rationale: CI failure is ECONNREFUSED ::1:8000, a classic localhost/IPv6 mismatch.

2) Ensure backend binds to an address compatible with the URL
- When starting uvicorn for E2E, bind to either:
  - --host 127.0.0.1 (recommended if everything uses 127.0.0.1), OR
  - --host '::' (if we insist on localhost/IPv6).
Preferred: 127.0.0.1 everywhere for simplicity.

3) Start BOTH frontend and backend automatically for Playwright tests
- Use Playwright `webServer` feature and configure it as an ARRAY (multi-server):
  - Frontend server (Vite preview or dev)
  - Backend server (uvicorn)
- Configure each with:
  - command
  - url (not port) so Playwright waits until it accepts connections
  - timeout ~120s on CI
  - reuseExistingServer: !process.env.CI

IMPORTANT:
- When `webServer` is an array, explicitly set `use.baseURL` in config (required).

4) Ensure DB is ready + migrations applied before backend serves requests
- In the E2E CI job:
  - Start Postgres (docker compose or service container).
  - Wait for readiness using healthcheck (preferred) or pg_isready loop.
  - Run `alembic upgrade head` against Postgres.
- Backend webServer command should assume migrations are already done,
  OR it can run `alembic upgrade head && uvicorn ...` but only if DB readiness is guaranteed.

5) Add fail-fast smoke checks before Playwright runs (guardrail)
- Add a CI step (or script) that curls:
  - backend health endpoint (or /docs if health exists)
  - frontend root
- If either fails, print logs and exit early (actionable failure).

B) Make E2E reproducible locally (DX)
6) Add a single make target or npm script to run E2E locally
Examples:
- `make e2e` or `npm run e2e:local`
- Should:
  - bring up Postgres (compose),
  - run migrations,
  - start frontend/backend via Playwright webServer,
  - run Playwright tests.

C) Pass criteria
7) CI E2E job is GREEN and remains stable
- The formerly failing trace upload test passes.
- Run CI at least 2 additional times (rerun workflow) to confirm stability.
- No reliance on “retries” to mask failures (keep retries minimal/0 unless strictly necessary).

====================================================
PHASED DELIVERY PLAN
====================================================

Phase 0 — Baseline gate (mandatory)
- Pull latest main
- Confirm current CI is green
- Reproduce the failing E2E locally if possible OR inspect the CI e2e job to identify what starts/stops.

Phase 1 — Fix IPv6/localhost mismatch (fast win)
- Find anywhere using `localhost:8000` in e2e config/tests/app config.
- Replace with `127.0.0.1:8000` (or align uvicorn host).
- Ensure frontend baseURL uses 127.0.0.1 too.

Phase 2 — Add Playwright webServer orchestration (frontend + backend)
- Update playwright.config.ts:
  - webServer: [ {Frontend}, {Backend} ]
  - use.baseURL explicitly set (required when webServer is array).
- Prefer:
  - frontend: `npm run build && npm run preview -- --host 127.0.0.1 --port 4173`
  - backend: `uvicorn ... --host 127.0.0.1 --port 8000`
  Adjust ports to match repo conventions.

Phase 3 — DB readiness + migrations in CI e2e job
- Ensure Postgres is started for e2e job.
- Add readiness wait (healthcheck or pg_isready loop).
- Run `alembic upgrade head` (Postgres) before starting backend server.

Phase 4 — Fail-fast smoke checks + better diagnostics
- Add a script `scripts/e2e_sanity_check.sh` (or inline CI) that:
  - curls backend and frontend
  - prints `docker compose ps`, backend logs, etc on failure.

Phase 5 — Local DX target + docs
- Add `make e2e` (or npm script).
- Add a short README section: “Running E2E locally”.

====================================================
FILES YOU WILL LIKELY TOUCH
====================================================
- e2e/playwright.config.ts (or equivalent)
- .github/workflows/ci.yml (e2e job steps)
- docker-compose.yml (optional: healthcheck for postgres)
- backend startup command/scripts (only if needed)
- README.md (small E2E section)
- Makefile/package.json scripts (one new target)

====================================================
GUARDRAILS
====================================================
- Keep changes minimal and reviewable.
- Prefer deterministic waits (healthchecks / Playwright url wait) over `sleep`.
- Do not add new endpoints for “test reset”.
- Do not lower any coverage thresholds.
- Do not introduce flaky retries as the primary fix.

====================================================
VERIFICATION CHECKLIST (must pass before merge)
====================================================

Local:
1) E2E passes: `make e2e` (or `npm run e2e:local`)
2) Backend tests still pass
3) Frontend tests still pass

CI:
4) E2E job passes on push
5) Rerun workflow twice: still passes (stability confirmation)

====================================================
COMMIT STYLE
====================================================
Use Conventional Commits, suggested commits:
- fix(e2e): start frontend+backend via playwright webServer and standardize baseURL
- fix(ci): ensure postgres readiness + run migrations before e2e
- chore(docs): document running e2e locally
