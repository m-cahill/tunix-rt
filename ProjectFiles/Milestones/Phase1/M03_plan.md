Implement Milestone M3 for m-cahill/tunix-rt.

Context:
- M2 is complete and we’ve officially called it done.
- M2 audit identified 4 low-severity opportunities to address next:
  (1) Frontend coverage artifacts not actually generated,
  (2) Migration revision ID hardcoded as '001' (we should stop manual numbering),
  (3) Trace UI lacks frontend unit tests,
  (4) DB pool settings exist but not applied, and no created_at index.
- M3 should be “Trace System Enhancements” (hardening + DX + perf), not new features.

M3 North Star:
Raise confidence + maintainability of the trace subsystem (DB, CI, frontend tests) with minimal product scope changes.

====================================================
M3 DELIVERABLES (Definition of Done)
====================================================

A) Backend hardening (small, measurable)
1) Apply DB pool config to create_async_engine
- In backend/tunix_rt_backend/db/base.py, wire these settings into create_async_engine:
  pool_size, max_overflow, pool_timeout (settings already exist).
- Keep defaults conservative; ensure values are validated (positive ints).

2) Add a created_at index for list performance
- Create a new Alembic migration that adds an index on traces.created_at.
- Keep it cross-db friendly (works under SQLite smoke test + Postgres in dev).
- Do NOT rewrite the existing 001 migration or renumber history. Add a new migration.

3) Alembic revision ID policy (no churn)
- Keep existing revision '001' as-is (don’t break existing DBs).
- Add a short doc note + team policy: all future migrations use Alembic-generated IDs by default (no manual numbering).
- Optional: enforce via Makefile target or docs snippet (no tooling enforcement required yet).

B) Frontend correctness + coverage realism
4) Add frontend unit tests for Trace UI (2–3 tests)
- Add tests covering:
  - “Load Example” populates textarea
  - “Upload” success: calls POST, renders returned id
  - “Fetch” success: calls GET, renders pretty JSON
- Use existing vitest + RTL patterns in the repo; keep tests fast/deterministic.

5) Fix/confirm frontend coverage artifact generation in CI
- Ensure `npm run test -- --coverage` produces a coverage directory.
- Ensure CI uploads the correct directory as an artifact.
- If coverage is being generated in an unexpected path, align:
  - Vitest config (coverage.reportsDirectory)
  - CI artifact upload path
- Keep existing coverage thresholds (do not lower). Only adjust if artifacts are missing.

C) DX improvements (tiny)
6) Add curl examples + DB troubleshooting to README
- Add minimal curl examples for:
  - POST /api/traces
  - GET /api/traces/{id}
  - GET /api/traces?limit=&offset=
- Add a short “DB Troubleshooting” section:
  - docker compose ps
  - how to check Postgres is up
  - alembic upgrade command

====================================================
PHASED DELIVERY (keep PR reviewable)
====================================================

Phase 0 — Baseline gate (must do first)
- Pull latest main
- Run all tests locally (backend, frontend, e2e)
- Confirm CI is currently green before edits

Phase 1 — Backend pool config + tests
- Wire pool params into create_async_engine
- Add/adjust settings validation if needed
- Run backend tests + type checks

Phase 2 — created_at index migration
- Create a new migration (do not modify 001)
- Ensure `alembic upgrade head` still passes with SQLite in CI
- Run local Postgres migration too (docker compose up)

Phase 3 — Frontend unit tests (Trace UI)
- Add 2–3 tests as described
- Ensure no flakiness: all fetch calls mocked, deterministic

Phase 4 — Coverage artifact fix
- Run locally: `npm run test -- --coverage` and confirm coverage output
- Align CI artifact upload path
- Confirm CI shows coverage artifact after run

Phase 5 — Docs / DX
- Add curl examples and troubleshooting to README
- Keep docs concise

====================================================
GUARDRAILS (do not violate)
====================================================
- Do NOT add new endpoints (e.g., DELETE, filtering) in M3 — keep it hardening only.
- Do NOT refactor App.tsx into a hook unless it’s necessary to test it cleanly.
- Do NOT change CI permissions.
- Do NOT lower coverage thresholds; fix generation/upload instead.
- Do NOT rewrite Alembic history (no renaming 001).

====================================================
VERIFICATION CHECKLIST (must pass before merge)
====================================================

Local:
1) Backend: ruff + mypy + pytest all green
2) Frontend: vitest unit tests green
3) Frontend: `npm run test -- --coverage` creates coverage output
4) E2E: Playwright tests green
5) DB: `alembic upgrade head` works against:
   - SQLite (same as CI smoke)
   - Postgres (docker compose)

CI:
- All jobs green
- Frontend coverage artifact is present and contains expected files
- Migration smoke test passes

====================================================
COMMIT STYLE
====================================================
Use Conventional Commits, suggested sequence:
- perf(db): apply async engine pool settings
- perf(db): add created_at index migration
- test(frontend): add trace UI unit tests
- fix(ci): ensure vitest coverage artifacts are generated and uploaded
- docs: add curl examples and db troubleshooting
