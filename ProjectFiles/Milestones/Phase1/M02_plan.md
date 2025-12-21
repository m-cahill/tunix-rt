Implement Milestone M2 for repo m-cahill/tunix-rt.

Context:
- M1 is complete and pushed. CI is green.
- M1 audit explicitly recommends several small patches to apply in M2:
  (1) Frontend coverage measurement, (2) cache security tool installs in CI,
  (3) add validator for rediai_health_path, (4) add JSDoc to frontend API client,
  (5) add frontend coverage to CI.
- M1 audit also outlines M2 as “Trace Storage & Retrieval” with minimal DB integration.
(See ProjectFiles/Milestones/Phase1/M01_audit.md and M01_summary.md.)

M2 North Star:
Make “Reasoning Trace” a first-class artifact: validate → persist → retrieve → view, end-to-end, with tests and CI gates, without scope creep into evaluation/judging.

========================================
M2 DELIVERABLES (Definition of Done)
========================================
A) Backend: DB + migrations + trace API
1) Add Alembic migrations and a minimal SQLAlchemy model for Trace storage.
2) Implement endpoints:
   - POST /api/traces
     * Accepts a ReasoningTrace JSON payload (schema-validated).
     * Persists it to DB.
     * Returns { id, created_at } (and optionally trace_version).
   - GET /api/traces/{id}
     * Returns the stored payload + metadata.
   - GET /api/traces
     * Returns a small paginated list: [{ id, created_at, trace_version }] (no full payload).
3) Add guardrails:
   - Validate payload size (e.g., max bytes) and reject with 413.
   - Validate trace_version and required fields.
   - Keep the existing RediAI health integration unchanged.
4) Add a settings validator ensuring rediai_health_path starts with "/".

B) Frontend: trace upload/view (minimal UI)
1) Add a small “Traces” section in App:
   - Textarea for JSON
   - “Load example” button (prefills textarea with a known-good example trace)
   - “Upload” button -> POST /api/traces -> shows returned id
   - “Fetch” button -> GET /api/traces/{id} -> renders JSON (pretty)
   Keep UI minimal; do not add routing unless truly necessary.
2) Extend the typed API client (frontend/src/api/client.ts):
   - createTrace(trace: ReasoningTrace) -> {id, created_at}
   - getTrace(id) -> TraceResponse
   - listTraces(params) -> list response
3) Add JSDoc comments to exported client functions (audit quick win).

C) Tests: keep it “small and tested”
1) Backend unit tests (pytest):
   - POST then GET trace works (happy path).
   - Invalid schema rejected (400).
   - Oversized payload rejected (413).
   - list endpoint returns expected shape.
   - DB session dependency is overridden for tests (deterministic).
2) Migration smoke test in CI:
   - Run `alembic upgrade head` against a SQLite URL in CI to ensure migrations apply cleanly.
3) Frontend unit tests (vitest + RTL):
   - Upload success path (mock fetch).
   - Fetch/display trace path.
4) E2E Playwright:
   - New E2E test: “Load example -> Upload -> Fetch -> Displays JSON”
   - Keep existing “API: healthy” assertion.
   - RediAI must remain deterministic in CI (REDIAI_MODE=mock).

D) CI improvements (audit quick wins)
1) Frontend coverage measurement:
   - Enable vitest coverage (v8 provider) and run with `--coverage`.
   - Upload coverage artifact.
   - Add a modest initial coverage gate (e.g., 60%) so it’s measurable and trends upward.
2) Cache security tool installs in CI:
   - Cache pip downloads / wheels for pip-audit + cyclonedx-bom in security job to reduce runtime.

E) Local dev updates
1) docker-compose should now support backend connecting to postgres for trace storage.
2) Update .env.example with DATABASE_URL and any new limits (TRACE_MAX_BYTES).
3) Update README + tunix-rt.md with:
   - How to run migrations locally
   - Trace API usage examples
   - How to use the trace UI
   - How to run the new E2E test

========================================
IMPLEMENTATION DETAILS / GUIDELINES
========================================

Backend DB approach (keep minimal):
- Use SQLAlchemy 2.x style + FastAPI dependency injection using a dependency that yields a session (FastAPI supports dependencies with yield).
- Use Alembic autogenerate for migrations and commit the migration scripts.
- DB schema:
  traces table:
    - id (uuid primary key)
    - created_at (timezone aware)
    - trace_version (text)
    - payload (JSON/JSONB)
  Keep it minimal; avoid premature indexing.

ReasoningTrace schema (M2):
- Create pydantic models in backend (e.g., trace_schema.py):
  - TraceStep { i: int, type: str, content: str }
  - ReasoningTrace { trace_version: str, prompt: str, final_answer: str, steps: list[TraceStep], meta: dict[str, Any] | None }
- Validate minimum fields and basic bounds (non-empty, positive i, etc.)
- Trace payload stored as JSON, returned as JSON.

Frontend:
- Add types mirroring backend schema (ReasoningTrace, TraceStep, etc.)
- Keep UI minimal and robust (clear error state; show ApiError message).

No scope creep:
- No evaluation metrics, no LLM-as-judge, no RediAI job submission, no auth.
- Just trace persistence + retrieval + minimal UI + tests + CI gates.

========================================
PHASED DELIVERY (small PR-sized steps)
========================================
Phase 1: Apply audit quick wins (tiny)
- Add rediai_health_path validator.
- Add vitest coverage config + CI artifact upload + initial gate.
- Add JSDoc to api client exports.
- Add cache step for pip security tools.

Phase 2: DB + migrations (backend)
- Add SQLAlchemy + Alembic setup.
- Create Trace model + initial migration.
- Add DATABASE_URL config + compose wiring.
- Add migration smoke check (alembic upgrade head) in CI.

Phase 3: Trace API endpoints + backend tests
- Implement POST/GET/GET list endpoints.
- Add validation and payload size guardrails.
- Add deterministic tests with DB override.

Phase 4: Frontend trace UI + tests
- Add Traces section with Load example / Upload / Fetch.
- Extend typed API client.
- Add unit tests.

Phase 5: E2E trace flow
- Add new Playwright test for trace upload/retrieve via UI.
- Ensure CI uses REDIAI_MODE=mock and stays deterministic.

========================================
VERIFICATION (must run locally before final push)
========================================
Backend:
- make backend-test (or equivalent) -> all pass
- alembic upgrade head (against local DB) succeeds
Frontend:
- npm test + coverage
- npm run build
E2E:
- playwright test (including new trace flow)

CI:
- All jobs green
- Frontend coverage artifact present
- Migration smoke test passes

========================================
COMMIT / PR STYLE
========================================
- Use Conventional Commits.
- Keep commits small and logically grouped by phase.
- Update docs as part of the phase they affect.
