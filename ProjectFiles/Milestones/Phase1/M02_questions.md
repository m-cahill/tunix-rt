# M2 Clarifying Questions

**Asked:** 2025-12-20  
**Milestone:** M2 - Trace Storage & Retrieval

---

## 1. Database Connection & Session Management

### 1a) Async vs Sync SQLAlchemy?

The plan mentions "SQLAlchemy 2.x style + FastAPI dependency injection using a dependency that yields a session."

- Should we use **async SQLAlchemy** (`AsyncSession` + `asyncpg` driver) or **sync** (`Session` + `psycopg2`)?
- Current `DATABASE_URL` in docker-compose uses `postgresql+asyncpg://...` suggesting async
- FastAPI supports both, but async is more consistent with FastAPI's async nature

**Recommendation:** Use async SQLAlchemy for consistency with FastAPI async patterns?

### 1b) Connection pooling configuration?

- Should we configure pool size, overflow, timeouts in settings?
- Or keep defaults for M2 and defer tuning to later milestones?

---

## 2. Trace Schema Validation Strictness

The plan defines a minimal schema:
```
ReasoningTrace {
  trace_version: str
  prompt: str
  final_answer: str
  steps: list[TraceStep]
  meta: dict[str, Any] | None
}
```

### 2a) Field validation strictness?

- Should `prompt` and `final_answer` have min/max length constraints? (e.g., 1-10000 chars)
- Should we validate `trace_version` format (e.g., semver-like "1.0", "2.0")?
- Should `steps` list have min/max count constraints? (e.g., 1-1000 steps)

### 2b) TraceStep validation?

- Should `TraceStep.i` be strictly sequential (0, 1, 2...) or just unique?
- Should `TraceStep.type` be an enum or free-form string?
- Should `TraceStep.content` have length limits?

**Recommendation:** Start permissive (basic non-empty checks) and add strictness in M3 based on real usage?

---

## 3. Pagination for GET /api/traces

The plan says "Returns a small paginated list" but doesn't specify defaults.

### 3a) Pagination parameters?

- Query params: `?limit=20&offset=0` (offset-based)?
- Or cursor-based pagination: `?limit=20&after=<uuid>`?
- Default limit? (20? 50? 100?)
- Max limit? (to prevent abuse)

### 3b) Response structure?

Simple list:
```json
[
  {"id": "...", "created_at": "...", "trace_version": "..."},
  ...
]
```

Or envelope with pagination metadata:
```json
{
  "data": [...],
  "pagination": {"limit": 20, "offset": 0, "total": 150}
}
```

**Recommendation:** Offset-based with limit=20 default, max=100, simple list response for M2?

---

## 4. Frontend UI Placement & Styling

### 4a) Where in App.tsx should the Traces section go?

- Add as a third status card alongside API and RediAI health?
- Or create a new section below the health cards?
- Should it be collapsible or always visible?

### 4b) Styling approach?

- Extend existing `index.css` with new trace-specific classes?
- JSON display: use `<pre>` with syntax highlighting, or plain `<pre>`?
- Should we add a simple CSS library (e.g., classless.css) or keep it vanilla?

**Recommendation:** Add as a new section below health cards, vanilla CSS, plain `<pre>` for JSON display?

---

## 5. Example Trace Structure

The plan mentions "Load example" button should prefill a known-good trace.

### 5a) What should the example trace contain?

- How many steps? (3-5 for clarity?)
- What should the prompt/answer be? (generic example? math problem? coding task?)
- Should `meta` be populated or null in the example?

**Recommendation:** A simple 3-step reasoning example like "What is 2+2?" with clear thinking steps?

---

## 6. Alembic Configuration

### 6a) Migration message pattern?

- Should migration messages follow a convention? (e.g., "create_traces_table", "add_traces_table")?
- Include ticket/milestone numbers? (e.g., "M2: create traces table")?

### 6b) Alembic target_metadata?

- Should we import all models into `env.py` for autogenerate to work?
- Or maintain a registry pattern?

**Recommendation:** Use descriptive lowercase_with_underscores for migration messages, import all models into env.py?

---

## 7. Test Database Strategy

### 7a) Test DB approach?

For backend unit tests:
- Option A: SQLite in-memory (`:memory:`) - fast, no cleanup
- Option B: SQLite file (`test.db`) - persistent for debugging, needs cleanup
- Option C: Postgres test container - realistic but slower

### 7b) Migration testing in CI?

The plan says "Run `alembic upgrade head` against a SQLite URL in CI"
- Should this be a separate CI job or part of the backend job?
- Should we test migrations both up AND down (`alembic downgrade -1`)?

**Recommendation:** SQLite in-memory for unit tests, separate migration smoke test in backend CI job, test upgrade only for M2?

---

## 8. Frontend Coverage Configuration

The audit recommends adding vitest coverage with "initial gate (e.g., 60%)".

### 8a) Coverage provider?

- Use `v8` (recommended by vitest, faster) or `istanbul`?

### 8b) Initial thresholds?

- Lines: 60%? 70%?
- Branches: 50%? 60%?
- Should we fail CI on coverage drop, or just measure for M2?

**Recommendation:** v8 provider, 60% lines / 50% branches, warn-only (not blocking) for M2?

---

## 9. Trace Size Limits

The plan mentions "max bytes" validation for trace payloads.

### 9a) What should TRACE_MAX_BYTES default to?

- 1MB? (allows ~1M characters of text)
- 10MB? (allows larger traces with verbose steps)
- Configurable via env var?

### 9b) Should we validate individual field sizes too?

- Or just the total JSON payload size?

**Recommendation:** TRACE_MAX_BYTES=1048576 (1MB), validate total payload size only for M2?

---

## 10. Error Handling & User Feedback

### 10a) Validation error response format?

When schema validation fails, should we return:
- Simple message: `{"detail": "Invalid trace format"}`
- Detailed Pydantic errors: `{"detail": [{"loc": ["steps", 0, "i"], "msg": "..."}]}`

### 10b) Frontend error display?

- Show raw error JSON in the UI?
- Parse and format error messages nicely?

**Recommendation:** Return detailed Pydantic errors (FastAPI default), display raw in `<pre>` for M2, format nicely in M3?

---

## 11. Database Migrations & docker-compose

### 11a) Should migrations run automatically in docker-compose?

- Option A: Add a migration init container / script that runs `alembic upgrade head` before backend starts
- Option B: Developers run migrations manually locally
- Option C: Backend runs migrations on startup (not recommended for production patterns)

### 11b) Local development migration workflow?

- Should README include step-by-step migration commands?
- Should Makefile include migration helpers? (`make db-migrate`, `make db-upgrade`)

**Recommendation:** Manual migrations for M2 (keep it explicit), add Makefile helpers, document in README?

---

## 12. Trace Retrieval: Payload Inclusion

The plan says:
- `GET /api/traces/{id}` → Returns full payload + metadata
- `GET /api/traces` → Returns list without payload

### 12a) Should GET /api/traces support `?include_payload=true`?

- Or should clients always fetch by ID if they need the full trace?

**Recommendation:** No include_payload param for M2 - keep list lightweight, fetch by ID for full data?

---

## 13. Timestamp Format & Timezone

### 13a) created_at serialization?

- ISO8601 with timezone: `2025-12-20T10:30:00+00:00`
- Or UTC marker: `2025-12-20T10:30:00Z`
- Or Unix timestamp: `1734693000`

### 13b) Should we add updated_at?

- Or keep it minimal for M2 (created_at only)?

**Recommendation:** ISO8601 UTC (`...Z` format), created_at only for M2?

---

## 14. Quick Win Patches Priority

The M1 audit lists 5 quick wins. Should all 5 be in M2 Phase 1, or should we prioritize?

**Priority order?**
1. Add rediai_health_path validator (security/correctness)
2. Add vitest coverage config + CI (testing)
3. Add JSDoc to api client (DX)
4. Cache security tools (CI performance)
5. Add frontend coverage gate (quality)

Or apply all 5 in Phase 1 since they're all small?

**Recommendation:** Apply all 5 in Phase 1 - they're all <15 min each and set good foundation?

---

## 15. Phase Delivery Confirmation

The plan outlines 5 phases. Should each phase be:
- **Separate commits** on a feature branch (all pushed together at the end)?
- **Separate PRs** (5 small PRs for fast review)?
- **Single PR** with logical commits?

**Recommendation:** Single feature branch, logical commits per phase, one PR at the end (like M1)?

---

## SUMMARY - Decisions Needed

**High Priority (affects architecture):**
1. Async vs sync SQLAlchemy (Q1a)
2. Trace validation strictness (Q2a, Q2b)
3. Pagination approach (Q3a, Q3b)
4. Test DB strategy (Q7a)
5. TRACE_MAX_BYTES default (Q9a)
6. Migration workflow (Q11a, Q11b)

**Medium Priority (affects implementation):**
7. Frontend UI placement (Q4a)
8. Example trace content (Q5a)
9. Frontend coverage thresholds (Q8b)
10. Error response format (Q10a)

**Low Priority (preferences/defaults):**
11. Alembic message pattern (Q6a)
12. Timestamp format (Q13a)
13. Quick win patches priority (Q14)
14. Phase delivery approach (Q15)

---

**Ready for your answers!** I'll finalize the implementation plan and create todos once you respond.

