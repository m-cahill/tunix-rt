Below are **direct answers / decisions** for each M2 question, tuned for **small + tested + maintainable**, and consistent with your current direction (asyncpg already implied, plus early RediAI integration). 

---

## 1. Database connection & session management

### 1a) Async vs sync SQLAlchemy

**Decision: Use async SQLAlchemy (`AsyncSession` + `asyncpg`) for the app runtime.** ([SQLAlchemy Documentation][1])

Why:

* SQLAlchemy’s guidance is effectively “**one AsyncSession per asyncio task**,” which maps cleanly to FastAPI request handling. ([SQLAlchemy Documentation][1])
* You’re already oriented toward `postgresql+asyncpg://...` in local stack expectations. 

**Implementation detail (important):**

* Use a FastAPI dependency that yields an `AsyncSession` (or an `asynccontextmanager`). FastAPI explicitly supports dependencies using `yield` for setup/teardown. ([FastAPI][2])
* For Alembic, use Alembic’s **async environment template** (supported pattern) so you don’t need two different DB drivers/URLs. ([Alembic][3])

### 1b) Connection pooling configuration

**Decision: Keep pool defaults in M2; expose optional env vars but do not tune yet.** ([SQLAlchemy Documentation][4])

Concretely:

* No “pool size” tuning in M2 unless a test/CI issue forces it.
* If you want a minimal “enterprise-ready” hook: add optional settings like `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT` with conservative defaults, but don’t obsess over values until load exists.

---

## 2. Trace schema validation strictness

### 2a) Field strictness

**Decision: “Permissive but safe” in M2.**
We will enforce:

* `trace_version`: **required**, `max_length` (e.g., 32–64). (No strict semver yet.)
* `prompt` and `final_answer`: required, non-empty, plus **max length** (e.g., 50k each).
* `steps`: required, **min 1**, **max N** (recommend **1000** for M2).

Primary “abuse cap” is still `TRACE_MAX_BYTES` (see Q9), but these per-field limits prevent pathological payload shapes.

### 2b) TraceStep validation

**Decision:**

* `TraceStep.i`: must be **unique** and **non-negative**; **do NOT require perfect contiguity** in M2.

  * Rationale: strict sequential validation is nice, but it tends to create friction fast. We can sort by `i` for display and enforce contiguity later if needed.
* `TraceStep.type`: **free-form string** (max len ~32/64). No enum yet.
* `TraceStep.content`: non-empty, max length (e.g., 20k).

---

## 3. Pagination for `GET /api/traces`

### 3a) Pagination parameters

**Decision: Offset-based pagination**: `?limit=20&offset=0`

* Default `limit=20`
* Max `limit=100`

### 3b) Response structure

**Decision: Use an envelope** (future-proof without being heavy):

```json
{
  "data": [{"id": "...", "created_at": "...", "trace_version": "..."}],
  "pagination": {"limit": 20, "offset": 0, "next_offset": 20}
}
```

This avoids breaking changes later if we add cursor paging or totals.

---

## 4. Frontend UI placement & styling

### 4a) Placement in `App.tsx`

**Decision: New section below the health cards**, always visible (not collapsible in M2).

### 4b) Styling

**Decision: Vanilla CSS only** (extend `index.css`).
JSON display: plain `<pre>` with wrapping; no highlighting; no new CSS libs.

---

## 5. Example trace structure

### 5a) Example trace content

**Decision: 4–5 steps, very small, and “high-level steps” (no long reasoning text).**

Example prompt idea:

* “Compute 27×19” or “Convert 68°F to °C”
  Example steps:
* “Parse task”
* “Choose formula”
* “Compute”
* “Return result”

`meta`: include a tiny object (e.g., `{ "source": "example", "tags": ["demo"] }`) to validate the optional field.

---

## 6. Alembic configuration

### 6a) Migration message pattern

**Decision: Descriptive snake_case** like:

* `create_traces_table`
  No milestone numbers in the migration name (keep milestone tracking in PR/commit messages).

### 6b) `target_metadata`

**Decision: Centralize `Base.metadata` and ensure env imports load models.**
Alembic autogenerate compares DB schema vs the `target_metadata` you provide. ([Alembic][5])

So:

* `backend/db/base.py` defines `Base = declarative_base()`
* `backend/db/models/__init__.py` imports all models
* `alembic/env.py` imports Base + models package before setting `target_metadata = Base.metadata`

---

## 7. Test database strategy

### 7a) Unit test DB approach

**Decision: SQLite file DB per test session using async driver (`sqlite+aiosqlite`).**
Rationale: stable, debuggable, and avoids some in-memory edge cases with async/multiple connections.

### 7b) Migration testing in CI

**Decision: Migration smoke test runs as a step inside the backend CI job**, and only tests:

* `alembic upgrade head` (upgrade only) in M2

We can add downgrade testing later once more than one migration exists.

---

## 8. Frontend coverage configuration

### 8a) Coverage provider

**Decision: `v8` provider** (fast, recommended). ([Vitest][6])

### 8b) Initial thresholds + blocking behavior

**Decision: Make it blocking (fails CI), but modest thresholds:**

* Lines: **60%**
* Branches: **50%**
* Statements: **60%**
* Functions: **50%**

Vitest supports global thresholds in config. ([Vitest][6])

---

## 9. Trace size limits

### 9a) Default `TRACE_MAX_BYTES`

**Decision: 1 MiB default** (`1048576`), configurable via env var.

### 9b) Field sizes vs total

**Decision: Validate total payload size + basic per-field caps** (from Q2).
Total payload size is the primary protection; per-field caps prevent single-field abuse.

---

## 10. Error handling & user feedback

### 10a) Validation error response format

**Decision: Keep FastAPI default detailed validation errors** (Pydantic-style). ([FastAPI][2])
It’s hugely helpful during early development.

### 10b) Frontend error display

**Decision: Display the raw error JSON in a `<pre>` for M2**, plus a short headline (HTTP status + message). “Pretty” formatting can wait.

---

## 11. DB migrations & docker-compose

### 11a) Auto-run migrations?

**Decision: No automatic migrations in docker-compose for M2** (manual/explicit).

### 11b) Local migration workflow

**Decision: Yes to Makefile helpers + README steps**, e.g.:

* `make db-up`
* `make db-upgrade`
* `make db-revision msg="create_traces_table"`

---

## 12. Trace retrieval: payload inclusion

### 12a) `include_payload=true`?

**Decision: No** for M2.
List stays lightweight; fetch-by-id returns the full payload.

---

## 13. Timestamp format & timezone

### 13a) `created_at` serialization

**Decision: ISO8601 UTC** (timezone-aware). Keep it simple and unambiguous.

### 13b) Add `updated_at`?

**Decision: No** in M2.

---

## 14. Quick win patches priority

**Decision: Apply all 5 early in M2 Phase 1**, in the same order proposed in the questions doc (and aligned with M1 audit recommendations).

---

## 15. Phase delivery approach

**Decision: Single feature branch, logical commits per phase, one PR at the end.**

---

### One extra note (to keep M2 “small and tested”)

Because you’re going **async + Alembic**, explicitly start from Alembic’s supported async setup (template/cookbook) rather than inventing your own—this avoids a whole class of migration/runtime mismatch issues. ([Alembic][3])

If you want, I can now generate an **updated M2 Cursor prompt** that bakes these decisions in so Cursor doesn’t have to guess.

[1]: https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html?utm_source=chatgpt.com "Asynchronous I/O (asyncio) — SQLAlchemy 2.0 ..."
[2]: https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/?utm_source=chatgpt.com "Dependencies with yield"
[3]: https://alembic.sqlalchemy.org/en/latest/cookbook.html?utm_source=chatgpt.com "Cookbook — Alembic 1.17.2 documentation"
[4]: https://docs.sqlalchemy.org/en/latest/core/pooling.html?utm_source=chatgpt.com "Connection Pooling — SQLAlchemy 2.0 Documentation"
[5]: https://alembic.sqlalchemy.org/en/latest/autogenerate.html?utm_source=chatgpt.com "Auto Generating Migrations - Alembic's documentation!"
[6]: https://vitest.dev/config/coverage?utm_source=chatgpt.com "coverage | Config"
