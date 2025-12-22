Below is a **Cursor-ready M10 prompt** that builds directly on the **M09 audit + summary** (low-risk improvements, app-layer hardening, coverage recovery via refactor—not hero tests).  
I’m also anchoring a few implementation choices to upstream docs for correctness (FastAPI validation, timezone-aware datetimes, coverage exclusions, SQLAlchemy async patterns). ([FastAPI][1])

---

## Prompt to handoff to Cursor — **M10: App Layer Refactor + Determinism Guardrails + Small Perf/Deprecation Fixes**

### Context

M09 delivered the reproducible SFT loop v1 and kept CI green, but the audit flagged several **low-risk, high-leverage improvements**:

* `app.py` grew and is doing too much; refactor to thin endpoints and reusable helpers/services. 
* Export format validation is still manual; should be typed (Literal/Enum) for automatic validation + OpenAPI correctness.  ([GitHub][2])
* `datetime.utcnow()` usage should be replaced with timezone-aware UTC to avoid deprecation warnings in newer Python.  ([Discussions on Python.org][3])
* Batch endpoint does per-row `refresh()` (N queries); optimize or skip refresh safely. 
* Training scripts are intentionally outside backend coverage; add lightweight “script validation” tests (dry-run / config parsing) without pulling training runtime into main CI. 

### M10 Goal

**Reduce `app.py` complexity**, improve testability/coverage *organically*, fix deprecations, and apply a small performance improvement—**without changing M09 behavior**.

### Non-goals (explicit)

* No new DB tables/migrations.
* No “real” Tunix training in default CI.
* No optional dependency coupling (keep Tunix/UNGAR optional as in M09).

---

# Phase 0 — Baseline Gate (must do first)

1. Branch from `main` at current green commit.
2. Run locally:

   * `backend`: ruff + mypy + pytest + coverage gate
   * `frontend`: tests
   * `e2e`: playwright
3. Create `docs/M10_BASELINE.md` with commit SHA + pass/fail checklist.

**Acceptance:** baseline is reproducible and green before changes.

---

# Phase 1 — Export Format: Typed Validation (Literal/Enum)

### Problem

`format` query param is manually validated in `app.py` (string list check). This is brittle and duplicates validation that FastAPI/Pydantic can provide. 

### Implementation

1. In `backend/tunix_rt_backend/schemas/dataset.py`:

   * Introduce:

     * `ExportFormat = Literal["trace", "tunix_sft", "training_example"]`
     * (Alternative acceptable: `Enum`, but if using enum, ensure it’s a *real* enum for validation, not just OpenAPI decoration.) ([GitHub][2])
2. Update export endpoint signature to use `format: ExportFormat = "trace"`.
3. Remove manual validation block from `app.py`.
4. Update tests:

   * Ensure invalid format returns 422 from FastAPI validation (not manual 400/422).
5. Update docs where export formats are described (keep behavior identical).

**Acceptance**

* Export endpoint behavior unchanged for valid formats.
* Invalid format produces **422 validation error** automatically.
* OpenAPI shows enum-like options for `format`. ([FastAPI][1])

---

# Phase 2 — `app.py` Refactor: Thin Controllers + Services

### Problem

M09 expanded `app.py` substantially; branch coverage drop is mostly in this file. The audit recommends extracting logic into helpers/services for maintainability and testability. 

### Implementation (minimal, pragmatic)

Create:

* `backend/tunix_rt_backend/services/traces_batch.py`
* `backend/tunix_rt_backend/services/datasets_export.py`

Move logic out of `app.py`:

1. **Batch import**

   * Validation (empty list, max batch size, per-trace validation failures)
   * Building Trace model objects
   * Commit strategy (transaction all-or-nothing)
2. **Dataset export**

   * Exporting “trace”, “tunix_sft”, “training_example”
   * Formatting selection lives in service, not controller

Keep `app.py` responsibilities:

* Parse request/params (FastAPI)
* Call service
* Return response
* Map known errors to HTTPException

### Tests strategy (enterprise-grade, fast)

* Add **unit tests** for service functions (pure validation + conversion).
* Keep existing **integration tests** for endpoints.
* Do not inflate tests with exotic failure simulation—target meaningful branches.

**Acceptance**

* `app.py` shrinks meaningfully (you should see the diff).
* Coverage for moved logic becomes easier to raise without “coverage hacks.”
* All existing behavior preserved.

---

# Phase 3 — Batch Endpoint Perf: Remove N refresh() calls

### Problem

The batch endpoint commits and then refreshes each trace one-by-one. At max (1000), that’s 1000 SELECTs. 

### Implementation options (pick safest)

**Preferred (simple + deterministic):**

* After commit, do a single bulk SELECT `WHERE id IN (...)` and build response from those rows.

(If needed, use SQLAlchemy patterns that refresh existing instances via a bulk query; see common approach using query re-load / populate-existing patterns.) ([Stack Overflow][4])

**Constraints**

* Do NOT do concurrent refreshes on the same `AsyncSession`. SQLAlchemy explicitly warns that `AsyncSession` is mutable/stateful; concurrency needs separate sessions. ([SQLAlchemy][5])

### Add a micro perf test (optional, not CI-blocking)

* A local script or doc snippet: import 100 traces and record timing.
* No flaky perf asserts in CI.

**Acceptance**

* Batch endpoint response fields are identical.
* Query count reduced dramatically for large batches.
* Tests updated if necessary.

---

# Phase 4 — Deprecation Cleanup: timezone-aware UTC datetimes

### Problem

`datetime.utcnow()` is deprecated / discouraged because it produces a naive datetime; prefer timezone-aware UTC.  ([Discussions on Python.org][3])

### Implementation

1. Update `backend/tunix_rt_backend/training/schema.py`

   * Replace `datetime.utcnow()` with `datetime.now(UTC)` (or `datetime.now(datetime.UTC)` depending on import style).
2. Ensure serialization is consistent with existing JSON output.

**Acceptance**

* No deprecation warnings in tests on modern Python.
* JSON output remains stable (update snapshot tests if necessary).

---

# Phase 5 — Training Script “Validation Tests” (no heavy runtime)

### Problem

Training scripts are intentionally out-of-package and untested; add minimal test coverage to prevent regressions without running training. 

### Implementation

1. Ensure scripts support:

   * `--dry-run` that validates:

     * config YAML loads
     * required fields present
     * output directories computed
     * manifest schema validation
     * exits 0 without training execution
2. Add `backend/tests/test_training_scripts_smoke.py`:

   * Invoke scripts in subprocess with `--dry-run`
   * Use tiny config fixture
   * Assert exit code 0 and expected stdout markers

**Acceptance**

* Default CI remains fast (dry-run only).
* No optional deps required.
* Training scripts become safer to iterate on.

---

# Phase 6 — Docs + Guardrails

1. Update `docs/adr/ADR-005` only if the coverage philosophy changed (it shouldn’t).
2. Add `docs/M10_GUARDRAILS.md`:

   * “Thin controller” rule for endpoints
   * “Typed params” rule (Literal/Enum)
   * Batch endpoint limits + perf notes
   * “No concurrent use of AsyncSession” note (with rationale). ([SQLAlchemy][5])
3. Update API docs for export `format` param to reflect typed enum.

---

## Definition of Done (M10)

* ✅ All CI jobs green (backend 3.11/3.12, frontend, e2e).
* ✅ `app.py` thinner; main logic moved into services.
* ✅ Export format validation is typed (Literal/Enum), manual validation removed. ([FastAPI][1])
* ✅ Batch endpoint no longer does N refresh() calls.
* ✅ No datetime UTC deprecation warnings. ([Discussions on Python.org][3])
* ✅ Added dry-run script smoke tests (fast, deterministic).
* ✅ Added M10 baseline + guardrails docs.

---

## Suggested commit sequence (keeps diffs reviewable)

1. `chore(m10): baseline doc`
2. `refactor(m10): typed export format + remove manual validation`
3. `refactor(m10): extract batch + export services`
4. `perf(m10): optimize batch refresh`
5. `fix(m10): timezone-aware UTC datetime`
6. `test(m10): training script dry-run smoke tests`
7. `docs(m10): guardrails + updates`

---

If you want, I can also produce the **exact file skeletons** for the new `services/` modules and the **service-level unit test templates** (so Cursor can paste them in and fill in logic).

[1]: https://fastapi.tiangolo.com/tutorial/query-params-str-validations/?utm_source=chatgpt.com "Query Parameters and String Validations"
[2]: https://github.com/tiangolo/fastapi/discussions/6898?utm_source=chatgpt.com "Query params did not validate enum values #6898"
[3]: https://discuss.python.org/t/why-is-datetime-utcnow-deprecated/86868?utm_source=chatgpt.com "Why is datetime.utcnow deprecated? - Python Discussions"
[4]: https://stackoverflow.com/questions/68921946/is-there-a-way-to-refresh-an-array-of-sqlalchemy-objects?utm_source=chatgpt.com "Is there a way to refresh an array of SQLAlchemy objects?"
[5]: https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html?utm_source=chatgpt.com "Asynchronous I/O (asyncio) — SQLAlchemy 2.0 ..."
