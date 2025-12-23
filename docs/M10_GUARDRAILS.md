# M10 Guardrails & Architectural Rules

**Date:** December 21, 2025  
**Milestone:** M10 - App Layer Refactor + Determinism Guardrails  
**Status:** **ACTIVE** - Enforce these rules in all future development

---

## Purpose

This document establishes architectural guardrails introduced in M10 to maintain code quality, testability, and prevent common pitfalls as tunix-rt scales.

**Scope:** Backend architecture, API endpoint design, service layer, database operations

---

## Guardrail 1: Thin Controller Pattern

### Rule

API endpoints in `app.py` MUST be thin controllers that:
- Parse request parameters (FastAPI)
- Delegate to service layer
- Map exceptions to HTTP status codes
- Return responses

Endpoints MUST NOT contain business logic.

### Rationale

- **Testability:** Business logic in services can be unit tested without HTTP layer
- **Reusability:** Services can be called from CLI tools, background jobs, etc.
- **Maintainability:** `app.py` remains readable at scale (hundreds of endpoints)

### Example: Compliant Endpoint

```python
@app.post("/api/traces/batch")
async def create_traces_batch_endpoint(
    traces: list[ReasoningTrace],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraceBatchCreateResponse:
    """Create multiple traces (thin controller)."""
    try:
        return await create_traces_batch_optimized(traces, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Example: Non-Compliant Endpoint ❌

```python
@app.post("/api/traces/batch")
async def create_traces_batch_endpoint(...):
    """DO NOT DO THIS - business logic in endpoint."""
    # Validation logic ❌
    if not traces:
        raise HTTPException(...)
    
    # Model creation ❌
    db_traces = [Trace(...) for trace in traces]
    
    # Database operations ❌
    db.add_all(db_traces)
    await db.commit()
    ...
```

---

## Guardrail 2: Typed Query Parameters (Literal/Enum)

### Rule

Query parameters with restricted values MUST use Pydantic `Literal` or `Enum` types instead of manual string validation.

### Rationale

- **Automatic validation:** FastAPI/Pydantic validates before endpoint code runs
- **OpenAPI accuracy:** Swagger docs show enum-like options automatically
- **Type safety:** IDE autocomplete and type checkers catch errors
- **DRY:** No duplicate validation logic in endpoints

### Example: Compliant (M10+)

```python
# In schemas/dataset.py
ExportFormat = Literal["trace", "tunix_sft", "training_example"]

# In app.py
@app.get("/api/datasets/{dataset_key}/export.jsonl")
async def export_dataset(
    dataset_key: str,
    format: ExportFormat = "trace",  # ✅ Typed parameter
) -> Response:
    # No manual validation needed - FastAPI handles it
    content = await export_dataset_to_jsonl(manifest, db, format)
    return Response(content=content, media_type="application/x-ndjson")
```

### Example: Non-Compliant (M09 pattern) ❌

```python
@app.get("/api/datasets/{dataset_key}/export.jsonl")
async def export_dataset(
    dataset_key: str,
    format: str = "trace",  # ❌ Untyped parameter
) -> Response:
    # Manual validation ❌
    if format not in ["trace", "tunix_sft", "training_example"]:
        raise HTTPException(status_code=422, detail=f"Invalid format: {format}")
    ...
```

---

## Guardrail 3: Batch Endpoint Limits

### Rule

Batch/bulk endpoints MUST:
- Enforce maximum batch size (1000 traces max)
- Document batch limits in docstrings and OpenAPI
- Use transactional all-or-nothing semantics

### Rationale

- **Memory safety:** Prevents OOM from massive payloads
- **Performance:** Keeps request time < 30s
- **Correctness:** Transactions ensure data integrity

### Example: Compliant

```python
async def create_traces_batch_optimized(
    traces: list[ReasoningTrace],
    db: AsyncSession,
) -> TraceBatchCreateResponse:
    """Create traces in batch.
    
    Maximum batch size: 1000 traces per request.
    """
    max_batch_size = 1000
    if len(traces) > max_batch_size:
        raise ValueError(f"Batch size ({len(traces)}) exceeds maximum ({max_batch_size})")
    
    # All-or-nothing transaction
    db.add_all(db_traces)
    await db.commit()
    ...
```

---

## Guardrail 4: AsyncSession Concurrency (CRITICAL)

### Rule

**NEVER** use concurrent operations (asyncio.gather, concurrent.futures, etc.) on the same `AsyncSession` instance.

`AsyncSession` is mutable and stateful - concurrent access will cause data corruption and race conditions.

### Rationale

From [SQLAlchemy async documentation](https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html):

> "AsyncSession is mutable and not thread-safe or concurrency-safe in general."

### Example: Compliant (M10 Batch Optimization)

```python
async def create_traces_batch_optimized(traces, db):
    """Optimized batch with SEQUENTIAL refresh."""
    db.add_all(db_traces)
    await db.commit()
    
    # ✅ Single bulk SELECT (sequential, safe)
    result = await db.execute(select(Trace).where(Trace.id.in_(trace_ids)))
    refreshed_traces = result.scalars().all()
    ...
```

### Example: Non-Compliant ❌

```python
async def create_traces_batch_dangerous(traces, db):
    """DO NOT DO THIS - concurrent refresh on same session."""
    db.add_all(db_traces)
    await db.commit()
    
    # ❌ DANGER: Concurrent operations on same AsyncSession
    refresh_tasks = [db.refresh(trace) for trace in db_traces]
    await asyncio.gather(*refresh_tasks)  # Race conditions!
```

### Approved Patterns

If you need concurrency for database operations:
1. **Use separate sessions** (create new AsyncSession per task)
2. **Use bulk operations** (single SELECT with WHERE IN)
3. **Use connection pooling** (rely on SQLAlchemy's pool, not app-level concurrency)

---

## Guardrail 5: Service Layer Organization

### Rule

Business logic MUST be in the `services/` directory, not `helpers/`.

**Distinction:**
- **helpers/**: Stateless utilities (file I/O, stats, formatting)
- **services/**: Business logic and orchestration (validation + DB + transformation)

### Example Structure

```
tunix_rt_backend/
├── app.py                    # Thin controllers only
├── helpers/
│   ├── datasets.py           # ✅ load_manifest, save_manifest (file I/O)
│   └── traces.py             # ✅ get_trace_or_404 (validation helper)
├── services/
│   ├── traces_batch.py       # ✅ create_traces_batch (business logic)
│   └── datasets_export.py    # ✅ export_dataset_to_jsonl (formatting + DB)
```

---

## Guardrail 6: Timezone-Aware Datetimes

### Rule

Always use timezone-aware UTC datetimes. **Never use `datetime.utcnow()`** (deprecated in Python 3.13).

### Rationale

- Prevents timezone ambiguity
- Avoids Python 3.13+ deprecation warnings
- Explicit > implicit

### Example: Compliant

```python
from datetime import UTC, datetime

created_at: datetime = Field(
    default_factory=lambda: datetime.now(UTC),
    description="Creation time (UTC)"
)
```

### Example: Non-Compliant ❌

```python
from datetime import datetime

created_at: datetime = Field(
    default_factory=lambda: datetime.utcnow(),  # ❌ Deprecated
    description="Creation time"
)
```

---

## Guardrail 7: Export Format Determinism

### Rule

Dataset exports MUST:
- Maintain manifest order (deterministic)
- Skip deleted traces gracefully (no errors)
- Document format specifications

### Rationale

- **Reproducibility:** Same manifest → same output
- **Robustness:** Handle traces deleted between manifest creation and export
- **Debugging:** Predictable output order

### Example: Compliant

```python
async def export_dataset_to_jsonl(manifest, db, format):
    """Export dataset maintaining manifest order."""
    # Fetch all traces
    result = await db.execute(select(Trace).where(Trace.id.in_(manifest.trace_ids)))
    trace_map = {t.id: t for t in result.scalars().all()}
    
    # Build output in manifest order ✅
    lines = []
    for trace_id in manifest.trace_ids:
        trace = trace_map.get(trace_id)
        if not trace:
            continue  # ✅ Skip deleted traces gracefully
        ...
```

---

## Enforcement Strategy

### Code Review Checklist

Before merging PRs, verify:
- [ ] New endpoints delegate to services (Guardrail 1)
- [ ] Query params use Literal/Enum where applicable (Guardrail 2)
- [ ] Batch endpoints enforce size limits (Guardrail 3)
- [ ] No concurrent AsyncSession usage (Guardrail 4)
- [ ] Business logic in services/, not app.py (Guardrail 5)
- [ ] All datetimes are timezone-aware (Guardrail 6)
- [ ] Exports maintain determinism (Guardrail 7)

### Automated Enforcement

Future improvements (M11+):
- Linter rule: Detect `datetime.utcnow()` usage
- Linter rule: Detect asyncio.gather with db parameter
- Static analysis: Flag business logic in app.py

---

## Exceptions & Overrides

### When to Deviate

Guardrails can be broken in exceptional cases with:
1. **Explicit justification** in code comments
2. **Documentation** in PR description
3. **Approval** from maintainer

### Known Exceptions

None as of M10.

---

## References

- [SQLAlchemy Async I/O Documentation](https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html)
- [FastAPI Query Parameters](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/)
- [Python datetime.utcnow() Deprecation Discussion](https://discuss.python.org/t/why-is-datetime-utcnow-deprecated/86868)

---

**Last Updated:** December 21, 2025  
**Next Review:** M11 (when new patterns emerge)
