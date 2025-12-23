# M6 Development Guardrails

**Milestone:** M6 - Validation Refactor & CI Stability Hardening  
**Purpose:** Prevent regression and maintain code quality standards  
**Audience:** All contributors to tunix-rt

---

## Overview

This document defines **mandatory** development patterns and **forbidden** anti-patterns for the tunix-rt codebase. These guardrails emerged from M6's structural quality reset and are designed to prevent the coverage instability and validation duplication that plagued M5.

**Key Principle:** *Good structure beats coverage hacks every time.*

---

## Validation Rules

### ✅ DO: Use Helper Functions

**Pattern:** Centralized `get_X_or_404` helpers for all database fetches that may return 404.

```python
# CORRECT: Use helper
from tunix_rt_backend.helpers.traces import get_trace_or_404

@app.get("/api/traces/{trace_id}")
async def get_trace(
    trace_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraceDetail:
    db_trace = await get_trace_or_404(db, trace_id)
    return TraceDetail(...)
```

**Why:**
- Consistent error messages across endpoints
- Single responsibility (DRY principle)
- Easy to test in isolation
- Natural branch coverage

### ❌ DON'T: Inline Validation Logic

```python
# WRONG: Inline fetch + None check
result = await db.execute(select(Trace).where(Trace.id == trace_id))
db_trace = result.scalar_one_or_none()

if db_trace is None:
    raise HTTPException(404, f"Trace with id {trace_id} not found")
```

**Problems:**
- Duplicates validation across endpoints
- Inconsistent error messages
- Harder to maintain
- Inflates endpoint complexity

### ✅ DO: Use Label Parameter for Context

```python
# CORRECT: Context-specific error messages
base_trace = await get_trace_or_404(db, base_id, label="Base")
other_trace = await get_trace_or_404(db, other_id, label="Other")

# Errors: "Base trace {id} not found" vs "Other trace {id} not found"
```

**Why:**
- Clear error messages for users
- Easier debugging
- No code duplication

### ❌ DON'T: Synthetic Branch Flags

```python
# WRONG: Coverage workarounds with flags
trace_found = False
if db_trace is None:
    raise HTTPException(404, ...)
else:
    trace_found = True  # Synthetic flag

assert trace_found  # Unnecessary assertion
```

**Problems:**
- Masks real coverage gaps
- Adds noise to codebase
- Makes refactoring harder
- False sense of coverage

---

## Selector Rules (E2E & Frontend Tests)

### ✅ DO: Use data-testid for UI Elements

**Pattern:** Prefix-based naming convention for test IDs.

```typescript
// CORRECT: data-testid with semantic prefixes
<div data-testid="trace:section">
  <textarea data-testid="trace:json" />
  <button data-testid="trace:upload">Upload</button>
</div>

<div data-testid="compare:section">
  <input data-testid="compare:base-id" />
  <button data-testid="compare:submit">Fetch & Compare</button>
</div>

<div data-testid="sys:api-card">
  <p data-testid="sys:api-status">API: healthy</p>
</div>
```

**Prefix Convention:**
- `sys:*` - System status (health, errors)
- `trace:*` - Trace view/editor primitives
- `compare:*` - Comparison UI elements
- `score:*` - Evaluation outputs

### ✅ DO: Use getByTestId in E2E Tests

```typescript
// CORRECT: Explicit data-testid selectors
const uploadBtn = page.getByTestId('trace:upload');
const compareResult = page.getByTestId('compare:result');
const apiStatus = page.getByTestId('sys:api-status');
```

**Why:**
- Resilient to copy changes
- Explicit testing contract
- Fast and reliable
- No selector collision

### ❌ DON'T: Global Text Selectors

```typescript
// WRONG: Unscoped text matching
const uploadBtn = page.locator('text=Upload');
const result = page.locator('.comparison-result').getByText('What is 2 + 2?');
```

**Problems:**
- Breaks when copy changes
- Ambiguous in multilingual apps
- May match multiple elements
- Flaky in complex UIs

### ✅ DO: Use Role-Based Selectors (When Appropriate)

```typescript
// CORRECT: Semantic HTML with role selectors
await expect(page.getByRole('heading', { name: 'Base Trace' })).toBeVisible();
const fetchBtn = page.getByRole('button', { name: 'Fetch', exact: true });
```

**When to use:**
- Semantic HTML elements (headings, buttons, links)
- Accessibility-critical elements
- Standard UI patterns

### ❌ DON'T: hasText Without Scoping

```typescript
// WRONG: hasText without container scope
const button = page.locator('button', { hasText: 'Upload' });
```

**Allowed:**
```typescript
// CORRECT: hasText with scoped container
const traceSection = page.getByTestId('trace:section');
const button = traceSection.locator('button', { hasText: 'Upload' });
```

---

## Coverage Rules

### ✅ DO: Write Natural Tests

**Pattern:** Test real behavior, not coverage metrics.

```python
# CORRECT: Tests meaningful scenarios
async def test_get_trace_or_404_success(test_db):
    """Test helper returns trace when it exists."""
    trace = Trace(trace_version="1.0", payload={...})
    test_db.add(trace)
    await test_db.commit()

    result = await get_trace_or_404(test_db, trace.id)

    assert result.id == trace.id
```

**Why:**
- Tests real functionality
- Coverage is natural byproduct
- Easy to understand

### ❌ DON'T: Add Tests Just for Coverage

```python
# WRONG: Coverage padding test
async def test_coverage_filler():
    """Test to hit uncovered branch."""
    # Contrived scenario that doesn't reflect real usage
    ...
```

**Problems:**
- Brittle tests
- Maintenance burden
- False sense of quality

### ✅ DO: Document Structural Coverage Gaps

**Pattern:** If a branch is structurally hard to hit, document it.

```python
# CORRECT: Documented in docs/M6_COVERAGE_DELTA.md

@field_validator("step_index")
def validate_step_index(cls, value):
    if value < 0:  # Hard to hit: Pydantic validates at API boundary
        raise ValueError("Step index must be non-negative")
    return value
```

**Why:**
- Transparency about coverage limits
- Prevents false alarms
- Guides future improvements

### ❌ DON'T: Lower Coverage Gates

```python
# WRONG: Lowering thresholds to pass
# pyproject.toml
[tool.coverage.report]
fail_under = 60  # Down from 70% because we couldn't hit it
```

**Problems:**
- Masks real quality decline
- Ratchet effect (harder to raise later)
- Defeats purpose of gates

---

## PR Checklist

Before submitting a PR, verify:

### Backend Changes

- [ ] **No inline `None → 404` validation** - Use `get_trace_or_404` helper
- [ ] **No synthetic branch flags** (`trace_found = False`, etc.)
- [ ] **Helper usage is consistent** - All endpoints use same pattern
- [ ] **Error messages are clear** - Use `label` parameter for context
- [ ] **Tests cover real scenarios** - Not just coverage padding
- [ ] **Coverage gates pass** - ≥ 90% line, ≥ 88% branch (or documented delta)

### Frontend Changes

- [ ] **data-testid on interactive elements** - Buttons, inputs, containers
- [ ] **Prefix convention followed** - `sys:*`, `trace:*`, `compare:*`, etc.
- [ ] **IDs kept for accessibility** - Forms still have `id` attributes
- [ ] **Tests use `getByTestId`** - No global text selectors
- [ ] **Role-based selectors used appropriately** - Semantic HTML only

### E2E Changes

- [ ] **All selectors use `getByTestId` or scoped `getByRole`**
- [ ] **No global `page.locator('text=...')`** - Forbidden
- [ ] **hasText only within scoped containers** - No global text matching
- [ ] **Tests pass locally** - Full E2E suite runs clean
- [ ] **Selector guardrail comment present** - Top of smoke.spec.ts

---

## Decision Flowcharts

### When to Create a Helper?

```
┌─────────────────────────────────────┐
│ Do you fetch an entity from DB and  │
│ need to return 404 if not found?    │
└────────────┬────────────────────────┘
             │
     ┌───────┴───────┐
     │ YES           │ NO
     ▼               ▼
┌─────────────┐   ┌──────────────────┐
│ Use helper  │   │ Inline logic OK  │
│ (e.g., get_ │   │ (e.g., filters,  │
│ trace_or_404)│   │ aggregations)    │
└─────────────┘   └──────────────────┘
```

### When to Use data-testid vs Role?

```
┌──────────────────────────────────────┐
│ Is the element semantic HTML?        │
│ (button, heading, link, input)       │
└────────────┬─────────────────────────┘
             │
     ┌───────┴───────┐
     │ YES           │ NO
     ▼               ▼
┌──────────────┐  ┌──────────────────┐
│ Prefer role  │  │ Use data-testid  │
│ (getByRole)  │  │ (getByTestId)    │
└──────────────┘  └──────────────────┘
     │                 │
     │                 │
     ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│ Is label     │  │ Follow prefix    │
│ stable?      │  │ convention       │
└──────┬───────┘  └──────────────────┘
       │
  ┌────┴────┐
  │ YES     │ NO
  ▼         ▼
┌─────┐  ┌────────────────┐
│ Use │  │ Use data-testid│
│ role│  │ instead        │
└─────┘  └────────────────┘
```

---

## Anti-Pattern Examples

### Backend: The Triple Fetch

```python
# WRONG: Fetching same entity multiple ways in one endpoint
result1 = await db.execute(select(Trace).where(Trace.id == id1))
trace1 = result1.scalar_one_or_none()

result2 = await db.execute(select(Trace).where(Trace.id == id2))
trace2 = result2.scalar_one_or_none()

result3 = await db.execute(select(Trace).where(Trace.id == id3))
trace3 = result3.scalar_one_or_none()

# CORRECT: Use helper for all fetches
trace1 = await get_trace_or_404(db, id1)
trace2 = await get_trace_or_404(db, id2)
trace3 = await get_trace_or_404(db, id3)
```

### Frontend: The Selector Soup

```typescript
// WRONG: Mix of selector strategies
const button1 = page.locator('text=Upload');
const button2 = page.locator('#upload-btn');
const button3 = page.getByTestId('upload');
const button4 = page.locator('button').nth(2);

// CORRECT: Consistent data-testid usage
const uploadBtn = page.getByTestId('trace:upload');
const compareBtn = page.getByTestId('compare:submit');
const loadExampleBtn = page.getByTestId('trace:load-example');
```

### E2E: The Text Hunt

```typescript
// WRONG: Searching for text across entire page
await expect(page.getByText('What is 2 + 2?')).toBeVisible();
await expect(page.getByText('Explain photosynthesis')).toBeVisible();

// CORRECT: Scoped to specific data-testid containers
const basePrompt = page.getByTestId('compare:base-prompt');
const otherPrompt = page.getByTestId('compare:other-prompt');

await expect(basePrompt).toContainText('What is 2 + 2?');
await expect(otherPrompt).toContainText('Explain photosynthesis');
```

---

## Enforcement

### Automated (CI)

1. **Coverage Regression Check** (M6.4.1)
   - Fails if branch coverage drops > 5% from main
   - Emits file-level delta summary

2. **Inline Validation Grep** (M6.4.4)
   - Flags `scalar_one_or_none()` + `if ... is None` patterns in app.py
   - Suggests helper usage

3. **Coverage Gates** (Existing)
   - Line coverage ≥ 90%
   - Branch coverage ≥ 88%

### Manual (Code Review)

- Reviewers check PR checklist compliance
- Flag anti-patterns in comments
- Require helper usage for new endpoints
- Verify selector patterns in E2E tests

---

## Exceptions & Overrides

### When Can You Skip a Guardrail?

**Only when:**
1. Performance-critical path (document in PR)
2. Third-party library constraint (document workaround)
3. Temporary tech debt (create issue to fix)

**Process:**
1. Add comment explaining exception
2. Link to issue/ADR documenting rationale
3. Get explicit approval from maintainer

**Example:**
```python
# EXCEPTION: Inline validation for performance-critical bulk operation
# See issue #123 for helper optimization plan
# TODO: Refactor to use helper once issue #123 is resolved
result = await db.execute(select(Trace).where(...))
traces = result.scalars().all()
if not traces:
    raise HTTPException(404, "No traces found")
```

---

## Related Documentation

- **M6 Coverage Delta:** [docs/M6_COVERAGE_DELTA.md](./M6_COVERAGE_DELTA.md)
- **M6 Validation Refactor:** [docs/M6_VALIDATION_REFACTOR.md](./M6_VALIDATION_REFACTOR.md)
- **Helper Implementation:** `backend/tunix_rt_backend/helpers/traces.py`
- **E2E Selector Policy:** `e2e/tests/smoke.spec.ts` (header comment)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-21 | Initial guardrails from M6 completion |

---

**Remember:** These guardrails exist to make your life *easier*, not harder. They encode lessons learned from M6's successful refactor. Follow them, and your code will be cleaner, tests more stable, and coverage more natural.

---

**Next Review:** After M7 (or when patterns evolve significantly)
