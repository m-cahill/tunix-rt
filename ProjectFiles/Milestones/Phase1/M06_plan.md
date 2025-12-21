Excellent choice. Option 2 is the **right sequencing move**: you’re buying down structural risk *before* introducing a second domain (UNGAR), and you’re doing it in a way that reinforces your **enterprise-grade, audit-first narrative**.

Below is a **Cursor-ready M6 plan**: tightly scoped, phased, end-to-end verifiable, and explicitly designed to prevent a repeat of the M05 CI/coverage pain.

---

# M6 Plan — Validation Refactor & CI Stability Hardening

**Milestone:** M6
**Theme:** Structural Quality Reset (Before UNGAR Integration)
**Goal:** Make CI, coverage, and validation *boringly predictable* so future milestones scale safely
**Status Target:** 🟢 CI Green, Coverage Stable, Guardrails Enforced

---

## 🎯 M6 North Star

> **No new features. No new domains.**
> M6 exists to **stabilize the platform surface** so future complexity (UNGAR, LLM judges, multi-scorers) does not destabilize CI, coverage, or E2E reliability.

This is a **defensive engineering milestone**—and that’s a strength, not a slowdown.

---

## 📦 M6 Scope (What Is In / Out)

### ✅ IN

* Validation refactoring
* Branch coverage normalization
* E2E selector hardening
* CI guardrails to prevent regression
* Minor DX improvements (test helpers, shared utilities)

### ❌ OUT

* No UNGAR integration
* No new scoring features
* No LLM judge
* No new endpoints beyond refactors
* No schema changes (unless required for refactor)

---

## 🧱 Phase Breakdown (Cursor-Executable)

---

## **Phase M6.1 — Validation Helper Extraction (Core Structural Fix)**

### Problem Being Solved

Early-return validation logic duplicated across endpoints:

* inflates branch counts
* makes coverage unpredictable
* spreads correctness logic across files

### Deliverables

#### 1. New shared validation helpers

Create a new module:

```
backend/tunix_rt_backend/db/helpers.py
```

Implement (minimum):

```python
async def get_trace_or_404(
    db: AsyncSession,
    trace_id: UUID,
) -> Trace
```

Optionally (if useful now):

```python
async def get_score_or_404(...)
```

#### 2. Replace inline validation logic

Refactor **all** endpoints that:

* fetch traces
* check for `None`
* raise `HTTPException(404)`

Replace with helper usage.

#### 3. Centralize error semantics

* All “not found” errors must:

  * use identical message format
  * raise from helper
* No endpoint should implement its own fetch+404 logic

### Acceptance Criteria

* No duplicate “fetch + None check” patterns remain
* All affected endpoints still return identical status codes
* Existing tests pass without modification
* Net reduction in branch count in `app.py`

---

## **Phase M6.2 — Branch Coverage Normalization**

### Problem Being Solved

Branch coverage penalizes:

* early returns
* guard-style validation
* structurally correct but syntactically sparse logic

### Deliverables

#### 1. Re-run coverage analysis

* Capture **pre-refactor** branch/line metrics
* Capture **post-refactor** metrics
* Store results in:

  ```
  docs/M6_COVERAGE_DELTA.md
  ```

#### 2. Normalize validation structure

Where needed:

* replace `if x is None: raise` with:

  * helper call
  * or explicit conditional blocks that are *semantically necessary*

⚠️ Do **not** add fake branches or no-op `else:` blocks.

#### 3. Add targeted tests (only if required)

If refactor removes coverage but introduces new paths:

* add **surgical tests**
* no broad “coverage padding” tests

### Acceptance Criteria

* Branch coverage ≥ **70%** (or documented, justified delta)
* Line coverage remains ≥ gate
* Coverage behavior is explainable and stable across Python 3.11 / 3.12
* No coverage-gate workarounds or gate lowering

---

## **Phase M6.3 — E2E Selector Hardening (Prevent M05 Repeat)**

### Problem Being Solved

Playwright strict-mode failures due to:

* overlapping visible text
* DOM reuse (textarea vs rendered content)
* unscoped selectors

### Deliverables

#### 1. Introduce `data-testid` strategy

Add `data-testid` attributes for:

* comparison container
* trace step list
* trace JSON textarea
* score display elements

Follow naming convention:

```
data-testid="trace-compare-base"
data-testid="trace-step-item"
data-testid="trace-json-view"
```

#### 2. Update all E2E selectors

* Replace `text=` and ambiguous selectors
* Prefer:

  * `getByTestId`
  * `getByRole` **with scoped container**
* No global `page.locator('text=...')`

#### 3. Add a selector guardrail comment

At top of `smoke.spec.ts`:

> “All selectors must be either role-based or data-testid scoped.
> Global text selectors are forbidden.”

### Acceptance Criteria

* All E2E tests pass
* No selector relies on unscoped text matching
* Comparison UI is resilient to copy changes

---

## **Phase M6.4 — CI Guardrails & Regression Protection**

### Problem Being Solved

CI failures should be:

* explainable
* localized
* hard to accidentally introduce

### Deliverables

#### 1. Coverage regression guard

Add CI logic to:

* fail if **branch coverage drops > X% from main**
* emit diff summary in job output

(Does *not* replace absolute gate—this is additive.)

#### 2. CI annotation improvements

* On coverage failure:

  * print branch count delta
  * list files contributing most new branches

#### 3. Document new rules

Add:

```
docs/M6_GUARDRAILS.md
```

Include:

* validation patterns to follow
* selector rules
* coverage expectations for new endpoints

### Acceptance Criteria

* CI output is more informative than M05-era failures
* Future contributors know *why* something failed, not just that it did

---

## 🧪 End-to-End Verification Checklist (Required)

Before M6 can be closed:

* [ ] All backend tests pass
* [ ] All frontend tests pass
* [ ] E2E passes **twice consecutively**
* [ ] Branch coverage ≥ 70% (or justified & documented)
* [ ] No duplicated validation logic remains
* [ ] `data-testid` present for all complex UI elements
* [ ] CI annotations clearly explain failures

---

## 📚 Documentation Updates (Lightweight)

Add / update:

* `docs/M6_VALIDATION_REFACTOR.md`
* `docs/M6_COVERAGE_DELTA.md`
* `docs/M6_GUARDRAILS.md`

No README changes required.

---

## ⛔ Explicit Non-Goals (Write These in the PR)

> * No new features
> * No UNGAR work
> * No LLM judge
> * No schema changes
> * No performance optimization

This keeps reviewers (human or AI) aligned.

---

## 🔜 What This Enables (Why M6 Matters)

After M6:

* UNGAR integration becomes **mechanical**, not risky
* LLM judge work won’t destabilize CI
* Coverage gates stop being a surprise tax
* E2E tests stop breaking on copy changes

In other words: **M6 is the runway.**

---

If you want, next I can:

* convert this into a **single Cursor prompt** (“apply M6 in phases…”), or
* pre-draft the **validation helper code** and test deltas Cursor should expect.

Just tell me how hands-off you want the handoff to be.
