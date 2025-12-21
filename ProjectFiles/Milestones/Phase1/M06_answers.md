Got it — here are concrete preferences for **Q1–Q17** so Cursor can proceed without more back-and-forth. I’m optimizing for: **enterprise-grade cleanliness, coverage stability, no premature abstraction, and minimal CI risk.**

---

## Phase M6.1 — Validation Helper Extraction

### Q1: Where should the helper live?

**Put it in the application layer**:

✅ `backend/tunix_rt_backend/helpers/traces.py` (new folder `helpers/`)
Reason: it raises `HTTPException`, so it’s not purely “db layer.” Keep DB-layer pure-ish, app-layer can know FastAPI semantics.

**Structure:**

* `helpers/traces.py`: `get_trace_or_404`
* `helpers/scores.py`: later if needed
* `helpers/__init__.py`: exports

Avoid `db/helpers.py` unless it’s truly DB-only.

---

### Q2: Error message format

Use **consistent core format**, with **optional context prefix** only where it’s user-facing valuable.

Default:

* `"Trace {trace_id} not found"`

For compare specifically, it’s helpful to keep context:

* `"Base trace {trace_id} not found"`
* `"Other trace {trace_id} not found"`

But implement this **via an optional `label` parameter**:

`get_trace_or_404(db, trace_id, label: str | None = None)`

If label is provided, message becomes: `"{label} trace {id} not found"`.

---

### Q3: Naming

Use the straightforward, idiomatic name:

✅ `get_trace_or_404`

(Keep symmetry for future helpers: `get_score_or_404`, etc.)

---

### Q4: Existing coverage workarounds / flags

✅ **Remove the branch flags** as part of the refactor.

Do not preserve synthetic “flag toggles.” The whole point of M6 is to make coverage *natural*, not gamed.

If coverage drops unexpectedly, fix via:

* better structure (helpers)
* targeted tests
* small validation refactor

…but no fake branches.

---

## Phase M6.2 — Branch Coverage Normalization

### Q5: Branch coverage target

✅ **No regression from current** (your analysis says 79% branch).
So: **keep ≥ 79%**. If a tiny drop happens (≤2 points) but still above gate, document it; otherwise treat as regression bug.

Target:

* Branch: **≥ 79%**
* Line: no regression (keep current)

---

### Q6: What goes in `docs/M6_COVERAGE_DELTA.md`?

Include:

1. Before/after total metrics (line + branch)
2. “Top 5 files by branch delta” (not every file)
3. Short explanation of why any branches are structurally hard to hit (if true)
4. “Rules of thumb” for future endpoints (validation patterns)

Do **not** dump verbose per-line coverage output into docs.

---

### Q7: Tests for helpers

✅ Add tests for helpers **explicitly**, but keep them light.

* If helper is app-layer and raises `HTTPException`, you can unit test with a tiny in-memory db fixture or mock AsyncSession execute.
* Also ensure endpoint integration tests still pass (they already exist).

So: **both**, but minimal unit tests:

* 1 success case
* 1 not-found case (assert status/message)

---

## Phase M6.3 — E2E Selector Hardening

### Q8: data-testid convention

✅ Yes — use a prefix convention.

Proposed:

* `trace:*` for trace view/editor primitives
* `compare:*` for comparison UI
* `score:*` for evaluation outputs
* `sys:*` for health/status (like `sys:api-status`)

Example:

* `data-testid="compare:container"`
* `data-testid="compare:base-trace"`
* `data-testid="trace:json"`
* `data-testid="trace:step-list"`
* `data-testid="trace:step-item"`

This reads well and scales.

---

### Q9: ID selectors like `#trace-json`

✅ Convert to `data-testid` for consistency and future-proofing.

IDs are *usually* stable, but testids are explicitly a testing contract.

So:

* keep the `id` if it’s useful for accessibility/forms
* but tests should use `getByTestId`.

---

### Q10: Replace all text-based selectors?

✅ Replace **all** global text-based selectors in `smoke.spec.ts` now.

This is exactly the kind of “stabilization” that belongs in M6, and it prevents future churn as UI copy evolves.

Rule:

* no `page.locator("text=...")`
* no `hasText:` without a scoped container or role/testid

---

## Phase M6.4 — CI Guardrails & Regression Protection

### Q11: Coverage regression threshold (X) + hard fail/warn

Use:
✅ **X = 5%**
✅ **Hard fail** on PRs.

Why: this is a stabilization repo, and coverage regressions are one of your biggest CI destabilizers.

If that feels too tight later, we can relax it, but start strict while stabilizing.

---

### Q12: CI annotation detail level

✅ File-level summary + top contributors is the sweet spot.

Produce:

* overall line/branch deltas
* top 5 files by branch increase
* top 5 files by branch decrease

Avoid function-level and line-level in CI output (too noisy and flaky).

If you want deeper detail, put it into an uploaded artifact, not console spam.

---

### Q13: Guardrails doc contents

✅ Include:

* code examples ✅/❌
* PR checklist
* “selector rules”
* “validation rules”
* short decision flow (“when to use helper”)

Skip the flowchart unless Cursor can do it quickly; checklist + examples is enough.

---

## General

### Q14: UNGAR prep

✅ Keep it simple. No namespace reservation, no generic `get_entity_or_404`.

Do not abstract for UNGAR yet.
M6 is stabilization only; abstraction now risks over-engineering.

---

### Q15: Docs location

✅ Do **both** in the lightest way:

1. Keep the three docs in `docs/` as listed (primary, easy to find)
2. Add a **short milestone summary** in your project’s milestone folder (if that’s your established pattern)

But do not duplicate content — milestone summary should link to the docs.

---

### Q16: Helper tests philosophy

✅ Both, minimal:

* Unit tests for helper behavior (2 tests)
* Integration tests already cover “wiring,” so don’t rewrite them

---

### Q17: Execution order

✅ Sequential phases:
**M6.1 → M6.2 → M6.3 → M6.4**

Reason: refactor first, then see coverage impact, then stabilize E2E, then add CI guardrails once behavior is known-good.

---

## One extra guardrail I want added (important)

Add a tiny policy to the repo docs:

**“New endpoints must call helper fetchers; inline `None → 404` is disallowed.”**

Enforce via:

* a simple grep-style CI check, or
* a lint comment rule, or
* code review checklist (acceptable if you prefer not to automate)

Given your “enterprise-grade” goal, I’d prefer **a lightweight CI grep** that flags common anti-patterns.

---

If you want, I can now rewrite your M6 plan into a single **Cursor execution prompt** with:

* exact file paths
* exact functions to create
* exact tests to add
* exact Playwright selector rewrites
* exact CI snippet for coverage regression guard

Just say “yes, generate Cursor prompt” and I’ll output it in one shot.
