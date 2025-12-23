# M06 Clarifying Questions

## Project Analysis Summary

I've analyzed the tunix-rt project and understand:
- **Current State**: M5 complete with 89% line coverage, 79% branch coverage
- **Architecture**: FastAPI backend, React frontend, Playwright E2E tests
- **Database**: PostgreSQL with async SQLAlchemy + Alembic migrations
- **Current Validation Patterns**: Inline `if None: raise HTTPException(404)` in endpoints (app.py lines 262-266, 373-377, 191-205)
- **E2E Selectors**: Mix of `getByTestId`, `locator`, `getByRole`, and scoped text selectors
- **Coverage Gates**: 70% minimum (current: 89% line, 79% branch)

## Clarifying Questions

### Phase M6.1 — Validation Helper Extraction

**Q1: Scope of Refactoring**
- The plan mentions creating `backend/tunix_rt_backend/db/helpers.py`. Should this be:
  - `backend/tunix_rt_backend/db/helpers.py` (database layer), OR
  - `backend/tunix_rt_backend/helpers.py` (application layer)?
- **Rationale**: The helper deals with HTTPException (FastAPI), but also operates on DB models. Which layer is preferred?

**Q2: Error Message Format**
- Current error messages vary:
  - `get_trace`: "Trace with id {trace_id} not found"
  - `compare_traces`: "Base trace {base} not found" / "Other trace {other} not found"
  - `score_trace`: "Trace with id {trace_id} not found"
- Should the helper use a **consistent format** like: `"Trace {trace_id} not found"` (simpler), or keep context-specific messages?

**Q3: Helper Naming Convention**
- Proposed: `get_trace_or_404(db, trace_id) -> Trace`
- Alternative: `fetch_trace_or_404`, `require_trace`, `get_trace_checked`?
- **Preference for naming?**

**Q4: Current Coverage Workarounds**
- `app.py` has explicit branch flags (lines 188-208, 261-270) to satisfy branch coverage.
- After refactoring to helpers, these flags will move/disappear. Should I:
  - Keep similar flags in the helper, OR
  - Remove them and document the coverage delta?

### Phase M6.2 — Branch Coverage Normalization

**Q5: Branch Coverage Target**
- Plan says "≥ 70%" but current is 79%. Should I:
  - **Maintain current 79%** (no regression), OR
  - **Document any delta** if it drops to 70-79% range?

**Q6: Coverage Delta Documentation**
- The plan calls for `docs/M6_COVERAGE_DELTA.md`. Should this include:
  - Before/after metrics per file?
  - Explanation of why certain branches are unreachable?
  - Recommendations for future endpoints?

**Q7: Test Additions**
- Plan says "add targeted tests (only if required)". Given current 89% line coverage, should I:
  - Add tests for the **new helpers** specifically?
  - Add tests only if coverage **drops below gates**?

### Phase M6.3 — E2E Selector Hardening

**Q8: data-testid Naming Convention**
- Plan suggests:
  - `data-testid="trace-compare-base"`
  - `data-testid="trace-step-item"`
  - `data-testid="trace-json-view"`
- Should we use a **prefix convention** like:
  - `trace-*` for trace-related elements?
  - `comparison-*` for comparison UI?
  - `health-*` for health status?

**Q9: Current Selectors in E2E**
- Current E2E already uses some good patterns:
  - `getByTestId('api-status')` — ✅ Good
  - `locator('#trace-json')` — ⚠️ ID-based (acceptable?)
  - `locator('button', { hasText: 'Load Example' })` — ⚠️ Text-based
  - `getByRole('button', { name: 'Fetch', exact: true })` — ✅ Good
- Should ID-based selectors (`#trace-json`) be:
  - **Kept as-is** (IDs are stable), OR
  - **Converted to data-testid** for consistency?

**Q10: Selector Migration Strategy**
- Should I:
  - Replace **all text-based selectors** at once (smoke.spec.ts has ~15 instances), OR
  - Replace only **problematic ones** (e.g., lines 188-189 with text collision in comparison)?

### Phase M6.4 — CI Guardrails & Regression Protection

**Q11: Coverage Regression Threshold**
- Plan mentions "fail if branch coverage drops > X% from main". What should X be?
  - **5%** (strict), **10%** (moderate), **15%** (lenient)?
- Should this be a **hard fail** or **warning**?

**Q12: CI Annotation Detail Level**
- For coverage failures, how much detail should we output?
  - File-level summary (e.g., "app.py: +8 branches")?
  - Function-level detail (e.g., "compare_traces: +4 branches")?
  - Line-level diff (verbose, potentially noisy)?

**Q13: Guardrails Documentation**
- Should `docs/M6_GUARDRAILS.md` include:
  - Code examples (✅ do this, ❌ not this)?
  - Decision flowchart (when to use helper vs inline validation)?
  - Checklist for PR reviews?

### General Questions

**Q14: UNGAR Integration Preparation**
- The plan explicitly excludes UNGAR work from M06. However, should I:
  - **Reserve namespace** for future UNGAR helpers (e.g., separate helpers module)?
  - **Design helpers to be reusable** for UNGAR entities (generic `get_entity_or_404`)?
  - **Keep it simple** for traces/scores only (no premature abstraction)?

**Q15: Documentation Deliverables**
- Plan calls for 3 docs:
  - `docs/M6_VALIDATION_REFACTOR.md`
  - `docs/M6_COVERAGE_DELTA.md`
  - `docs/M6_GUARDRAILS.md`
- Should these be:
  - **Standalone docs** in `docs/` (as listed), OR
  - **Milestone summary** in `ProjectFiles/Milestones/Phase1/M06_summary.md` (following project pattern)?

**Q16: Testing Philosophy**
- Should the helper tests focus on:
  - **Unit tests** (test helper in isolation with mocked DB)?
  - **Integration tests** (test via existing endpoint tests)?
  - **Both** (unit + integration for full coverage)?

**Q17: Timeline & Phasing**
- Should I:
  - Execute phases **sequentially** (M6.1 → M6.2 → M6.3 → M6.4), OR
  - Interleave phases (e.g., M6.1 + M6.3 together since both touch app.py and frontend)?

---

## Next Steps

Once you answer these questions, I'll:
1. Create a comprehensive TODO list based on the M06 plan
2. Begin implementation in phases
3. Document progress and deltas as I go

Please respond with your preferences, and I'll proceed accordingly! 🚀
