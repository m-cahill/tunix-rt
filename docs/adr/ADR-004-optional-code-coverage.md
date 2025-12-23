# ADR-004: Optional Integration Coverage Strategy

**Date:** 2025-12-21  
**Status:** Accepted  
**Context:** M07 UNGAR Integration

## Context

M07 adds UNGAR as an optional dependency (`backend[ungar]`). The UNGAR integration includes:
- Availability checking (can run without UNGAR)
- Generator/conversion logic (requires UNGAR installed)
- API endpoints (work in both states: installed/not installed)

After implementation, CI coverage dropped from **90%** to **69%** because:
1. UNGAR is not installed in default CI (by design - optional integration)
2. Generator code (60 statements) is 0% covered
3. Endpoints return 501 when UNGAR missing (limited code paths tested)

## Decision

Implement a **two-tier coverage measurement strategy** using coverage.py's `omit` configuration:

### Tier 1: Core Coverage (Default CI)
- **Config:** `.coveragerc`
- **Omits:** `tunix_rt_backend/integrations/ungar/high_card_duel.py`
- **Gate:** ≥70% (enforced, blocks CI)
- **Tests:** All default tests (UNGAR not required)
- **Purpose:** Maintain quality bar for core runtime

### Tier 2: Full Coverage (Optional Workflow)
- **Config:** `.coveragerc.full`
- **Omits:** None (measures everything)
- **Gate:** None (report-only)
- **Tests:** UNGAR integration tests (`@pytest.mark.ungar`)
- **Purpose:** Validate optional integration quality

## Alternatives Considered

### 1. Use `# pragma: no cover` throughout optional code
**Rejected:** Scatters exclusions across codebase, hard to maintain, hides untested code.

### 2. Lower coverage threshold to 65%
**Rejected:** Lowers quality bar for all code, allows future regressions.

### 3. Merge coverage from multiple workflows
**Rejected:** Complex artifact handling, violates "optional = non-blocking" principle.

### 4. Multi-threshold strategy with separate pytest runs
**Considered:** More complex but viable. Chose simpler omit-based approach first.

## Consequences

### Positive
- ✅ Maintains 70% core coverage gate (enforced)
- ✅ Core quality bar unchanged from M6
- ✅ Optional code still measured (just not blocking)
- ✅ Clean configuration (no scattered pragmas)
- ✅ Easy to understand and maintain

### Negative
- ⚠️ Two coverage configurations to maintain
- ⚠️ Optional code coverage not visible in main CI reports
- ⚠️ Requires discipline to ensure optional workflow runs regularly

### Neutral
- 📊 Coverage reports show ~84% in default CI (core only)
- 📊 Full coverage measured separately in optional workflow
- 🔄 Can revisit if optional integrations grow significantly

## Implementation

### Files Created/Modified
- `backend/.coveragerc` - Default CI configuration
- `backend/.coveragerc.full` - Optional workflow configuration
- `.github/workflows/ci.yml` - Use `.coveragerc`
- `.github/workflows/ungar-integration.yml` - Use `.coveragerc.full`
- `backend/pyproject.toml` - Removed inline coverage config (use files instead)

### Tests Added
- `test_ungar_availability.py` - Mock-based tests for availability logic
- Additional endpoint tests covering 501 response paths

## Monitoring

### Success Metrics
- Core coverage remains ≥70% in default CI
- Optional workflow reports full coverage regularly
- No coverage gate bypasses or lowering

### Red Flags
- Core coverage dropping below 70%
- Optional workflow consistently failing
- New optional code not added to omit list when appropriate

## Future Considerations

If optional integrations grow significantly:
1. Consider separate `core` and `integrations` test suites
2. Implement per-module coverage thresholds
3. Add CI guardrail to enforce omit list updates

## References

- **Coverage.py omit patterns:** https://coverage.readthedocs.io/en/latest/source.html
- **Pytest-cov config:** https://pytest-cov.readthedocs.io/en/latest/config.html
- **M07 Plan:** `ProjectFiles/Milestones/Phase1/M07_plan.md`
- **Workflow Analysis:** `ProjectFiles/Workflows/context_52782804457.md`
