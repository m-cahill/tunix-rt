# M05 CI Failure Analysis

**Date:** 2025-12-21  
**Commit:** `d8743ca` (M05 implementation)  
**Status:** 2 FAILURES, 6 PASSES

---

## CI Job Results Summary

| Job | Status | Issue |
|-----|--------|-------|
| changes | ✅ PASSED | Correctly detected backend, frontend, e2e changes |
| security-backend | ✅ PASSED | No vulnerabilities found |
| security-frontend | ✅ PASSED | 5 moderate (pre-existing, warn-only) |
| security-secrets | ✅ PASSED | No leaks detected |
| backend (3.11) | ❌ FAILED | **Branch coverage: 59.38% < 68% gate** |
| backend (3.12) | ⚠️ CANCELED | Tests passed (46/46, 83% coverage) but job canceled |
| frontend | ✅ PASSED | 11/11 tests, 92% coverage, build successful |
| e2e | ❌ FAILED | **2 test failures (4 passed)** |

---

## Critical Issue #1: E2E Test Failures

### Failure 1: Button Selector Ambiguity

**Error:**
```
Error: strict mode violation: locator('button').filter({ hasText: 'Fetch' }) 
resolved to 2 elements:
  1) <button>Fetch</button>
  2) <button disabled>Fetch & Compare</button>
```

**Location:** `e2e/tests/smoke.spec.ts:74` (trace upload/fetch test)

**Root Cause:**
- The new "Fetch & Compare" button text contains the word "Fetch"
- Playwright's `hasText: 'Fetch'` is matching BOTH buttons
- This is a **selector regression** introduced by M05's comparison UI

**Impact:**
- Existing E2E test for trace upload/fetch is now failing
- Test was passing in M04, broken by M05 UI additions

---

### Failure 2: Trace ID Uniqueness Issue

**Error:**
```
expect(secondTraceId).not.toBe(firstTraceId);
Expected: not "634540ec-4ba5-496f-837c-c36a442deb29"
```

**Location:** `e2e/tests/smoke.spec.ts:142` (comparison test)

**Root Cause:**
- The test uploads two different traces
- But extracts the SAME trace ID from both success messages
- The `.trace-success` element is being reused, showing only the MOST RECENT trace ID
- Regex match is finding the same ID twice

**Impact:**
- New M05 comparison E2E test failing
- Cannot verify comparison functionality end-to-end

---

## Critical Issue #2: Branch Coverage Drop

### Coverage Gate Failure

**Current:** 59.38% branch coverage  
**Required:** 68.0% minimum  
**Gap:** -8.62 percentage points

**Evidence:**
```
Line Coverage:   83.28% (gate: >= 80.0%) ✅
Branch Coverage: 59.38% (gate: >= 68.0%) ❌
```

**Root Cause:**
The new M05 code added branches that are not fully covered by tests:
- `app.py`: 95 statements, 22 branches, 0 branch parts covered
- Likely untested paths in score_endpoint and compare_endpoint

**Existing Coverage (M4):**
- Was ~79% branch coverage
- M05 added new conditional logic without full branch coverage

**Impact:**
- CI job `backend (3.11)` failing
- Blocks merge/deployment
- M05 cannot be considered complete

---

## Detailed Analysis

### E2E Failure #1: Button Selector Fix

**Current Selector (broken):**
```typescript
const fetchBtn = page.locator('button', { hasText: 'Fetch' });
```

**Problem:** Matches both "Fetch" and "Fetch & Compare"

**Recommended Fix Options:**

**Option A: Use exact text match**
```typescript
const fetchBtn = page.getByRole('button', { name: 'Fetch', exact: true });
```

**Option B: Use more specific locator**
```typescript
const fetchBtn = page.locator('.trace-actions button', { hasText: 'Fetch' }).first();
```

**Option C: Change button text**
```typescript
// Change "Fetch & Compare" to "Compare Traces" or "Compare"
```

**Recommendation:** Option A (exact match) - most precise and idiomatic for Playwright

---

### E2E Failure #2: Trace ID Extraction Fix

**Current Issue:**
The test extracts trace IDs from the `.trace-success` div, but this element is being reused for both uploads. The regex is finding the same (most recent) ID both times.

**Recommended Fix:**
Store the first trace ID BEFORE uploading the second trace:

```typescript
// After first upload
await expect(successMessage).toBeVisible({ timeout: 5000 });
const firstSuccessText = await successMessage.textContent();
const firstTraceId = firstSuccessText?.match(/[0-9a-f]{8}-...)/)?.[0];
expect(firstTraceId).toBeTruthy();

// Clear the success message or wait for it to update
await traceTextarea.fill(complexTrace);
await uploadBtn.click();

// After second upload, wait for the element to update
await expect(successMessage).not.toContainText(firstTraceId!);  // Wait for change
const secondSuccessText = await successMessage.textContent();
const secondTraceId = secondSuccessText?.match(/[0-9a-f]{8}-...)/)?.[0];
```

**Alternative:** Add unique test IDs to trace success messages to avoid ambiguity

---

### Branch Coverage Failure: Missing Test Cases

**Analysis:**
The new M05 endpoints have untested branch paths. Let me identify specific gaps:

**In `score_trace` endpoint:**
```python
if score_request.criteria == "baseline":
    score_value, details = baseline_score(trace)
else:  # ← This branch is never tested
    raise HTTPException(...)
```

**In `compare_traces` endpoint:**
```python
if base not in db_traces:  # ← Tested
    raise HTTPException(...)
if other not in db_traces:  # ← Tested
    raise HTTPException(...)
# But: what if BOTH are missing? (two different code paths)
```

**Required New Tests:**
1. Test the `else` branch in score_trace (invalid criteria) - but wait, this can't happen due to Literal type!
2. Add more branch coverage for error handling paths in app.py

**The Real Issue:**
Looking at the coverage report, `app.py` shows:
- 95 statements
- 31-33 miss (depending on Python version)
- 22 branches, 0 branch parts

This suggests the **existing** app.py code (health endpoints, trace CRUD) has untested branches, not just the new M05 code.

**Hypothesis:** Adding new code increased the total branch count WITHOUT proportionally increasing covered branches, causing the percentage to drop.

---

## Recommendations

### Immediate Fixes (Required for CI Green)

1. **Fix E2E button selector** (5 min)
   - Change to exact match: `getByRole('button', { name: 'Fetch', exact: true })`
   - Update smoke.spec.ts line 73

2. **Fix E2E trace ID extraction** (10 min)
   - Store first ID before second upload
   - Wait for success message to update between uploads
   - Add assertion that IDs are different

3. **Add branch coverage tests** (30-45 min)
   - Identify specific untested branches in app.py
   - Add tests for error paths in health endpoints
   - Add tests for edge cases in trace CRUD
   - Target: bring branch coverage from 59% → 68%+

### Root Cause: M05 Added Branches Without Tests

**What happened:**
- M05 added new endpoints with error handling paths
- Unit tests cover happy paths well
- But branches in error handling (404s, validation) may not all be tested
- This diluted the overall branch coverage percentage

**What to do:**
Add missing branch coverage tests to get back above 68% threshold

---

## Proposed Action Plan

### Phase 1: E2E Fixes (Quick Wins - 15 min)
1. Update button selector in smoke.spec.ts (line 73)
2. Fix trace ID extraction logic in comparison test (lines 130-142)
3. Run E2E locally to verify

### Phase 2: Branch Coverage Investigation (30 min)
1. Run coverage with `--cov-report=html` to visualize gaps
2. Identify specific untested branches
3. Add targeted tests for missing branches
4. Verify branch coverage reaches 68%+

### Phase 3: Commit & Push (5 min)
1. Single commit: `fix(tests): resolve E2E selector conflicts and improve branch coverage`
2. Push to GitHub
3. Verify CI passes

---

## Expected Outcomes

After fixes:
- ✅ E2E: 6/6 tests passing
- ✅ Backend (3.11): Branch coverage ≥ 68%
- ✅ Backend (3.12): Tests passing (currently canceled, should pass)
- ✅ All other jobs: Continue passing

---

## Lessons Learned

1. **UI changes affect existing E2E tests** - need to audit test selectors when adding new buttons
2. **Branch coverage is fragile** - adding new code can dilute percentage even if code is well-tested
3. **Test isolation** - E2E tests should use more specific selectors (exact match, test IDs)
4. **Coverage monitoring** - should have run coverage check locally before pushing

---

## Next Steps

**Question for User:**
Should I proceed with the fixes immediately, or would you like to review this analysis first?

**Estimated Time to Green CI:** ~45-60 minutes total

---

**Prepared by:** AI Assistant (Claude Sonnet 4.5)  
**Analysis Date:** 2025-12-21  
**Status:** AWAITING USER DECISION
