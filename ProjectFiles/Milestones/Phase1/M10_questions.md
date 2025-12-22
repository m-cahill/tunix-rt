# M10 Clarifying Questions

## Project Analysis Summary

I've analyzed the codebase and understand the M10 goals:
- **Current State:** M09 complete, 127 tests passing, 79% line coverage, app.py is ~864 lines
- **M10 Goal:** Refactor for maintainability, improve coverage organically, fix deprecations, small perf improvements

---

## Questions Before Implementation

### 1. **Implementation Scope & Phasing**
The M10 plan has 6 phases (Phase 0-5). Should I:
- **Option A:** Complete all phases sequentially, committing after each phase
- **Option B:** Complete Phase 0 (baseline), pause for your review, then proceed
- **Option C:** Complete Phase 0-2 (baseline + typed validation + refactor), pause, then continue

**My Recommendation:** Option A (complete all phases) since the plan is well-defined and low-risk.

---

### 2. **Service Layer Pattern**
The plan mentions creating `services/traces_batch.py` and `services/datasets_export.py`. However, the current codebase uses a `helpers/` pattern. Should I:
- **Option A:** Create new `backend/tunix_rt_backend/services/` directory (aligns with plan)
- **Option B:** Keep `helpers/` pattern and add `helpers/traces_batch.py` and `helpers/datasets_export.py`

**My Recommendation:** Option A (create `services/`) to establish clear distinction:
- **helpers/** = utilities (validation, dataset file I/O, etc.)
- **services/** = business logic (trace processing, export formatting)

---

### 3. **Batch Endpoint Optimization (Phase 3)**
The plan suggests removing N `refresh()` calls. Two approaches:
- **Option A (Simple):** Skip refresh entirely, construct response from in-memory data
- **Option B (Bulk Query):** Single bulk SELECT after commit

**Trade-offs:**
- Option A: Faster (~10x), but IDs must be generated client-side or we return without DB-generated created_at
- Option B: Still faster (~5x), keeps DB as source of truth

**My Recommendation:** Option B (bulk query) - maintains consistency while improving perf.

---

### 4. **Training Script Tests (Phase 5)**
The plan suggests adding `--dry-run` support and tests. However, this will require:
- Modifying existing training scripts (`training/train_sft_tunix.py`, etc.)
- Adding new test file (`backend/tests/test_training_scripts_smoke.py`)

Should I:
- **Option A:** Full implementation as described (adds ~2-3 hours)
- **Option B:** Defer to M11 (focus M10 on app.py refactor only)

**My Recommendation:** Option B (defer) - training scripts are already well-documented and low-risk. Focus M10 on app-layer improvements.

---

### 5. **Coverage Expectations**
During refactoring (especially Phase 2), coverage may temporarily drop before recovering. Is this acceptable if:
- ✅ Final coverage meets or exceeds baseline (79% line)
- ✅ All tests still pass
- ⚠️ Intermediate commits might show 75-77% coverage

**My Recommendation:** Accept temporary dips in intermediate commits, verify final state meets baseline.

---

### 6. **Git Strategy**
Should I:
- **Option A:** Work directly on `main` branch (appropriate for low-risk refactor)
- **Option B:** Create `m10-refactor` branch, then merge to main
- **Option C:** Create branch but auto-merge after completion

**My Recommendation:** Option B (branch) - allows you to review before merging, safer for refactoring.

---

### 7. **Commit Granularity**
The plan suggests 7 commits. Should I:
- **Option A:** Follow exact sequence (7 commits)
- **Option B:** Group related changes (3-4 commits: baseline, refactor, fixes, docs)

**My Recommendation:** Option A (7 commits) - better for git history and potential rollback.

---

### 8. **Phase 5 Scope Clarification**
The plan says "Training script dry-run smoke tests" but also says "no heavy runtime". Should these tests:
- **Option A:** Actually invoke scripts in subprocess with `--dry-run` flag
- **Option B:** Just validate config YAML parsing (no script execution)
- **Option C:** Defer entirely (see Question 4)

**My Recommendation:** Option C (defer to M11) as mentioned in Question 4.

---

### 9. **Documentation Updates**
Beyond `docs/M10_GUARDRAILS.md`, should I also create:
- `docs/M10_BASELINE.md` (like M09)?
- `docs/M10_SUMMARY.md` (like M09)?

**My Recommendation:** Yes - maintain consistency with milestone documentation pattern.

---

### 10. **AsyncSession Concurrency Note**
Phase 3 mentions "No concurrent use of AsyncSession" guardrail. The batch endpoint doesn't currently do concurrent operations. Should I:
- **Option A:** Just document the guardrail (no code changes needed)
- **Option B:** Add a code comment in batch endpoint for clarity

**My Recommendation:** Option B - add inline comment for future maintainers.

---

## Summary of Recommendations

| Question | Recommended Approach |
|----------|---------------------|
| 1. Scope | Complete all phases (Option A) |
| 2. Pattern | Create `services/` directory (Option A) |
| 3. Batch Opt | Bulk SELECT query (Option B) |
| 4. Training Tests | Defer to M11 (Option B) |
| 5. Coverage | Accept temporary dips (Yes) |
| 6. Git | Create branch `m10-refactor` (Option B) |
| 7. Commits | 7 commits as planned (Option A) |
| 8. Phase 5 | Defer script tests (Option C) |
| 9. Docs | Create baseline + summary (Yes) |
| 10. Async Note | Add inline comment (Option B) |

---

**Ready to proceed after your confirmation on these decisions.** 🎯

