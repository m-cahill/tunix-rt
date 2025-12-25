# M22 Milestone Audit: Evaluation Lock + Coverage Recovery + Training Readiness

**Auditor:** CodeAuditorGPT  
**Date:** 2025-12-24  
**Commit Range:** `841fa73..cc4fdb3`  
**Current SHA:** `cc4fdb3d2ec63e1833d908d740dbb25c21ce6c20`

---

## 1. Delta Executive Summary

### Strengths
- **Evaluation Semantics Locked**: `answer_correctness@v1` metric fully specified with deterministic implementation and versioning
- **Test Coverage Expansion**: Added 194 lines of test code across backend services (TuningService, ModelRegistryService)
- **Golden Dataset Foundation**: `golden-v1` dataset established with 5 curated test cases and seed script for reproducibility
- **UI Visibility**: Evaluation metrics now displayed across Run History, Tuning Trials, and Model Registry
- **Documentation Quality**: Comprehensive `evaluation.md` and `training_readiness.md` establish clear contracts

### Risks & Opportunities
- **Coverage Below Target**: Backend coverage at 67.11% (target: ≥75%) - primarily due to low coverage in `judges.py` (48%), `tuning_service.py` (49%), and `tunix_execution.py` (43%)
- **Incomplete Judge Implementation**: `AnswerCorrectnessJudge.evaluate()` has placeholder logic that doesn't actually load datasets and compare predictions (lines 122-169 in judges.py)
- **Frontend Test Warnings**: React `act()` warnings in test output indicate improper async state handling in tests

### Quality Gates

| Gate | Status | Evidence | Fix |
|------|--------|----------|-----|
| Lint/Type Clean | ✅ PASS | All pre-commit hooks passing | N/A |
| Tests | ✅ PASS | 214 backend, 45 frontend passing | N/A |
| Coverage Non-Decreasing | ⚠️ PARTIAL | Backend 67% (was ~80% in M21) | Add tests for judges.py, tuning_service.py |
| Secrets Scan | ✅ PASS | No new secrets detected | N/A |
| Deps CVE | ✅ PASS | No new CVEs introduced | N/A |
| Schema/Infra Migration | N/A | No schema changes in M22 | N/A |
| Docs/DX Updated | ✅ PASS | evaluation.md, training_readiness.md added | N/A |

---

## 2. Change Map & Impact

```mermaid
graph TD
    A[M22: Evaluation Lock] --> B[judges.py +161 lines]
    A --> C[tests +194 lines]
    A --> D[docs +70 lines]
    A --> E[UI +93 lines]
    
    B --> F[AnswerCorrectnessJudge]
    B --> G[JudgeFactory update]
    
    C --> H[test_tuning_service +74]
    C --> I[test_model_registry +120]
    C --> J[Tuning.test.tsx +83]
    
    D --> K[evaluation.md]
    D --> L[training_readiness.md]
    
    E --> M[ModelRegistry.tsx]
    E --> N[Tuning.test.tsx]
    
    style F fill:#90EE90
    style H fill:#87CEEB
    style I fill:#87CEEB
    style K fill:#FFD700
```

**Dependency Analysis:**
- ✅ Clean layering: `judges.py` → `helpers/datasets.py` (proper helper usage)
- ✅ Service isolation: Tests use mocking appropriately for Ray Tune dependencies
- ⚠️ Missing import cleanup: `settings` imported but unused in `judges.py` (line 17, removed in user's change)

---

## 3. Code Quality Focus (Changed Files Only)

### Issue Q-001: Incomplete Judge Implementation
**File:** `backend/tunix_rt_backend/services/judges.py:122-169`

**Observation:**
```python
async def evaluate(self, run: TunixRun) -> JudgeResult:
    # ... validation ...
    
    # Placeholder: Assume a perfect score for completed runs for now,
    # until dataset integration is fully in place.
    score = 100.0
    verdict: Literal["pass", "fail", "uncertain"] = "pass"
    metrics = {"answer_correctness": 1.0}
```

**Interpretation:** The core evaluation logic is a stub. While the infrastructure is in place, the judge doesn't actually load datasets, match predictions to ground truths, or compute real correctness scores.

**Recommendation:** Implement the full dataset loading and comparison logic:
1. Load manifest from `get_datasets_dir() / run.dataset_key / "manifest.json"`
2. For each trace_id in manifest, fetch the trace and extract `final_answer`
3. Compare normalized prediction vs ground_truth
4. Compute mean correctness score

**Priority:** HIGH - This is the core M22 deliverable.

---

### Issue Q-002: Test Coverage Regression
**File:** Multiple service files

**Observation:**
- `judges.py`: 48% coverage (81 of 175 lines uncovered)
- `tuning_service.py`: 49% coverage (69 of 139 lines uncovered)
- `tunix_execution.py`: 43% coverage (140 of 265 lines uncovered)
- Overall backend: 67.11% (down from ~82% in M21)

**Interpretation:** New code in `judges.py` lacks corresponding tests. The M22 goal was "≥75% backend coverage," but current state is 67%.

**Recommendation:**
1. Add unit tests for `AnswerCorrectnessJudge._normalize_text()`
2. Add integration test for `AnswerCorrectnessJudge.evaluate()` with real dataset
3. Add tests for `JudgeFactory.get_judge()` with "answer_correctness" override
4. Add error path tests for dataset loading failures

**Priority:** HIGH - Required to meet M22 acceptance criteria.

---

### Issue Q-003: Empty Dataset Guardrail Implementation
**File:** `backend/tunix_rt_backend/services/datasets_builder.py:89-91`

**Observation:**
```python
if not trace_ids:
    raise ValueError("Dataset is empty. No traces matched the filters.")
```

**Interpretation:** Guardrail added but test was modified to remove the exception assertion:
```python
# test_datasets.py:152 (modified)
assert manifest.trace_count == 0
assert manifest_path.exists()  # Changed from expecting ValueError
```

**Recommendation:** The test modification contradicts the guardrail's purpose. Either:
1. Keep the guardrail and fix the test to assert `pytest.raises(ValueError)`, OR
2. Remove the guardrail if empty datasets are valid

**Priority:** MEDIUM - Clarify intended behavior.

---

### Issue Q-004: Model Registry Idempotency Tests
**File:** `backend/tests/test_model_registry.py:186-233`

**Observation:** New test `test_idempotency_same_sha` verifies that promoting identical content (same SHA256) returns the existing `ModelVersion` rather than creating a duplicate.

**Interpretation:** ✅ Excellent hardening - prevents duplicate storage and ensures content-addressable semantics.

**Recommendation:** None - this is a best practice.

---

### Issue Q-005: Tuning Service Guardrail Weak
**File:** `backend/tunix_rt_backend/services/tuning_service.py:128-132`

**Observation:**
```python
# Guardrail: Warn if metric is not a recognized/locked metric
if job.metric_name != "answer_correctness":
    logger.warning(
        f"TuningJob {job.id} uses metric '{job.metric_name}' which is not 'answer_correctness'. "
        "Ensure this metric is properly defined and stable."
    )
```

**Interpretation:** This is only a warning, not a blocking check. Per `training_readiness.md`, the guardrail should "prevent starting a tuning job if the evaluation metric specification is not explicitly set or recognized."

**Recommendation:** Change to:
```python
LOCKED_METRICS = {"answer_correctness"}
if job.metric_name not in LOCKED_METRICS:
    raise ValueError(
        f"Metric '{job.metric_name}' is not a locked metric. "
        f"Allowed: {LOCKED_METRICS}"
    )
```

**Priority:** MEDIUM - Aligns with M22 "Training Readiness" goal.

---

### Issue Q-006: Frontend Test Act Warnings
**File:** Frontend test output

**Observation:**
```
Warning: An update to App inside a test was not wrapped in act(...).
```

**Interpretation:** Tests are triggering React state updates without proper `act()` wrapping, indicating potential race conditions in test assertions.

**Recommendation:** Wrap async operations in `waitFor()` or `act()`:
```tsx
await waitFor(() => {
  expect(screen.getByText(/Run #1/)).toBeInTheDocument();
});
```

**Priority:** LOW - Tests pass but warnings indicate fragility.

---

## 4. Tests & CI (Delta)

### Coverage Delta
- **Backend:** 67.11% (down from ~82% in M21) - **REGRESSION**
- **Frontend:** 45 tests passing (up from 31 in M21) - **IMPROVEMENT**
- **New Tests:** +194 lines across 3 files
  - `test_tuning_service.py`: +74 lines (error paths, Ray Tune mocking)
  - `test_model_registry.py`: +120 lines (promotion, idempotency, validation)
  - `Tuning.test.tsx`: +83 lines (Promote Best workflow)

### Test Adequacy
✅ **Strong:** Model Registry tests cover edge cases (failed runs, missing artifacts, SHA collisions)  
✅ **Strong:** Tuning Service tests mock Ray Tune dependencies properly  
⚠️ **Weak:** No tests for `AnswerCorrectnessJudge` actual evaluation logic  
⚠️ **Weak:** `seed_golden_dataset.py` script has no automated tests

### CI Impact
- ✅ Pre-commit hooks pass (ruff, mypy, trailing-whitespace, json/yaml validation)
- ✅ All 214 backend tests pass in 32.76s
- ⚠️ Coverage gate at 70% would FAIL (current: 67.11%)

---

## 5. Security & Supply Chain (Delta)

### Secrets Check
✅ **PASS** - No new secrets or credentials introduced

### Dangerous Patterns
✅ **PASS** - No SQL injection, command injection, or unsafe deserialization patterns
✅ **PASS** - Path traversal protection via `get_datasets_dir()` helper

### Dependency Changes
✅ **PASS** - No new dependencies added in this milestone
✅ **PASS** - No CVE changes from M21 baseline

### File Access Patterns
⚠️ **OBSERVE:** `seed_golden_dataset.py` writes to `backend/datasets/golden-v1/` without explicit permission checks. This is acceptable for a dev tool but should not be exposed as an API endpoint.

---

## 6. Performance & Hot Paths (If Touched)

**No hot request paths modified in M22.** Changes are primarily in:
- Test code (no runtime impact)
- Evaluation service (triggered async, not on request path)
- UI components (minimal frontend bundle impact: +10 lines in ModelRegistry.tsx)

**Recommendation:** Run baseline performance tests before M23 to establish evaluation latency budgets.

---

## 7. Docs & DX (Changed Surface)

### New Documentation
✅ **Excellent:** `docs/evaluation.md` - Clear metric specification with normalization rules  
✅ **Excellent:** `docs/training_readiness.md` - Comprehensive checklist with acceptance criteria  
✅ **Good:** `backend/tools/seed_golden_dataset.py` - Script has inline comments explaining golden dataset structure

### Missing Documentation
⚠️ **Gap:** How to run the seed script is not documented in `README.md` or `tunix-rt.md`  
⚠️ **Gap:** `AnswerCorrectnessJudge` implementation status (placeholder vs complete) not mentioned in docs

**Recommendation:** Add to `README.md`:
```markdown
## Seeding the Golden Dataset

docker compose up -d postgres
cd backend
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5433/postgres"
python -m tools.seed_golden_dataset
```

---

## 8. Ready-to-Apply Patches (≤5)

### Patch 1: Complete AnswerCorrectnessJudge Implementation
**Title:** `feat(judges): Implement actual dataset comparison in AnswerCorrectnessJudge`

**Why:** Core M22 deliverable is incomplete - judge returns hardcoded scores instead of evaluating against ground truth.

**Patch Hint:**
```python
# In AnswerCorrectnessJudge.evaluate(), replace placeholder (lines 122-169)
async def evaluate(self, run: TunixRun) -> JudgeResult:
    if run.status != "completed":
        return self._fail(f"Run status: {run.status}")
    
    # 1. Load manifest
    manifest_path = get_datasets_dir() / run.dataset_key / "manifest.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text())
    
    # 2. Fetch traces and ground truths
    stmt = select(Trace).where(Trace.trace_id.in_(manifest.trace_ids))
    result = await self.db.execute(stmt)
    traces = result.scalars().all()
    
    # 3. Compare predictions vs ground truths
    correct_count = 0
    for trace in traces:
        ground_truth = trace.payload.get("final_answer", "")
        prediction = run.stdout or ""  # Or extract from run artifacts
        if self._normalize_text(prediction) == self._normalize_text(ground_truth):
            correct_count += 1
    
    score = (correct_count / len(traces)) * 100 if traces else 0
    return JudgeResult(score=score, verdict="pass" if score > 0 else "fail", ...)
```

**Risk:** MEDIUM - Changes evaluation logic, requires new tests  
**Rollback:** Revert commit; existing placeholder returns optimistic scores

---

### Patch 2: Strengthen Tuning Metric Guardrail
**Title:** `fix(tuning): Block tuning jobs with unrecognized metrics`

**Why:** Current warning-only approach doesn't prevent premature tuning on unstable metrics.

**Patch Hint:**
```python
# backend/tunix_rt_backend/services/tuning_service.py:128
LOCKED_METRICS = {"answer_correctness"}

if job.metric_name not in LOCKED_METRICS:
    raise ValueError(
        f"Cannot start tuning job: metric '{job.metric_name}' is not locked. "
        f"Allowed metrics: {', '.join(LOCKED_METRICS)}"
    )
```

**Risk:** LOW - Fails fast, clear error message  
**Rollback:** Change back to warning if blocking is too strict

---

### Patch 3: Add Tests for AnswerCorrectnessJudge
**Title:** `test(judges): Add unit tests for AnswerCorrectnessJudge`

**Why:** Coverage for `judges.py` is 48%; no tests for the new judge class.

**Patch Hint:**
```python
# backend/tests/test_judges.py (new file)
import pytest
from tunix_rt_backend.services.judges import AnswerCorrectnessJudge

def test_normalize_text():
    judge = AnswerCorrectnessJudge(db=None)
    assert judge._normalize_text("  Hello World!  ") == "hello world"
    assert judge._normalize_text("CAPS") == "caps"

@pytest.mark.asyncio
async def test_evaluate_perfect_score(test_db, golden_dataset):
    judge = AnswerCorrectnessJudge(db=test_db)
    run = create_completed_run(dataset_key="golden-v1", stdout="4")
    result = await judge.evaluate(run)
    assert result.metrics["answer_correctness"] == 1.0
```

**Risk:** LOW - Pure additive test code  
**Rollback:** Delete test file

---

### Patch 4: Document Seed Script Usage
**Title:** `docs(readme): Add golden dataset seeding instructions`

**Why:** New seed script exists but not documented for new developers.

**Patch Hint:**
```markdown
# Add to README.md after "Development Setup"

### Seeding the Golden Dataset

The golden dataset (`golden-v1`) is used for evaluation and regression testing.

# Start Postgres
docker compose up -d postgres

# Run seed script
cd backend
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5433/postgres"
python -m tools.seed_golden_dataset

# Verify
ls backend/datasets/golden-v1/manifest.json
```

**Risk:** LOW - Documentation only  
**Rollback:** Remove section

---

### Patch 5: Fix Empty Dataset Guardrail Test
**Title:** `test(datasets): Fix empty dataset guardrail test expectations`

**Why:** Test was modified to contradict the guardrail's purpose.

**Patch Hint:**
```python
# backend/tests/test_services_datasets.py:146
async def test_build_dataset_empty_result(test_db: AsyncSession) -> None:
    """Test that build fails when no traces match filters."""
    service = DatasetsBuilderService(test_db)
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        await service.build_dataset_manifest(
            dataset_name="empty",
            dataset_version="v1",
            filters={"nonexistent_tag": "xyz"}
        )
```

**Risk:** LOW - Restores original test intent  
**Rollback:** Revert to current assertion

---

## 9. Next Milestone Plan (M23 - fits in <1 day)

### Task 1: Complete AnswerCorrectnessJudge (90 min)
**Acceptance:** Judge loads golden-v1, compares predictions, returns correct scores. Coverage for `judges.py` ≥70%.

### Task 2: Add Judge Integration Tests (60 min)
**Acceptance:** E2E test that seeds golden-v1, runs eval, verifies score storage in `tunix_run_evaluations`.

### Task 3: Strengthen Tuning Guardrail (30 min)
**Acceptance:** `POST /api/tuning/jobs` returns 400 if `metric_name` not in `LOCKED_METRICS`.

### Task 4: Raise Backend Coverage to 75% (90 min)
**Acceptance:** `pytest --cov` reports ≥75% line coverage. Focus on `tuning_service.py` and `judges.py`.

### Task 5: Document Golden Dataset Workflow (30 min)
**Acceptance:** README.md includes seed script instructions and golden-v1 usage examples.

### Task 6: Fix Frontend Act Warnings (45 min)
**Acceptance:** `npm test` runs with zero React warnings.

### Task 7: Update tunix-rt.md with M22 Status (15 min)
**Acceptance:** M22 line updated with actual coverage numbers and golden-v1 status.

---

## 10. Machine-Readable Appendix (JSON)

```json
{
  "delta": {
    "base": "841fa73",
    "head": "cc4fdb3d2ec63e1833d908d740dbb25c21ce6c20",
    "files_changed": 25,
    "insertions": 1213,
    "deletions": 47
  },
  "quality_gates": {
    "lint_type_clean": "pass",
    "tests": "pass",
    "coverage_non_decreasing": "fail",
    "secrets_scan": "pass",
    "deps_cve_nonew_high": "pass",
    "schema_infra_migration_ready": "n/a",
    "docs_dx_updated": "pass"
  },
  "issues": [
    {
      "id": "Q-001",
      "file": "backend/tunix_rt_backend/services/judges.py:122-169",
      "category": "code_quality",
      "severity": "high",
      "summary": "AnswerCorrectnessJudge.evaluate() is a placeholder stub",
      "fix_hint": "Implement dataset loading and prediction comparison logic",
      "evidence": "Lines 155-157: score = 100.0 (hardcoded, not computed from dataset)"
    },
    {
      "id": "Q-002",
      "file": "backend/tunix_rt_backend/services/judges.py",
      "category": "tests",
      "severity": "high",
      "summary": "judges.py coverage at 48% (target: 75%)",
      "fix_hint": "Add unit tests for _normalize_text() and evaluate()",
      "evidence": "Coverage report: 81 of 175 lines uncovered"
    },
    {
      "id": "Q-003",
      "file": "backend/tunix_rt_backend/services/datasets_builder.py:89-91",
      "category": "code_quality",
      "severity": "medium",
      "summary": "Empty dataset guardrail test contradicts implementation",
      "fix_hint": "Restore pytest.raises(ValueError) assertion or remove guardrail",
      "evidence": "Guardrail raises ValueError but test expects manifest.exists()"
    },
    {
      "id": "Q-005",
      "file": "backend/tunix_rt_backend/services/tuning_service.py:128-132",
      "category": "code_quality",
      "severity": "medium",
      "summary": "Tuning metric guardrail is warning-only, not blocking",
      "fix_hint": "Change logger.warning() to raise ValueError()",
      "evidence": "training_readiness.md requires 'prevent starting' but code only warns"
    },
    {
      "id": "Q-006",
      "file": "frontend/src/App.test.tsx",
      "category": "tests",
      "severity": "low",
      "summary": "React act() warnings in frontend tests",
      "fix_hint": "Wrap state updates in waitFor() or act()",
      "evidence": "Test output: 'An update to App inside a test was not wrapped in act(...)'"
    }
  ],
  "metrics": {
    "backend_coverage_pct": 67.11,
    "backend_tests_passed": 214,
    "backend_tests_skipped": 13,
    "frontend_tests_passed": 45,
    "new_documentation_files": 2,
    "new_test_files": 1
  }
}
```

---

## Summary

**M22 Status:** SUBSTANTIAL PROGRESS, 2 CRITICAL GAPS

**Ship Decision:** ⚠️ **HOLD FOR FIXES**

**Blockers:**
1. `AnswerCorrectnessJudge` is a stub (HIGH severity)
2. Backend coverage regressed to 67% (target: ≥75%)

**Quick Wins to Ship:**
1. Complete judge implementation (90 min)
2. Add judge tests (60 min)
3. Re-run coverage report

**Recommendation:** Apply Patches 1-3, re-audit coverage, then ship M22 proper.
