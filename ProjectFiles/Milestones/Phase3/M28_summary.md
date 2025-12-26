# M28 Milestone Completion Summary

**Milestone:** M28 - Hyperparameter Tuning & Leaderboard  
**Status:** ✅ Complete  
**Completion Date:** December 25, 2025  
**Commits:** 4 (`6288467`, `a696052`, `83cd6d1`)  
**CI Status:** All 8 jobs green ✅

---

## 1. Milestone Goals (from M28_plan.md)

| Goal | Status | Notes |
|------|--------|-------|
| Quick Wins / Audit Patch Set | ✅ | `.gitignore`, `tunix-rt.md` updates |
| Hyperparameter Sweep (Ray Tune) | ✅ | 3+ trials, search space validation |
| Run Comparison UI | ✅ | Side-by-side modal with loss curves |
| Leaderboard Integration | ✅ | `answer_correctness` scoring wired |
| UNGAR Closure | ✅ | Episode API fix (preferred option) |
| CI Hygiene | ✅ | ruff, cyclonedx fixes |

---

## 2. Key Deliverables

### 2.1 Hyperparameter Tuning Infrastructure

**Files Modified/Created:**
- `backend/tunix_rt_backend/services/tuning_service.py` - Ray Tune integration
- `backend/tunix_rt_backend/db/models/tuning.py` - `search_space_json` column
- `scripts/run_m28_sweep.py` - Helper script for triggering sweeps

**Capabilities:**
- Create tuning jobs via `POST /api/tuning/jobs`
- Support for `choice`, `uniform`, `loguniform`, `randint` search spaces
- Automatic best-trial selection
- Trial results persisted to `tunix_tuning_trials` table

**Example Usage:**
```bash
cd backend
uv run python ../scripts/run_m28_sweep.py
```

---

### 2.2 Run Comparison UI

**Files Created:**
- `frontend/src/components/RunComparison.tsx` - Comparison modal component
- `frontend/src/components/RunComparison.test.tsx` - 5 unit tests

**Features:**
- Side-by-side run details (model, dataset, status, duration)
- Loss curve overlay chart (SVG-based)
- Evaluation score diff with color-coded deltas
- Deep-link support: `/?runA={uuid}&runB={uuid}`

---

### 2.3 Leaderboard Integration

**Metric:** `answer_correctness`  
**Definition:** Mean of normalized exact match scores across evaluation set  
**Range:** 0.0 to 100.0 (percentage)

**Source:** `AnswerCorrectnessJudge` in `backend/tunix_rt_backend/services/judges.py`

---

### 2.4 UNGAR Integration Fix

**Problem:** UNGAR 0.2+ changed `play_random_episode()` to return an `Episode` object instead of a dictionary, causing `AttributeError: 'Episode' object has no attribute 'get'`.

**Solution:** Updated `high_card_duel.py` to handle both API formats:

```python
# Handle Episode object (new API)
if hasattr(episode_result, "states"):
    states = episode_result.states
    initial_state = states[0] if states else None
    returns = episode_result.rewards if hasattr(episode_result, "rewards") else [0, 0]
else:
    # Handle legacy dict format (fallback)
    initial_state = episode_result.get("initial_state")
    returns = episode_result.get("returns", [0, 0])
```

**Files Modified:**
- `backend/tunix_rt_backend/integrations/ungar/high_card_duel.py`
  - `_convert_episode_to_trace()` - Handle Episode object
  - `_extract_opponent_card()` - Handle Episode object

---

### 2.5 CI/CD Fixes

| Issue | Fix | Commit |
|-------|-----|--------|
| `ruff format` failure on `test_tunix_registry.py` | Applied `ruff format` | `83cd6d1` |
| `cyclonedx-py --outfile` deprecated | Changed to `--output-file` | `a696052` |

---

## 3. Test Results

### Backend (Python)
```
234 passed, 11 skipped, 8 warnings in 102.94s
Coverage: 72.80% (gate: 70.0%) ✅
```

### Frontend (TypeScript/React)
```
5 test files, 49 tests passed
```

### E2E (Playwright)
```
8 tests passed
```

---

## 4. Documentation Updates

| Document | Update |
|----------|--------|
| `tunix-rt.md` | Added M27 description, M28 stub |
| `docs/m28_tuning_and_comparison.md` | Created - tuning/comparison guide |
| `.gitignore` | Added `training_log*.txt`, `eval_log.txt`, `backend/artifacts/` |

---

## 5. Files Changed Summary

| Category | Files |
|----------|-------|
| Backend Services | `tuning_service.py`, `high_card_duel.py` |
| Backend Tests | `test_m28_tuning.py`, `test_services_ungar.py`, `test_tunix_registry.py` |
| Frontend Components | `RunComparison.tsx`, `RunComparison.test.tsx` |
| CI/CD | `.github/workflows/ci.yml` |
| Config | `.gitignore`, `tunix-rt.md` |
| Docs | `docs/m28_tuning_and_comparison.md` |
| Scripts | `scripts/run_m28_sweep.py` |

---

## 6. Known Limitations

1. **UNGAR Tests Skip in CI** - 10 tests skip because `ungar` is not installed in default CI. Tests pass when installed locally.

2. **Ray Tune Local Only** - Tuning runs locally; distributed execution not tested.

3. **Comparison UI Basic** - SVG chart is functional but not interactive (no hover, zoom).

---

## 7. Next Steps (M29 Preview)

Based on the M28 audit recommendations:

1. **Extract Routes** - Modularize `app.py` (1,563 lines) into `routers/` submodules
2. **Resolve TODOs** - Address 3 remaining TODO markers in services
3. **Performance Baseline** - Add `pytest-benchmark` markers for critical paths
4. **Data Scale-Up** - Expand beyond `golden-v2` for competition readiness

---

## 8. Verification Checklist

- [x] Hyperparameter sweep creates 3+ trials
- [x] Run comparison UI loads two runs side-by-side
- [x] Leaderboard displays `answer_correctness` score
- [x] UNGAR integration tests pass (when installed)
- [x] CI pipeline fully green (8/8 jobs)
- [x] Documentation updated
- [x] Pre-commit hooks pass
- [x] Changes pushed to GitHub

---

**M28 Complete.** 🎉
