# M32 Milestone Summary — Data Scale-Up & Coverage Uplift

**Status:** ✅ Complete  
**Completion Date:** December 26, 2025  
**Branch:** `milestone/M32-data-scale-up`  
**CI Status:** Green (all checks passing)

---

## Overview

M32 focused on three main objectives:
1. **Submission execution evidence** — Reproducible runbook and evidence capture
2. **Data scale-up** — Scaled dataset from 200 to 550+ traces with strict schema
3. **Coverage uplift** — Added tests for `datasets_ingest.py` and `worker.py`

**Acceptance Criteria (All Met):**
1. ✅ dev-reasoning-v2 exists with 550 valid traces (schema-validated)
2. ✅ Smoke training on dev-reasoning-v2 succeeds
3. ✅ New unit tests for datasets_ingest.py (9 tests, 0%→full coverage)
4. ✅ New edge case tests for worker.py (7 new tests)
5. ✅ Packaging tool bundles evidence files
6. ✅ CI green

---

## Deliverables Completed

### Phase 0 — Baseline Gate ✅

| Check | Result |
|-------|--------|
| `ruff format --check` | 124 files formatted |
| `ruff check` | All checks passed |
| `mypy` | 66 files, no issues |
| `pytest` | 236 passed, 11 skipped (baseline) |
| `npm test` | All passed |

### Phase 1 — Submission Execution Evidence ✅

**Created:** `docs/submission_execution_m32.md`
- Step-by-step runbook for Kaggle execution
- Evidence capture instructions
- Troubleshooting guide

**Created:** `submission_runs/m32_v1/`
- `run_manifest.json` — Run configuration template
- `eval_summary.json` — Evaluation results template
- `kaggle_output_log.txt` — Console output placeholder

**Updated:** `.gitignore`
- Track small evidence files
- Ignore large artifacts (checkpoints, models)

**Updated:** `backend/tools/package_submission.py`
- Added `docs/submission_execution_m32.md`
- Added `submission_runs/m32_v1/run_manifest.json`
- Added `submission_runs/m32_v1/eval_summary.json`
- Added `datasets/dev-reasoning-v2/manifest.json`
- Updated archive prefix to `tunix_rt_m32`

### Phase 2 — Data Scale-Up ✅

**Created:** `backend/tools/seed_dev_reasoning_v2.py`

| Metric | Value |
|--------|-------|
| Total traces | 550 |
| Reasoning (70%) | 385 |
| Synthetic (20%) | 110 |
| Golden-style (6.4%) | 35 |
| Edge cases (3.6%) | 20 |
| Schema | Strict ReasoningTrace with `steps: [{i, type, content}, ...]` |
| Seed | 42 (deterministic) |

**Dataset Composition:**
- **Reasoning:** Multi-step math (distance, percentage, cost), decomposition, verification
- **Synthetic:** Basic arithmetic, string reverse, string sort, simple logic
- **Golden-style:** Text repetition, word counting, uppercase transforms
- **Edge cases:** Minimal traces, long prompts, special chars, Unicode, whitespace

**Created:** `backend/datasets/dev-reasoning-v2/`
- `dataset.jsonl` (550 lines)
- `manifest.json` (with stats and provenance)

**Created:** `backend/tests/test_dev_reasoning_v2_schema.py`

| Test | Description |
|------|-------------|
| `test_dataset_files_exist` | Verify files exist |
| `test_manifest_has_required_fields` | Validate manifest |
| `test_all_traces_validate_against_reasoning_trace_schema` | Full schema validation |
| `test_sampled_traces_have_correct_metadata` | Metadata structure |
| `test_dataset_composition_matches_manifest` | Composition accuracy |
| `test_step_indices_are_sequential` | Step ordering |
| `test_no_duplicate_step_indices` | Index uniqueness |
| `test_edge_cases_exist` | Edge case presence |

**Smoke Training Verified:**
```
python training/train_jax.py \
  --config training/configs/sft_tiny.yaml \
  --output ./output/smoke_test_m32 \
  --dataset dev-reasoning-v2 \
  --device cpu \
  --smoke_steps 2

🚀 Starting SFT Training (JAX/Flax)...
   🛑 Smoke steps limit reached (2). Stopping.
```

Output artifacts created: `checkpoints/1/`, `metrics.jsonl`

### Phase 3 — Coverage Uplift ✅

**Created:** `backend/tests/test_datasets_ingest.py`

| Test | Coverage |
|------|----------|
| `test_ingest_valid_jsonl_happy_path` | Happy path with 2 traces |
| `test_ingest_skips_invalid_json_lines` | Invalid JSON handling |
| `test_ingest_skips_invalid_trace_schema` | Schema violation handling |
| `test_ingest_file_not_found` | FileNotFoundError |
| `test_ingest_path_is_directory` | Directory path handling |
| `test_ingest_empty_file_raises_value_error` | Empty file handling |
| `test_ingest_only_invalid_traces_raises_value_error` | All-invalid handling |
| `test_ingest_skips_empty_lines` | Whitespace handling |
| `test_ingest_adds_source_metadata` | Metadata injection |

**Updated:** `backend/tests/test_worker.py`

| Test | Coverage |
|------|----------|
| `test_process_run_safely_missing_config` | Null config handling |
| `test_process_run_safely_empty_config` | Empty config handling |
| `test_process_run_safely_with_duration_metrics` | Metrics recording |
| `test_process_run_safely_auto_eval_on_completion` | Auto-evaluation trigger |
| `test_process_run_safely_skips_eval_on_dry_run` | Dry-run skip |
| `test_process_run_safely_eval_error_does_not_crash` | Evaluation error handling |
| `test_process_run_safely_sets_completed_at_on_failure` | Failure timestamp |

**Documented:** Postgres-only `claim_pending_run` limitation (SKIP LOCKED semantics)

### Phase 4 — CI Verification ✅

| Metric | Before | After |
|--------|--------|-------|
| Backend tests | 236 | 260 |
| New tests | - | +24 |
| Coverage (line) | ~70% | ~70%+ |
| ruff errors | 0 | 0 |
| mypy errors | 0 | 0 |

---

## Files Changed/Added

### New Files (8)

| File | Purpose |
|------|---------|
| `docs/submission_execution_m32.md` | Execution runbook |
| `submission_runs/m32_v1/run_manifest.json` | Run config template |
| `submission_runs/m32_v1/eval_summary.json` | Eval results template |
| `submission_runs/m32_v1/kaggle_output_log.txt` | Output placeholder |
| `backend/tools/seed_dev_reasoning_v2.py` | Dataset seeder (550 traces) |
| `backend/datasets/dev-reasoning-v2/dataset.jsonl` | Dataset file |
| `backend/datasets/dev-reasoning-v2/manifest.json` | Dataset manifest |
| `backend/tests/test_dev_reasoning_v2_schema.py` | Schema validation (8 tests) |
| `backend/tests/test_datasets_ingest.py` | Ingest tests (9 tests) |

### Modified Files (5)

| File | Change |
|------|--------|
| `.gitignore` | Added submission_runs patterns, dataset exceptions |
| `backend/tools/package_submission.py` | Added evidence files, updated prefix to m32 |
| `backend/tests/test_worker.py` | Added 7 edge case tests, Postgres docs |
| `tunix-rt.md` | Updated to M32 status |
| `ProjectFiles/Milestones/Phase3/M32_questions.md` | Clarifying questions |

---

## Metrics Summary

| Metric | Value | Gate |
|--------|-------|------|
| Backend Coverage (Line) | ~70% | ≥70% ✅ |
| Backend Tests | 260 passed | All pass ✅ |
| New Tests Added | 24 | - |
| mypy Errors | 0 | 0 ✅ |
| Ruff Errors | 0 | 0 ✅ |
| dev-reasoning-v2 traces | 550 | ≥500 ✅ |
| Packaging size | 65.5 KB | <100 KB ✅ |

---

## Commands Reference

### Generate dev-reasoning-v2
```bash
cd backend
uv run python tools/seed_dev_reasoning_v2.py
```

### Run Schema Tests
```bash
cd backend
uv run pytest tests/test_dev_reasoning_v2_schema.py -v
```

### Run Ingest Tests
```bash
cd backend
uv run pytest tests/test_datasets_ingest.py -v
```

### Run Worker Tests
```bash
cd backend
uv run pytest tests/test_worker.py -v
```

### Smoke Training on dev-reasoning-v2
```bash
python training/train_jax.py \
  --config training/configs/sft_tiny.yaml \
  --output ./output/smoke_test_m32 \
  --dataset dev-reasoning-v2 \
  --device cpu \
  --smoke_steps 2
```

### Package Submission
```bash
python backend/tools/package_submission.py
```

---

## Architecture Status (Post-M32)

```
tunix-rt/
├── app.py (56 lines) — FastAPI with router registration
├── routers/ (10 modules) — HTTP endpoint handlers
├── services/ (15 modules) — Business logic
├── db/ — SQLAlchemy models + Alembic migrations
├── training/ — JAX/Flax pipeline + submission configs
├── tools/ — seed_dev_reasoning_v2.py (new), package_submission.py
├── datasets/
│   ├── dev-reasoning-v1/ (200 traces)
│   ├── dev-reasoning-v2/ (550 traces, strict schema) ← NEW
│   └── golden-v2/ (100 traces)
├── submission_runs/m32_v1/ — Evidence capture ← NEW
├── e2e/tests/ (3 specs) — Playwright E2E
├── tests/ (37 files, 260 tests) — Backend unit tests
└── docs/ (50+ files) — ADRs, guides, runbooks
```

---

## Next Steps (M33+)

### Immediate
1. Execute training run on Kaggle with `dev-reasoning-v2`
2. Fill in evidence files with actual run data
3. Record and upload video

### Future
1. **Tuning sweep** — Hyperparameter optimization on dev-reasoning-v2
2. **Evaluation loop** — Compare models trained on v1 vs v2
3. **Production hardening** — Finalize submission package

---

## Conclusion

M32 achieved all acceptance criteria:
- ✅ dev-reasoning-v2 dataset with 550 traces (strict ReasoningTrace schema)
- ✅ Smoke training verified on new dataset
- ✅ 24 new tests (9 ingest, 8 schema, 7 worker)
- ✅ Coverage for datasets_ingest.py (was 0%)
- ✅ Evidence capture runbook and folder structure
- ✅ Packaging tool updated with new artifacts
- ✅ CI green

**Test Count: 260 passed (+24 from M31)**
