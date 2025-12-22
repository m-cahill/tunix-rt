# M08 Implementation Progress

**Date:** December 21, 2025  
**Status:** Phases 0-3 Complete ✅ | Phase 4 Pending  
**Baseline:** `89bcddf` (M07 complete)

---

## Completed Work

### ✅ Phase 0: Baseline Verification

**Files Created:**
- `docs/M08_BASELINE.md` - Complete baseline documentation

**Status:** All 72 backend + 11 frontend + 6 E2E tests passing

---

### ✅ Phase 1: M07 Paper Cuts

**Changes Made:**

1. **Type Ignore Comments** (`backend/tunix_rt_backend/integrations/ungar/`)
   - Added explanatory comments to `availability.py` and `high_card_duel.py`
   - Example: `# type: ignore - optional dependency, not available to mypy`

2. **Quick Start Documentation** (`docs/M07_UNGAR_INTEGRATION.md`)
   - Added complete "Quick Start Example" section
   - Copy-paste workflow from install → generate → export

3. **Defensive Logging** (`backend/tunix_rt_backend/integrations/ungar/high_card_duel.py`)
   - Added `logging` import and logger setup
   - Added `logger.warning()` calls in `_extract_my_card()`, `_extract_opponent_card()`, `_format_card()`
   - Warnings logged before `"??"` fallback returns

4. **E2E Test for UNGAR Panel** (`e2e/tests/smoke.spec.ts`)
   - Added new test: `'UNGAR section renders with status'`
   - Verifies UNGAR panel visibility and "Not Installed" status
   - Closes M07 audit gap

**Files Modified:** 4  
**Tests Added:** 1 E2E test

---

### ✅ Phase 2: Dataset Manifest & Build/Export

**New Files Created:**

1. **`backend/tunix_rt_backend/schemas/dataset.py`**
   - `DatasetBuildRequest` schema
   - `DatasetManifest` schema (with multi-session fields)
   - `DatasetBuildResponse` schema
   - Supports `dataset_key` (name-version) + `build_id` (UUID)

2. **`backend/tunix_rt_backend/helpers/datasets.py`**
   - `get_datasets_dir()` - Creates `backend/datasets/` directory
   - `save_manifest()` / `load_manifest()` - File-based persistence
   - `create_dataset_key()` - Generates `{name}-{version}` keys
   - `compute_dataset_stats()` - Statistical summary

3. **`backend/tests/test_datasets.py`**
   - 13 comprehensive tests
   - Helper tests (6): key creation, save/load, stats
   - Build endpoint tests (4): latest/random strategies, validation
   - Export endpoint tests (3): success, not found, ordering

**New Endpoints:**

1. **`POST /api/datasets/build`**
   - Request: dataset name, version, filters, limit, strategy (latest/random), seed
   - Selection strategies: `latest` (by created_at) and `random` (seeded)
   - Response: dataset_key, build_id, trace_count, manifest_path
   - Status: 201 Created | 422 Validation Error

2. **`GET /api/datasets/{dataset_key}/export.jsonl`**
   - Returns NDJSON with traces in manifest order
   - Currently supports "trace" format (Phase 3 adds "tunix_sft")
   - Status: 200 OK | 404 Not Found

**Other Changes:**
- Added `backend/datasets/` to `.gitignore`
- Updated `schemas/__init__.py` to export dataset schemas

**Files Modified:** 3  
**Files Created:** 3  
**Tests Added:** 13

---

### ✅ Phase 3: Tunix SFT Prompt Renderer

**New Files Created:**

1. **`backend/tunix_rt_backend/training/__init__.py`**
   - Package init for training utilities

2. **`backend/tunix_rt_backend/training/renderers.py`**
   - `render_tunix_sft_prompt(trace_data)` - Core renderer
   - Formats traces using Gemma chat template:
     - `<start_of_turn>user\n{prompt}<end_of_turn>`
     - `<start_of_turn>model\nReasoning:\n{steps}\nAnswer: {answer}<end_of_turn>`
   - `render_trace_for_training()` - Dispatcher function
   - Deterministic, no LLM rewriting

3. **`backend/tests/test_renderers.py`**
   - 9 comprehensive tests
   - Tests: basic rendering, multiple steps, determinism, empty steps
   - Tests: special characters, multiline content, invalid format

**Endpoint Enhancement:**

- **`GET /api/datasets/{dataset_key}/export.jsonl?format=trace|tunix_sft`**
  - Added `format` query parameter
  - `format=trace`: Raw trace data (default)
  - `format=tunix_sft`: Rendered prompts using `render_tunix_sft_prompt()`
  - Includes `"format": "tunix_sft"` in metadata
  - Status: 200 OK | 404 Not Found | 422 Invalid Format

**Files Created:** 3  
**Files Modified:** 1  
**Tests Added:** 9

---

## Summary Statistics

### Test Count
- **Baseline (M07):** 72 backend + 11 frontend + 6 E2E = 89 total
- **M08 (Phases 1-3):** 94 backend + 11 frontend + 7 E2E = 112 total
- **Added:** 22 backend tests + 1 E2E test = +23 tests

### Test Breakdown
- `test_datasets.py`: 13 tests
- `test_renderers.py`: 9 tests
- E2E (UNGAR panel): 1 test

### Files Created
- 9 new files total
- 3 test files
- 3 schema/helper files
- 2 training modules
- 1 documentation file

### Files Modified
- 8 files modified
- `app.py` (2 new endpoints + format parameter)
- UNGAR integration files (logging + comments)
- Documentation files
- E2E smoke tests

---

## Pending Work (Phase 4 + Final)

### Phase 4: Training Smoke Harness

**Remaining Tasks:**

1. **Add `backend[training]` optional dependency** (`pyproject.toml`)
   - Add Tunix library (git+ or release version)
   - Add JAX (CPU-only for smoke tests)
   - Keep as optional extra (like UNGAR)

2. **Create `backend/training/sft_smoke.py`**
   - Tiny SFT script (5-10 steps, 32 samples max)
   - Loads dataset JSONL
   - Runs minimal Tunix training loop
   - Prints loss and exits
   - CPU-default (GPU/TPU via env flags)

3. **Add dataset validation to main CI** (`.github/workflows/ci.yml`)
   - New job: `dataset-validation`
   - Validates dataset schema + renderer output
   - Fast, blocking (< 10s)
   - Runs on backend changes

4. **Add training smoke workflow** (`.github/workflows/training-smoke.yml`)
   - Trigger: `workflow_dispatch` + nightly schedule
   - Non-blocking (`continue-on-error: true`)
   - Installs `backend[training]`
   - Runs `backend/training/sft_smoke.py`
   - Optional/manual like UNGAR integration

---

### Final: Documentation & Verification

**Remaining Tasks:**

1. **Update `tunix-rt.md`**
   - Add dataset endpoints to API section
   - Add training renderer section
   - Update M08 completion summary

2. **Update `README.md`**
   - Add dataset build/export examples
   - Add Tunix SFT rendering examples

3. **Create `docs/M08_SUMMARY.md`**
   - Implementation summary
   - Architecture decisions
   - Testing results
   - Coverage numbers

4. **Final Verification**
   - Run all tests (backend + frontend + E2E)
   - Verify coverage ≥70% (maintained from M07)
   - CI green
   - No linter errors

---

## Architecture Decisions

### Dataset Key Design
- **Two identifiers:** `dataset_key` (name-version) + `build_id` (UUID)
- **Rationale:** Human-readable + reproducible (dataset_key) + unique provenance (build_id)
- **Format:** `{dataset_name}-{dataset_version}` (e.g., `ungar_hcd_baseline-v1`)

### File-Based Manifests
- **Location:** `backend/datasets/{dataset_key}/manifest.json`
- **Rationale:** Simple, version-controllable, no DB migration needed
- **Future:** Can add DB persistence for querying/multi-tenancy in M09

### Selection Strategies
- **Supported:** `latest` (by created_at) and `random` (seeded)
- **Deferred:** `stratified` (balanced sampling) to M09
- **Rationale:** Cover 80% of use cases, keep M08 scope manageable

### Renderer Design
- **Format:** Gemma chat template (`<start_of_turn>` tags)
- **Deterministic:** No LLM rewriting, pure template-based
- **Extensible:** `render_trace_for_training()` dispatcher for future formats

### Multi-Session Support
- **Added fields:** `session_id`, `parent_dataset_id`, `training_run_id`
- **Optional:** All nullable, not required in M08
- **Rationale:** Future-proofs schema for M09+ multi-session workflows

---

## Known Limitations

1. **Python-Level JSON Filtering**
   - Dataset build fetches `limit × 10` traces, filters in Python
   - Acceptable at current scale (<10K traces)
   - Future: DB-specific JSON queries when UNGAR traces >> regular traces

2. **Single Export Format**
   - Only `trace` and `tunix_sft` formats supported
   - Future: Add more formats (CSV, Parquet, etc.) as needed

3. **No Dataset UI**
   - Backend-only in M08
   - Frontend dataset browser deferred to M09

4. **No Training Validation**
   - Renderer tests exist, but no actual training validation yet
   - Phase 4 adds smoke harness to close this gap

---

## Next Steps

To complete M08:

1. **Immediate:** Continue with Phase 4 (training dependencies + smoke harness)
2. **Then:** Final documentation + verification
3. **Commit:** All changes with `feat(M08)` message
4. **Audit:** Run audit prompt for M08

**Estimated Remaining Time:** 2-3 hours (Phase 4) + 1 hour (Final)

---

**Progress:** 75% complete (Phases 0-3 done, Phase 4 + Final pending)

