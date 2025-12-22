# M12: Tunix Integration Documentation

**Status:** ✅ Complete (Phase 1)  
**Tunix Integration Type:** Mock-first, artifact-based (no runtime dependency)  
**Milestone:** M12 - Tunix Integration Skeleton + Run Manifest Pipeline

---

## Overview

M12 adds **Tunix integration** to tunix-rt as a **mock-first, artifact-based bridge**. Unlike UNGAR (which generates traces), Tunix integration focuses on **exporting datasets** and **generating training manifests** that can be executed by developers with Tunix installed on their local machines or TPU VMs.

### Key Design Decision: Mock-First

**M12 does NOT require Tunix to be installed.** The integration generates Tunix-compatible artifacts (JSONL exports + YAML manifests) without importing Tunix runtime. This means:

- ✅ Default CI remains green without Tunix
- ✅ Core functionality works for all developers
- ✅ Artifacts are portable and can be consumed by Tunix elsewhere
- ⏭️ Real Tunix runtime integration deferred to M13+

---

## Quick Start

### Installation

**No special installation required!** M12 works with standard tunix-rt installation:

```bash
cd backend
pip install -e ".[dev]"
```

Tunix integration is **always available** - no optional extra needed.

### Workflow Example

Complete workflow from dataset creation to manifest generation:

```bash
# 1. Build a dataset (using existing /api/datasets/build endpoint)
curl -X POST http://localhost:8000/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "my_training_data",
    "dataset_version": "v1",
    "limit": 100,
    "selection_strategy": "latest"
  }'

# 2. Export dataset in Tunix SFT format
curl -X POST http://localhost:8000/api/tunix/sft/export \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "my_training_data-v1"
  }' > training_data.jsonl

# 3. Generate Tunix training manifest
curl -X POST http://localhost:8000/api/tunix/sft/manifest \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "my_training_data-v1",
    "model_id": "google/gemma-2b-it",
    "output_dir": "./output/run_001",
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "batch_size": 8,
    "max_seq_length": 2048
  }' | jq -r '.manifest_yaml' > training_config.yaml

# 4. (Optional) Execute training locally with Tunix CLI
# tunix train --config training_config.yaml
```

---

## API Endpoints

### 1. Tunix Status

**Endpoint:** `GET /api/tunix/status`

**Description:** Check Tunix integration status and configuration.

**Response:**
```json
{
  "available": false,
  "version": null,
  "runtime_required": false,
  "message": "Tunix artifacts (JSONL + manifests) can be generated without Tunix runtime. Compatible with Tunix SFT workflows."
}
```

**Status Code:** Always `200 OK`

**M12 Behavior:**
- `available`: Always `false` (M12 is mock-first)
- `runtime_required`: Always `false` (artifact-based approach)
- `message`: Explains that artifacts can be generated without Tunix

**Example:**
```bash
curl http://localhost:8000/api/tunix/status
```

---

### 2. Export Traces in Tunix SFT Format

**Endpoint:** `POST /api/tunix/sft/export`

**Description:** Export traces in Tunix SFT format (JSONL). Reuses the `tunix_sft` export format from M09 (Gemma chat templates with reasoning steps).

**Request Body:**
```json
{
  "dataset_key": "my_dataset-v1",
  "trace_ids": null,
  "limit": 100
}
```

**Parameters:**
- `dataset_key` (optional): Dataset identifier to export
- `trace_ids` (optional): Array of specific trace IDs to export
- `limit` (optional, default 100): Maximum traces to export (1-10000)

**Note:** Either `dataset_key` OR `trace_ids` must be provided (not both).

**Response:** `application/x-ndjson` (JSONL content)

**JSONL Format (tunix_sft from M09):**
```jsonl
{"id": "550e8400-...", "prompts": "<start_of_turn>user\nWhat is 27 × 19?<end_of_turn>\n<start_of_turn>model\nReasoning:\n1. Parse the multiplication task\n2. Break down: 27 × 19 = 27 × (20 - 1)\n3. Calculate: 540 - 27 = 513\nAnswer: 513<end_of_turn>", "final_answer": "513", "metadata": {"created_at": "2025-12-22T10:00:00Z", "format": "tunix_sft", ...}}
```

**Status Codes:**
- `200 OK`: Export successful
- `400 Bad Request`: Neither dataset_key nor trace_ids provided
- `404 Not Found`: Dataset not found

**Examples:**

**Export from dataset:**
```bash
curl -X POST http://localhost:8000/api/tunix/sft/export \
  -H "Content-Type: application/json" \
  -d '{"dataset_key": "ungar_hcd-v1"}' > export.jsonl
```

**Export specific traces:**
```bash
curl -X POST http://localhost:8000/api/tunix/sft/export \
  -H "Content-Type: application/json" \
  -d '{
    "trace_ids": [
      "550e8400-e29b-41d4-a716-446655440000",
      "660f9500-f39c-52e5-b827-557766551111"
    ]
  }' > export.jsonl
```

---

### 3. Generate Tunix Training Manifest

**Endpoint:** `POST /api/tunix/sft/manifest`

**Description:** Generate a Tunix SFT (Supervised Fine-Tuning) training run manifest as YAML. The manifest can be executed locally by developers with Tunix installed.

**Request Body:**
```json
{
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./output/run_001",
  "learning_rate": 2e-5,
  "num_epochs": 3,
  "batch_size": 8,
  "max_seq_length": 2048
}
```

**Parameters:**
- `dataset_key` (required): Dataset identifier (must exist)
- `model_id` (required): Model identifier (e.g., "google/gemma-2b-it")
- `output_dir` (required): Output directory path for training artifacts
- `learning_rate` (optional, default 2e-5): Learning rate (0.0-1.0)
- `num_epochs` (optional, default 3): Number of training epochs (1-100)
- `batch_size` (optional, default 8): Batch size (1-512)
- `max_seq_length` (optional, default 2048): Maximum sequence length (128-32768)

**Response:**
```json
{
  "manifest_yaml": "version: \"1.0\"\nrunner: tunix\nmode: sft\n...",
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "format": "tunix_sft",
  "message": "Manifest generated for dataset my_dataset-v1. Save as YAML and execute with Tunix CLI."
}
```

**Generated YAML Structure:**
```yaml
version: "1.0"
runner: tunix
mode: sft
model:
  model_id: google/gemma-2b-it
dataset:
  format: tunix_sft
  path: ./datasets/my_dataset-v1.jsonl
training:
  learning_rate: 2.0e-05
  num_epochs: 3
  batch_size: 8
  max_seq_length: 2048
output:
  output_dir: ./output/run_001
```

**Status Codes:**
- `201 Created`: Manifest generated successfully
- `404 Not Found`: Dataset not found

**Example:**
```bash
curl -X POST http://localhost:8000/api/tunix/sft/manifest \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "ungar_hcd-v1",
    "model_id": "google/gemma-2b-it",
    "output_dir": "./output/run_001"
  }' | jq -r '.manifest_yaml' > config.yaml
```

---

## Frontend Usage

The frontend includes a **Tunix Integration panel** (similar to UNGAR panel) for easy artifact generation:

### Panel Features

1. **Status Display:**
   - Shows integration message
   - Displays "Runtime Required: No (Artifact-based)"

2. **Export & Manifest Form:**
   - Dataset Key input (e.g., "ungar_hcd-v1")
   - Model ID input (default: "google/gemma-2b-it")
   - Output Directory input (default: "./output/tunix_run")
   - **Export JSONL** button (downloads JSONL file)
   - **Generate Manifest** button (displays YAML)

3. **Results Display:**
   - Success message for manifest generation
   - Expandable YAML preview
   - Error messages for failures

**Test IDs:**
- `tunix:section` - Main panel container
- `tunix:status` - Status message
- `tunix:runtime-required` - Runtime requirement text
- `tunix:dataset-key` - Dataset key input
- `tunix:model-id` - Model ID input
- `tunix:output-dir` - Output directory input
- `tunix:export-btn` - Export button
- `tunix:manifest-btn` - Manifest button
- `tunix:manifest-result` - Results container
- `tunix:manifest-yaml` - YAML content
- `tunix:error` - Error message

---

## Export Format Details

### Tunix SFT Format (Reused from M09)

M12 reuses the **existing tunix_sft export format** established in M09. This format uses **Gemma chat templates** with reasoning steps.

**Record Structure:**
```json
{
  "id": "trace-uuid",
  "prompts": "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\nReasoning:\n1. {step1}\n2. {step2}\n...\nAnswer: {answer}<end_of_turn>",
  "final_answer": "answer text",
  "metadata": {
    "created_at": "ISO-8601 timestamp",
    "format": "tunix_sft",
    "trace_version": "1.0",
    ...
  }
}
```

**Why Reuse tunix_sft?**
- ✅ Already tested and validated in M09
- ✅ Gemma-aligned chat template
- ✅ Reasoning-aware (includes step-by-step thinking)
- ✅ No new schema complexity
- ✅ Consistent with training pipeline

See `docs/M09_DATASET_FORMAT.md` for complete format specification.

---

## Manifest Format Details

### YAML Structure

M12 generates **convention-based YAML manifests** designed for Tunix CLI consumption.

**Required Fields:**
- `version`: Manifest schema version (always "1.0")
- `runner`: Training runner (always "tunix")
- `mode`: Training mode (always "sft" in M12)
- `model.model_id`: HuggingFace model identifier
- `dataset.format`: Dataset format (always "tunix_sft")
- `dataset.path`: Path to JSONL dataset file
- `training.*`: Hyperparameters (learning_rate, num_epochs, batch_size, max_seq_length)
- `output.output_dir`: Output directory for checkpoints/logs

**Hyperparameter Defaults:**

| Parameter | Default | Range |
|-----------|---------|-------|
| `learning_rate` | `2e-5` | 0.0 - 1.0 |
| `num_epochs` | `3` | 1 - 100 |
| `batch_size` | `8` | 1 - 512 |
| `max_seq_length` | `2048` | 128 - 32768 |

**Note:** M12 does NOT validate against Tunix CLI. Manifests are **best-effort** based on common SFT patterns. Future milestones may add schema validation when Tunix documentation stabilizes.

---

## Testing

### Default Tests (No Tunix Required)

M12 includes **14 comprehensive tests** that all run without Tunix installed:

```bash
cd backend
pytest tests/test_tunix.py -v
```

**Test Categories:**
1. **Availability Tests (3):** Verify mock-first behavior
2. **Status Endpoint (1):** Check /api/tunix/status response
3. **Export Endpoint Tests (3):** Validate JSONL export
4. **Manifest Endpoint Tests (3):** Validate YAML generation
5. **Service Layer Tests (3):** Test business logic
6. **End-to-End (1):** Complete workflow test

**All tests pass with exit code 0** and contribute to 91.59% overall coverage.

### CI/CD

**Default CI:** Runs all tests without Tunix (fast, stable)

**Optional Tunix CI:** `.github/workflows/tunix-integration.yml`
- Trigger: Manual dispatch or nightly at 3 AM UTC
- Non-blocking (`continue-on-error: true`)
- Runs Tunix-specific tests
- Validates JSONL export format
- Validates YAML manifest structure
- Uploads coverage reports

**Manual Trigger:**
```bash
# Via GitHub UI: Actions > Tunix Integration (Optional) > Run workflow
# Or via gh CLI:
gh workflow run tunix-integration.yml
```

---

## Architecture

### Design Principles

1. **Mock-First:** No Tunix runtime dependency in M12
2. **Artifact-Based:** Generate portable JSONL + YAML
3. **Reuse Existing:** Leverage tunix_sft format from M09
4. **Service Layer:** Business logic in services/, not app.py
5. **Graceful Degradation:** Never fail, always provide artifacts

### File Structure

```
backend/tunix_rt_backend/
├── integrations/tunix/
│   ├── __init__.py
│   ├── availability.py      # Mock-first availability checks
│   └── manifest.py           # YAML manifest builder
├── services/
│   └── tunix_export.py       # JSONL export service (reuses tunix_sft)
├── schemas/
│   └── tunix.py              # Request/response schemas
└── app.py                    # Thin controller endpoints

backend/tests/
└── test_tunix.py             # 14 comprehensive tests

frontend/src/
├── api/client.ts             # Tunix API client functions
├── App.tsx                   # Tunix panel UI
└── App.test.tsx              # 5 frontend tests
```

### Data Flow

**Export Flow:**
```
Client Request
  ↓
POST /api/tunix/sft/export (app.py)
  ↓
export_tunix_sft_jsonl() (services/tunix_export.py)
  ↓
export_dataset_to_jsonl() (services/datasets_export.py)
  ↓
_build_tunix_sft_record() (services/datasets_export.py)
  ↓
render_tunix_sft_prompt() (training/renderers.py)
  ↓
JSONL Response (application/x-ndjson)
```

**Manifest Flow:**
```
Client Request
  ↓
POST /api/tunix/sft/manifest (app.py)
  ↓
load_manifest() - verify dataset exists
  ↓
build_sft_manifest() (integrations/tunix/manifest.py)
  ↓
yaml.dump() - serialize to YAML
  ↓
TunixManifestResponse (JSON with manifest_yaml field)
```

---

## Limitations (M12 Scope)

### By Design (Mock-First)

1. **No Tunix Runtime:** M12 does NOT execute training
2. **No CLI Validation:** Manifests are best-effort, not validated against Tunix
3. **SFT Only:** Only Supervised Fine-Tuning manifests (no GRPO/PPO/DPO)
4. **No TPU Orchestration:** Local execution only

### Future Expansion (M13+)

**M13 could add:**
- Real Tunix runtime integration (optional)
- Training job execution adapter
- Run registry for tracking training runs
- RL pipeline support (GRPO, PPO, DPO)
- LLM-as-judge scoring integration

**M14 could add:**
- Training result ingestion
- Checkpoint management
- Evaluation loop closure (trace → train → compare)

---

## Troubleshooting

### Export Returns 400 "dataset_key or trace_ids"

**Cause:** Request body missing both `dataset_key` and `trace_ids`

**Solution:**
```bash
# Provide dataset_key
curl -X POST .../tunix/sft/export \
  -d '{"dataset_key": "my_dataset-v1"}'

# OR provide trace_ids
curl -X POST .../tunix/sft/export \
  -d '{"trace_ids": ["uuid1", "uuid2"]}'
```

### Export Returns 404 "Dataset not found"

**Cause:** Dataset manifest doesn't exist

**Solution:**
```bash
# 1. List datasets directory
ls backend/datasets/

# 2. Build dataset if missing
curl -X POST http://localhost:8000/api/datasets/build \
  -d '{"dataset_name": "my_dataset", "dataset_version": "v1", ...}'
```

### Manifest Returns 404 "Dataset not found"

**Cause:** Dataset manifest must exist before generating manifest

**Solution:**
```bash
# Always build/export dataset before generating manifest
curl -X POST .../datasets/build -d '{...}'
curl -X POST .../tunix/sft/manifest -d '{"dataset_key": "..."}'
```

### Frontend Shows "Failed to fetch status"

**Cause:** Backend not running or CORS issue

**Solution:**
```bash
# 1. Verify backend is running
curl http://localhost:8000/api/health

# 2. Check CORS configuration in app.py (should allow localhost:5173)

# 3. Restart frontend dev server
cd frontend && npm run dev
```

---

## Best Practices

### 1. Dataset Organization

**Convention:** Use descriptive dataset keys with versions
```bash
# Good
my_training_data-v1
ungar_hcd_baseline-v2
eval_set_202512-v1

# Avoid
dataset1
test
data
```

### 2. Manifest Naming

**Convention:** Include model and date in output directory
```yaml
output:
  output_dir: ./output/gemma2b_20251222_run001
```

### 3. Hyperparameter Tuning

**Start with defaults**, then iterate:
```bash
# First run: Use defaults
curl -X POST .../manifest \
  -d '{"dataset_key": "...", "model_id": "...", "output_dir": "..."}'

# Second run: Tune based on results
curl -X POST .../manifest \
  -d '{..., "learning_rate": 1e-4, "num_epochs": 5}'
```

### 4. Export Workflow

**Recommended order:**
1. Build dataset (`POST /api/datasets/build`)
2. Export JSONL (`POST /api/tunix/sft/export`)
3. Generate manifest (`POST /api/tunix/sft/manifest`)
4. Execute training locally (with Tunix CLI)

---

## References

### Internal Documentation
- `docs/M09_DATASET_FORMAT.md` - tunix_sft format specification
- `docs/M09_TRAINING_QUICKSTART.md` - Training workflows
- `docs/M07_UNGAR_INTEGRATION.md` - Optional integration pattern
- `docs/M12_BASELINE.md` - Pre-M12 state

### External Resources
- **Tunix Repository:** https://github.com/google/tunix (if available)
- **Tunix Documentation:** https://tunix.readthedocs.io/ (if available)
- **Gemma Models:** https://huggingface.co/google/gemma-2b-it

---

## Changelog

### M12 (December 22, 2025)
- ✅ Added Tunix integration (mock-first, artifact-based)
- ✅ Added 3 API endpoints (/status, /sft/export, /sft/manifest)
- ✅ Reused tunix_sft export format from M09
- ✅ Added YAML manifest generation
- ✅ Added frontend Tunix panel
- ✅ Added 14 backend tests + 5 frontend tests
- ✅ Added optional CI workflow
- ✅ Coverage: 91.59% (160 tests passing)

---

**M12 Status:** ✅ **COMPLETE**  
**Next Milestone:** M13 - Real Tunix Execution Hooks (Optional)  
**Documentation Version:** 1.0  
**Last Updated:** December 22, 2025
