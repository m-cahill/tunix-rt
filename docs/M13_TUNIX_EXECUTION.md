# M13: Tunix Runtime Execution Guide

**Date:** December 22, 2025  
**Milestone:** M13 (Tunix Runtime Execution - Phase 2)  
**Status:** ✅ Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Execution Modes](#execution-modes)
4. [Installation & Setup](#installation--setup)
5. [API Reference](#api-reference)
6. [Frontend Integration](#frontend-integration)
7. [Testing Strategy](#testing-strategy)
8. [CI/CD Integration](#cicd-integration)
9. [Troubleshooting](#troubleshooting)
10. [Future Enhancements (Out of Scope for M13)](#future-enhancements)

---

## Executive Summary

M13 introduces **optional, gated execution** of Tunix training runs using artifacts generated in M12. This milestone enables actual training execution while maintaining the following principles:

- **Default CI passes with no Tunix installed** ✅
- **Tunix execution is opt-in** via `backend[tunix]` extra
- **Graceful degradation** with 501 status when Tunix unavailable and `dry_run=false`
- **No TPU assumptions** - CPU/GPU only for M13
- **No coupling in core code paths** - follows UNGAR optional-integration pattern

### Key Capabilities

1. **Dry-run mode (default)**: Validates parameters without invoking Tunix CLI
2. **Local execution mode**: Runs `tunix train` via subprocess, captures output
3. **Execution metadata**: Returns run_id, status, timestamps, stdout/stderr (truncated), exit_code
4. **Frontend UI**: Minimal "Run with Tunix" buttons with status/log display

### Out of Scope for M13

- Training result ingestion (deferred to M14)
- Evaluation metrics computation
- TPU orchestration
- Background/async execution workers
- Result persistence (database storage)
- Benchmarking and performance tuning

---

## Architecture Overview

### Service Layer: `TunixExecutionService`

The core business logic is encapsulated in `tunix_rt_backend/services/tunix_execution.py`:

```python
async def execute_tunix_run(request: TunixRunRequest, db: AsyncSession) -> TunixRunResponse:
    """
    Execute a Tunix training run in either dry-run or local mode.
    
    Dry-run mode (request.dry_run=True):
    - Validates dataset existence
    - Generates Tunix manifest YAML
    - Returns immediately with status='completed'
    
    Local mode (request.dry_run=False):
    - Requires Tunix CLI installation
    - Exports dataset to temporary JSONL
    - Writes manifest to temporary file
    - Executes `tunix train --config <manifest_path>` via subprocess
    - Captures stdout/stderr and exit code
    - Returns structured execution metadata
    
    Args:
        request: TunixRunRequest with dataset_key, model_id, hyperparameters
        db: AsyncSession for dataset validation
    
    Returns:
        TunixRunResponse with run metadata (no persistence)
    
    Raises:
        ValueError: If dataset not found or invalid
        subprocess.TimeoutExpired: If execution exceeds timeout (30s)
    """
```

### Availability Pattern

Following the UNGAR pattern exactly:

```python
# tunix_rt_backend/integrations/tunix/availability.py

def tunix_available() -> bool:
    """Check if Tunix Python package is importable."""
    try:
        import tunix  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False

def check_tunix_cli() -> dict[str, Any]:
    """Check if `tunix` CLI is accessible and return version info."""
    if not tunix_available():
        return {"accessible": False, "version": None, "error": "Tunix Python package not installed."}
    try:
        result = subprocess.run(
            ["tunix", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        version_line = result.stdout.strip().split('\n')[0]
        return {"accessible": True, "version": version_line, "error": None}
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"accessible": False, "version": None, "error": f"Tunix CLI not accessible: {e}"}
```

### API Endpoint

```python
# tunix_rt_backend/app.py

@app.post("/api/tunix/run", response_model=TunixRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def tunix_run(
    request: TunixRunRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TunixRunResponse:
    """
    Execute a Tunix training run.
    
    Dry-run mode (dry_run=true, default): Validates configuration without executing.
    Local mode (dry_run=false): Executes training locally (requires Tunix installation).
    
    Returns 501 if Tunix not available and dry_run=false.
    """
    if not request.dry_run and not tunix_available():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Tunix runtime not available. Install with `pip install -e '.[tunix]'` to enable local execution.",
        )
    return await execute_tunix_run(request, db)
```

---

## Execution Modes

### 1. Dry-Run Mode (Default)

**Purpose:** Validate run configuration without invoking Tunix CLI.

**Behavior:**
- Validates `dataset_key` exists
- Generates Tunix manifest YAML
- Returns immediately with `status='completed'`, `exit_code=0`

**Use Cases:**
- Pre-flight validation in CI
- Testing dataset/model configuration
- Development without Tunix installed

**Example:**

```bash
curl -X POST http://localhost:8000/api/tunix/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "test_tunix-v1",
    "model_id": "google/gemma-2b-it",
    "dry_run": true
  }'
```

**Response:**

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "mode": "dry-run",
  "dataset_key": "test_tunix-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./output/tunix_run_550e8400",
  "exit_code": 0,
  "stdout": "Dry-run validation passed",
  "stderr": "",
  "duration_seconds": 0.05,
  "started_at": "2025-12-22T10:00:00.000Z",
  "completed_at": "2025-12-22T10:00:00.050Z",
  "message": "Dry-run completed successfully. Configuration is valid."
}
```

### 2. Local Execution Mode

**Purpose:** Execute actual Tunix training on local machine (CPU/GPU).

**Behavior:**
- Requires Tunix CLI installed (`pip install -e '.[tunix]'`)
- Exports dataset to temporary JSONL file
- Writes manifest to temporary YAML file
- Executes `tunix train --config <manifest>`
- Captures stdout/stderr (truncated to 10,000 chars)
- Returns execution metadata with exit code

**Timeout:** 30 seconds (configurable in service layer)

**Use Cases:**
- Local development and debugging
- Smoke testing Tunix integration
- Generating training artifacts for downstream evaluation

**Example:**

```bash
curl -X POST http://localhost:8000/api/tunix/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_key": "test_tunix-v1",
    "model_id": "google/gemma-2b-it",
    "dry_run": false,
    "num_epochs": 1,
    "batch_size": 1,
    "max_seq_length": 128
  }'
```

**Response (Success):**

```json
{
  "run_id": "660f9500-f39c-52e5-b827-557766551111",
  "status": "completed",
  "mode": "local",
  "dataset_key": "test_tunix-v1",
  "model_id": "google/gemma-2b-it",
  "output_dir": "./output/tunix_run_660f9500",
  "exit_code": 0,
  "stdout": "Tunix training started\nEpoch 1/1: loss=0.05\nTraining completed",
  "stderr": "",
  "duration_seconds": 15.3,
  "started_at": "2025-12-22T10:05:00.000Z",
  "completed_at": "2025-12-22T10:05:15.300Z",
  "message": "Local execution completed successfully."
}
```

**Response (Tunix Unavailable, 501):**

```json
{
  "detail": "Tunix runtime not available. Install with `pip install -e '.[tunix]'` to enable local execution."
}
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- tunix-rt backend installed (`pip install -e backend`)

### Installing Tunix (Optional)

For **local execution mode only**, install the Tunix extra:

```bash
cd backend
pip install -e ".[tunix]"
```

**Note:** Tunix availability is assumed to be public. For private/unreleased Tunix:
- Use a Git dependency in `pyproject.toml`
- Or install from a local wheel/source directory

### Verifying Installation

```bash
# Check if Tunix is available
curl http://localhost:8000/api/tunix/status

# Expected response (with Tunix):
{
  "available": true,
  "version": "0.1.0",
  "runtime_required": true,
  "message": "Tunix runtime is available for execution."
}

# Expected response (without Tunix):
{
  "available": false,
  "version": null,
  "runtime_required": false,
  "message": "Tunix runtime not available. Install with `pip install -e '.[tunix]'` for local execution."
}
```

---

## API Reference

### `POST /api/tunix/run`

Execute a Tunix training run.

#### Request Body: `TunixRunRequest`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dataset_key` | string | ✅ | - | Dataset identifier (e.g., `test_tunix-v1`) |
| `model_id` | string | ✅ | - | Model identifier (e.g., `google/gemma-2b-it`) |
| `output_dir` | string | ❌ | Auto-generated | Output directory for artifacts |
| `dry_run` | boolean | ❌ | `true` | If true, validate only; if false, execute training |
| `learning_rate` | float | ❌ | `2e-5` | Learning rate (range: 0 < lr ≤ 1.0) |
| `num_epochs` | int | ❌ | `3` | Number of epochs (range: 1-100) |
| `batch_size` | int | ❌ | `8` | Batch size (range: 1-512) |
| `max_seq_length` | int | ❌ | `2048` | Maximum sequence length (range: 128-32768) |

#### Response: `TunixRunResponse` (202 Accepted)

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique UUID for this run |
| `status` | enum | `"pending" | "completed" | "failed" | "timeout"` |
| `mode` | enum | `"dry-run" | "local"` |
| `dataset_key` | string | Dataset key used |
| `model_id` | string | Model ID used |
| `output_dir` | string | Output directory path |
| `exit_code` | int \\| null | Process exit code (null if not executed) |
| `stdout` | string | Standard output (truncated to 10,000 chars) |
| `stderr` | string | Standard error (truncated to 10,000 chars) |
| `duration_seconds` | float | Execution duration |
| `started_at` | string | ISO-8601 timestamp |
| `completed_at` | string | ISO-8601 timestamp |
| `message` | string | Human-readable status message |

#### Error Responses

| Status Code | Condition | Response |
|-------------|-----------|----------|
| `400 Bad Request` | Invalid request parameters | Pydantic validation error |
| `404 Not Found` | Dataset not found | `{"detail": "Dataset not found: <dataset_key>"}` |
| `501 Not Implemented` | Tunix not available and `dry_run=false` | `{"detail": "Tunix runtime not available..."}` |
| `500 Internal Server Error` | Unexpected execution failure | Generic error message |

---

## Frontend Integration

The frontend (`frontend/src/App.tsx`) provides minimal UI for Tunix execution:

### UI Components

1. **Run Buttons:**
   - "Run (Dry-run)" - Always enabled when dataset_key is provided
   - "Run with Tunix (Local)" - Disabled if dataset_key empty or run in progress

2. **Loading State:**
   - Displays "Executing Tunix run..." spinner during execution
   - Disables all buttons during execution

3. **Result Display:**
   - Run status badge (`completed`, `failed`, `timeout`)
   - Mode indicator (`dry-run`, `local`)
   - Exit code
   - Duration
   - Message
   - Collapsible stdout/stderr sections

### Example Usage

```typescript
import { executeTunixRun, type TunixRunRequest, type TunixRunResponse } from './api/client'

const handleRun = async (dryRun: boolean) => {
  const request: TunixRunRequest = {
    dataset_key: 'test_tunix-v1',
    model_id: 'google/gemma-2b-it',
    dry_run: dryRun,
    num_epochs: 1,
    batch_size: 1,
    max_seq_length: 128,
  }
  
  try {
    const result: TunixRunResponse = await executeTunixRun(request)
    console.log('Run completed:', result.status)
    console.log('Output:', result.stdout)
  } catch (error) {
    if (error.status === 501) {
      console.error('Tunix not available - install backend[tunix]')
    }
  }
}
```

---

## Testing Strategy

### Test Categories

1. **Default Tests (No Tunix Required)**
   - Dry-run validation
   - 501 error responses
   - Schema validation
   - Dataset validation

2. **Optional Tests (`@pytest.mark.tunix`)**
   - Local execution smoke tests
   - CLI availability checks
   - Subprocess output capture

### Running Tests

```bash
# Default tests (CI-safe, no Tunix needed)
cd backend
pytest tests/test_tunix_execution.py -m "not tunix"

# Optional Tunix runtime tests (requires backend[tunix])
pip install -e ".[tunix]"
pytest tests/test_tunix_execution.py -m tunix
```

### Test Coverage

- **Target:** +10-15 tests for M13
- **Coverage Goal:** Coverage-neutral or positive
- **M13 Test File:** `backend/tests/test_tunix_execution.py`

### Example Test

```python
@pytest.mark.asyncio
async def test_dry_run_with_valid_dataset(db_session):
    """Dry-run mode should validate and return success without invoking Tunix."""
    # Setup: Create dataset manifest
    dataset_key = "test_valid-v1"
    dataset_dir = get_dataset_dir(dataset_key)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = DatasetManifest(
        name="test_valid",
        version="v1",
        description="Valid test dataset",
        trace_ids=["trace-1", "trace-2"],
        size=2,
        created_at=datetime.now(timezone.utc).isoformat(),
        format="tunix_sft",
    )
    save_manifest(dataset_key, manifest)
    
    # Execute
    request = TunixRunRequest(
        dataset_key=dataset_key,
        model_id="google/gemma-2b-it",
        dry_run=True,
    )
    result = await execute_tunix_run(request, db_session)
    
    # Assert
    assert result.status == "completed"
    assert result.mode == "dry-run"
    assert result.exit_code == 0
    assert "Dry-run completed successfully" in result.message
```

---

## CI/CD Integration

### Default CI (`.github/workflows/ci.yml`)

- **No Tunix installed** ✅
- Runs all default tests: `pytest -m "not tunix and not ungar"`
- **Coverage gates enforced:** Line ≥80%, Branch ≥68%
- **Status:** Always required for merge

### Tunix Runtime CI (`.github/workflows/tunix-runtime.yml`)

- **Trigger:** Manual workflow dispatch only
- **Purpose:** Validate dry-run path in isolated environment
- **No Tunix installed** in CI (dry-run tests only)
- **Status:** Never blocks merge (`continue-on-error: true`)

**Key Workflow Steps:**

```yaml
- name: Run Tunix execution tests (dry-run mode, no Tunix required)
  working-directory: backend
  run: |
    pytest -v tests/test_tunix_execution.py -k "not tunix"

- name: Verify 501 responses without Tunix installed
  working-directory: backend
  run: |
    pytest -v tests/test_tunix_execution.py::test_local_execution_without_tunix_returns_501
```

---

## Troubleshooting

### Issue: 501 Error "Tunix runtime not available"

**Cause:** Tunix not installed, but `dry_run=false`

**Solution:**

```bash
cd backend
pip install -e ".[tunix]"
```

### Issue: "Dataset not found: <dataset_key>"

**Cause:** Dataset manifest doesn't exist in `backend/datasets/<dataset_key>/`

**Solution:**

```bash
# Create dataset via API or manually
curl -X POST http://localhost:8000/api/datasets \
  -H "Content-Type: application/json" \
  -d '{"name": "my_dataset", "version": "v1", "trace_ids": ["trace-1"]}'
```

### Issue: Timeout after 30 seconds

**Cause:** Training taking longer than timeout limit

**Solution:**
- Use smaller dataset (1-2 traces)
- Reduce `num_epochs`, `batch_size`, or `max_seq_length`
- Increase timeout in `tunix_execution.py` (for development only)

### Issue: Stdout/Stderr truncated

**Cause:** Output exceeds 10,000 character limit

**Solution:**
- This is expected behavior for M13 (no persistence)
- For full logs, run Tunix locally outside API: `tunix train --config manifest.yaml`

---

## Future Enhancements

### M14: Result Ingestion & Persistence

- Store `TunixRunResponse` in database
- Parse training metrics from stdout
- Associate trained models with datasets
- Historical run tracking and comparison

### M15: Asynchronous Execution

- Background workers (Celery/Ray)
- Long-running training support
- Progress updates via WebSocket
- Cancellation and retry logic

### M16: TPU Integration

- TPU pod orchestration
- Multi-host distributed training
- Ray clusters for hyperparameter tuning

### M17: Evaluation Pipeline

- Automated evaluation runs post-training
- Metric computation (accuracy, F1, BLEU, etc.)
- Leaderboard and model ranking

---

## Appendix: File Locations

| Component | File Path |
|-----------|-----------|
| Execution Service | `backend/tunix_rt_backend/services/tunix_execution.py` |
| Availability Check | `backend/tunix_rt_backend/integrations/tunix/availability.py` |
| Schemas | `backend/tunix_rt_backend/schemas/tunix.py` |
| API Endpoint | `backend/tunix_rt_backend/app.py` (lines ~320-335) |
| Tests | `backend/tests/test_tunix_execution.py` |
| Frontend UI | `frontend/src/App.tsx` (Tunix panel) |
| CI Workflow | `.github/workflows/tunix-runtime.yml` |

---

**Document Version:** 1.0  
**Last Updated:** December 22, 2025  
**Maintained By:** tunix-rt M13 Team
