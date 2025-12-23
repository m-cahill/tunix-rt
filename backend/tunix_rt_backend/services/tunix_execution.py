"""Tunix execution service (M13/M14).

This module provides business logic for executing Tunix training runs:
- Dry-run mode: Validate manifest + dataset without executing
- Local mode: Execute tunix CLI via subprocess with timeout

M14 Enhancement: Persist all runs to database for audit trail and history.

The service handles optional Tunix dependency gracefully via lazy imports.
"""

import json
import logging
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import TunixRun
from tunix_rt_backend.helpers.datasets import load_manifest
from tunix_rt_backend.integrations.tunix.availability import check_tunix_cli
from tunix_rt_backend.integrations.tunix.manifest import build_sft_manifest
from tunix_rt_backend.schemas import TunixRunRequest, TunixRunResponse
from tunix_rt_backend.services.tunix_export import export_tunix_sft_jsonl

logger = logging.getLogger(__name__)


async def execute_tunix_run(
    request: TunixRunRequest,
    db: AsyncSession,
) -> TunixRunResponse:
    """Execute a Tunix training run (dry-run or local mode) with persistence (M14).

    Args:
        request: Run configuration (dataset_key, model_id, hyperparameters, dry_run flag)
        db: Database session

    Returns:
        TunixRunResponse with execution results

    Raises:
        HTTPException: If Tunix not available and dry_run=False (handled by caller)

    Note:
        - Dry-run mode: Validates manifest + dataset, does NOT execute
        - Local mode: Executes tunix CLI via subprocess with 30s timeout
        - M14: Persists all runs to database (status=running → completed/failed/timeout)
        - M14: DB failure does NOT fail the user request (execution success > persistence)
    """
    run_id_uuid = uuid.uuid4()
    run_id = str(run_id_uuid)
    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    mode = "dry-run" if request.dry_run else "local"

    # Generate output_dir if not provided
    output_dir = request.output_dir or f"./output/tunix_run_{run_id[:8]}"

    # M14: Create run record immediately with status='running'
    db_run = TunixRun(
        run_id=run_id_uuid,
        dataset_key=request.dataset_key,
        model_id=request.model_id,
        mode=mode,
        status="running",
        exit_code=None,
        started_at=started_at_dt,
        completed_at=None,
        duration_seconds=None,
        stdout="",
        stderr="",
    )

    try:
        db.add(db_run)
        await db.commit()
        await db.refresh(db_run)
    except Exception as e:
        # M14 Decision: DB failure does not fail user request
        logger.error(f"Failed to create run record {run_id}: {e}")
        # Continue with execution despite persistence failure

    # Execute run (dry-run or local)
    if request.dry_run:
        response = await _execute_dry_run(
            run_id=run_id,
            request=request,
            output_dir=output_dir,
            started_at=started_at,
            db=db,
        )
    else:
        response = await _execute_local(
            run_id=run_id,
            request=request,
            output_dir=output_dir,
            started_at=started_at,
            db=db,
        )

    # M14: Update run record with final results
    try:
        db_run.status = response.status
        db_run.exit_code = response.exit_code
        db_run.completed_at = (
            datetime.fromisoformat(response.completed_at) if response.completed_at else None
        )
        db_run.duration_seconds = response.duration_seconds
        db_run.stdout = response.stdout
        db_run.stderr = response.stderr
        await db.commit()
    except Exception as e:
        # M14 Decision: DB failure does not fail user request
        logger.error(f"Failed to update run record {run_id}: {e}")
        # Return response to user despite persistence failure

    return response


async def _execute_dry_run(
    run_id: str,
    request: TunixRunRequest,
    output_dir: str,
    started_at: str,
    db: AsyncSession,
) -> TunixRunResponse:
    """Execute dry-run mode (validate without executing).

    Args:
        run_id: Unique run identifier
        request: Run configuration
        output_dir: Output directory path
        started_at: Start timestamp (ISO-8601)
        db: Database session

    Returns:
        TunixRunResponse with validation results
    """
    completed_at = datetime.now(timezone.utc).isoformat()
    duration = (
        datetime.fromisoformat(completed_at).timestamp()
        - datetime.fromisoformat(started_at).timestamp()
    )

    # Step 1: Validate dataset exists
    try:
        _manifest = load_manifest(request.dataset_key)
    except FileNotFoundError:
        return TunixRunResponse(
            run_id=run_id,
            status="failed",
            mode="dry-run",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr=f"Dataset not found: {request.dataset_key}",
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message="Dry-run failed: dataset not found",
        )

    # Step 2: Generate manifest (validates schema)
    try:
        from tunix_rt_backend.schemas import TunixManifestRequest

        manifest_request = TunixManifestRequest(
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            max_seq_length=request.max_seq_length,
        )
        dataset_path = f"./datasets/{request.dataset_key}.jsonl"
        manifest_yaml = build_sft_manifest(manifest_request, dataset_path)
    except Exception as e:
        return TunixRunResponse(
            run_id=run_id,
            status="failed",
            mode="dry-run",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr=f"Manifest generation failed: {str(e)}",
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message="Dry-run failed: manifest generation error",
        )

    # Step 3: Validate dataset export (ensures traces exist)
    try:
        # Export first 5 traces as validation (don't need full dataset)
        export_request = type(
            "ExportRequest",
            (),
            {"dataset_key": request.dataset_key, "trace_ids": None, "limit": 5},
        )()
        jsonl_lines = await export_tunix_sft_jsonl(export_request, db)
        trace_count = len(list(jsonl_lines))

        if trace_count == 0:
            return TunixRunResponse(
                run_id=run_id,
                status="failed",
                mode="dry-run",
                dataset_key=request.dataset_key,
                model_id=request.model_id,
                output_dir=output_dir,
                exit_code=None,
                stdout="",
                stderr="Dataset is empty (no traces found)",
                duration_seconds=duration,
                started_at=started_at,
                completed_at=completed_at,
                message="Dry-run failed: empty dataset",
            )
    except Exception as e:
        return TunixRunResponse(
            run_id=run_id,
            status="failed",
            mode="dry-run",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr=f"Dataset export validation failed: {str(e)}",
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message="Dry-run failed: export validation error",
        )

    # Success: all validations passed
    completed_at = datetime.now(timezone.utc).isoformat()
    duration = (
        datetime.fromisoformat(completed_at).timestamp()
        - datetime.fromisoformat(started_at).timestamp()
    )

    stdout_summary = f"""Dry-run validation completed successfully.

Dataset: {request.dataset_key}
Model: {request.model_id}
Output Directory: {output_dir}

Validation Steps:
✅ Dataset manifest found
✅ Manifest YAML generated ({len(manifest_yaml)} bytes)
✅ Dataset export validated ({trace_count} traces sampled)

Training Configuration:
- Learning Rate: {request.learning_rate}
- Epochs: {request.num_epochs}
- Batch Size: {request.batch_size}
- Max Sequence Length: {request.max_seq_length}

To execute this run, set dry_run=false in the request.
"""

    return TunixRunResponse(
        run_id=run_id,
        status="completed",
        mode="dry-run",
        dataset_key=request.dataset_key,
        model_id=request.model_id,
        output_dir=output_dir,
        exit_code=0,
        stdout=stdout_summary,
        stderr="",
        duration_seconds=duration,
        started_at=started_at,
        completed_at=completed_at,
        message="Dry-run validation successful",
    )


async def _execute_local(
    run_id: str,
    request: TunixRunRequest,
    output_dir: str,
    started_at: str,
    db: AsyncSession,
) -> TunixRunResponse:
    """Execute local mode (run tunix CLI via subprocess).

    Args:
        run_id: Unique run identifier
        request: Run configuration
        output_dir: Output directory path
        started_at: Start timestamp (ISO-8601)
        db: Database session

    Returns:
        TunixRunResponse with execution results
    """
    # Check Tunix CLI availability
    cli_status = check_tunix_cli()
    if not cli_status["accessible"]:
        return TunixRunResponse(
            run_id=run_id,
            status="failed",
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr=f"Tunix CLI not accessible: {cli_status['error']}",
            duration_seconds=0.0,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            message="Local execution failed: Tunix CLI not found",
        )

    # Step 1: Prepare temporary files (manifest + dataset)
    try:
        from tunix_rt_backend.schemas import TunixManifestRequest

        temp_dir = Path(tempfile.mkdtemp(prefix=f"tunix_run_{run_id[:8]}_"))

        # Generate manifest YAML
        manifest_request = TunixManifestRequest(
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            max_seq_length=request.max_seq_length,
        )
        dataset_path = f"./datasets/{request.dataset_key}.jsonl"
        manifest_yaml = build_sft_manifest(manifest_request, dataset_path)
        manifest_path = temp_dir / "manifest.yaml"
        manifest_path.write_text(manifest_yaml, encoding="utf-8")

        # Export dataset JSONL
        export_request = type(
            "ExportRequest",
            (),
            {
                "dataset_key": request.dataset_key,
                "trace_ids": None,
                "limit": request.num_epochs * request.batch_size * 10,
            },  # noqa: E501
        )()
        jsonl_lines = await export_tunix_sft_jsonl(export_request, db)

        dataset_path_str = str(temp_dir / f"{request.dataset_key}.jsonl")
        with open(dataset_path_str, "w", encoding="utf-8") as f:
            for line in jsonl_lines:
                f.write(json.dumps(line) + "\n")

    except Exception as e:
        return TunixRunResponse(
            run_id=run_id,
            status="failed",
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr=f"Failed to prepare files: {str(e)}",
            duration_seconds=0.0,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            message="Local execution failed: file preparation error",
        )

    # Step 2: Execute tunix CLI
    try:
        result = subprocess.run(
            ["tunix", "train", "--config", str(manifest_path)],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout per M13 requirements
            cwd=str(temp_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        completed_at = datetime.now(timezone.utc).isoformat()
        duration = (
            datetime.fromisoformat(completed_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp()
        )

        # Truncate stdout/stderr to 10KB (safety limit)
        stdout_truncated = _truncate_output(result.stdout, max_bytes=10240)
        stderr_truncated = _truncate_output(result.stderr, max_bytes=10240)

        status: str = "completed" if result.returncode == 0 else "failed"
        message = (
            "Local execution completed successfully"
            if status == "completed"
            else f"Local execution failed with exit code {result.returncode}"
        )

        return TunixRunResponse(
            run_id=run_id,
            status=status,  # type: ignore[arg-type]
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=result.returncode,
            stdout=stdout_truncated,
            stderr=stderr_truncated,
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message=message,
        )

    except subprocess.TimeoutExpired:
        completed_at = datetime.now(timezone.utc).isoformat()
        duration = (
            datetime.fromisoformat(completed_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp()
        )

        return TunixRunResponse(
            run_id=run_id,
            status="timeout",
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr="Execution timeout (exceeded 30 seconds)",
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message="Local execution timed out after 30 seconds",
        )

    except Exception as e:
        completed_at = datetime.now(timezone.utc).isoformat()
        duration = (
            datetime.fromisoformat(completed_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp()
        )

        return TunixRunResponse(
            run_id=run_id,
            status="failed",
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr=f"Execution error: {str(e)}",
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message="Local execution failed with unexpected error",
        )


def _truncate_output(output: str, max_bytes: int = 10240) -> str:
    """Truncate output string to maximum byte length.

    Args:
        output: Output string to truncate
        max_bytes: Maximum byte length (default: 10KB)

    Returns:
        Truncated string with notice if truncation occurred
    """
    if not output:
        return ""

    encoded = output.encode("utf-8")
    if len(encoded) <= max_bytes:
        return output

    # Truncate and add notice
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}\n\n[... output truncated to {max_bytes} bytes ...]"
