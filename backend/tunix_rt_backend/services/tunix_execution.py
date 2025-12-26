"""Tunix execution service (M13/M14/M16).

This module provides business logic for executing Tunix training runs:
- Dry-run mode: Validate manifest + dataset without executing
- Local mode: Execute tunix CLI via subprocess with timeout
- M16: Real-time log streaming and persistence
- M16: Run cancellation and artifact management

M14 Enhancement: Persist all runs to database for audit trail and history.

The service handles optional Tunix dependency gracefully via lazy imports.
"""
# mypy: disable-error-code="unused-ignore"

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import async_session_maker
from tunix_rt_backend.db.models import TunixRun, TunixRunLogChunk
from tunix_rt_backend.helpers.datasets import load_manifest
from tunix_rt_backend.integrations.tunix.manifest import build_sft_manifest

# M15: Observability
from tunix_rt_backend.metrics import TUNIX_RUN_DURATION_SECONDS, TUNIX_RUNS_TOTAL
from tunix_rt_backend.schemas import TunixRunRequest, TunixRunResponse
from tunix_rt_backend.services.tunix_export import export_tunix_sft_jsonl
from tunix_rt_backend.settings import settings

logger = logging.getLogger(__name__)


class LogManager:
    """Manages real-time log streaming and persistence (M16).

    Handles:
    - Buffering log lines from stdout/stderr
    - assigning monotonic sequence numbers
    - Batch writing to DB to avoid thrash
    - Truncating total output for summary
    """

    def __init__(self, db: AsyncSession, run_id: uuid.UUID):
        self.db = db
        self.run_id = run_id
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.seq = 1
        self.stdout_buffer: list[str] = []
        self.stderr_buffer: list[str] = []
        self.total_stdout_bytes = 0
        self.total_stderr_bytes = 0
        self.max_summary_bytes = settings.tunix_output_max_bytes

    async def stream_log(self, stream: asyncio.StreamReader, stream_name: str) -> None:
        """Read from stream line-by-line and enqueue."""
        while True:
            line_bytes = await stream.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8", errors="replace")
            await self.queue.put((stream_name, line))

    async def writer_loop(self) -> None:
        """Consumer loop that writes logs to DB in batches."""
        buffer: list[TunixRunLogChunk] = []
        last_flush = datetime.now()

        while True:
            try:
                # Wait for next item, with timeout for flushing
                try:
                    stream_name, text = await asyncio.wait_for(self.queue.get(), timeout=0.5)

                    # Update summary buffers (truncated)
                    if stream_name == "stdout":
                        if self.total_stdout_bytes < self.max_summary_bytes:
                            self.stdout_buffer.append(text)
                            self.total_stdout_bytes += len(text.encode("utf-8"))
                    else:
                        if self.total_stderr_bytes < self.max_summary_bytes:
                            self.stderr_buffer.append(text)
                            self.total_stderr_bytes += len(text.encode("utf-8"))

                    # Create chunk model
                    chunk = TunixRunLogChunk(
                        run_id=self.run_id, seq=self.seq, stream=stream_name, chunk=text
                    )
                    self.seq += 1
                    buffer.append(chunk)
                    self.queue.task_done()

                except asyncio.TimeoutError:
                    pass  # Just flush if buffer has items

                # Flush if buffer full or enough time passed
                now = datetime.now()
                if buffer and (len(buffer) >= 50 or (now - last_flush).total_seconds() > 1.0):
                    self.db.add_all(buffer)
                    await self.db.commit()
                    buffer = []
                    last_flush = now

            except asyncio.CancelledError:
                # Flush remaining items on cancel
                if buffer:
                    self.db.add_all(buffer)
                    await self.db.commit()
                raise

    def get_summary(self) -> tuple[str, str]:
        """Return truncated stdout/stderr summaries."""
        stdout = "".join(self.stdout_buffer)
        if self.total_stdout_bytes >= self.max_summary_bytes:
            stdout += f"\n\n[... output truncated to {self.max_summary_bytes} bytes ...]"

        stderr = "".join(self.stderr_buffer)
        if self.total_stderr_bytes >= self.max_summary_bytes:
            stderr += f"\n\n[... output truncated to {self.max_summary_bytes} bytes ...]"

        return stdout, stderr


async def execute_tunix_run(
    request: TunixRunRequest,
    db: AsyncSession,
    async_mode: bool = False,
) -> TunixRunResponse:
    """Execute a Tunix training run (dry-run or local mode) with persistence (M14).

    Args:
        request: Run configuration (dataset_key, model_id, hyperparameters, dry_run flag)
        db: Database session
        async_mode: If True, enqueue job and return immediately (M15)

    Returns:
        TunixRunResponse with execution results

    Raises:
        HTTPException: If Tunix not available and dry_run=False (handled by caller)

    Note:
        - Dry-run mode: Validates manifest + dataset, does NOT execute
        - Local mode: Executes tunix CLI via subprocess with 30s timeout
        - M14: Persists all runs to database (status=running → completed/failed/timeout)
        - M14: DB failure does NOT fail the user request (execution success > persistence)
        - M15: Supports async execution (returns pending status)
        - M16: Supports real-time log streaming
    """
    run_id_uuid = uuid.uuid4()
    run_id = str(run_id_uuid)
    started_at_dt = datetime.now(timezone.utc)

    # Determine execution mode (dry-run vs local)
    execution_mode: Literal["dry-run", "local"] = "dry-run" if request.dry_run else "local"

    # Generate output_dir if not provided
    output_dir = request.output_dir or f"./output/tunix_run_{run_id[:8]}"

    # Determine initial status
    initial_status = "pending" if async_mode else "running"

    # Create run record
    db_run = await create_tunix_run_record(
        db, run_id_uuid, request, execution_mode, initial_status, started_at_dt
    )

    # If async, return immediately
    if async_mode:
        TUNIX_RUNS_TOTAL.labels(status="pending", mode=execution_mode).inc()
        return TunixRunResponse(
            run_id=run_id,
            status="pending",
            mode=execution_mode,
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=None,
            started_at=started_at_dt.isoformat(),
            completed_at=None,
            message=f"{execution_mode.capitalize()} execution pending",
        )

    # If sync, execute immediately
    response = await process_tunix_run(
        run_id=run_id,
        request=request,
        output_dir=output_dir,
        started_at=started_at_dt.isoformat(),
        db=db,
    )

    # Update run record with final results
    await update_tunix_run_record(db, db_run, response)

    # M15: Record metrics
    TUNIX_RUNS_TOTAL.labels(status=response.status, mode=execution_mode).inc()
    if response.duration_seconds is not None:
        TUNIX_RUN_DURATION_SECONDS.labels(mode=execution_mode, status=response.status).observe(
            response.duration_seconds
        )

    # M17: Auto-evaluate if completed (skip dry-run)
    if response.status == "completed" and execution_mode != "dry-run":
        try:
            from tunix_rt_backend.services.evaluation import EvaluationService

            await EvaluationService(db).evaluate_run(uuid.UUID(run_id))
        except Exception as e:
            logger.error(f"Auto-evaluation failed for {run_id}: {e}")

    return response


async def create_tunix_run_record(
    db: AsyncSession,
    run_id_uuid: uuid.UUID,
    request: TunixRunRequest,
    mode: str,
    status: str,
    started_at: datetime,
) -> TunixRun:
    """Create initial Tunix run record in database."""
    db_run = TunixRun(
        run_id=run_id_uuid,
        dataset_key=request.dataset_key,
        model_id=request.model_id,
        mode=mode,
        status=status,
        exit_code=None,
        started_at=started_at,
        completed_at=None,
        duration_seconds=None,
        stdout="",
        stderr="",
        config=request.model_dump(),
    )

    try:
        db.add(db_run)
        await db.commit()
        await db.refresh(db_run)
    except Exception as e:
        # M14 Decision: DB failure does not fail user request
        logger.error(f"Failed to create run record {run_id_uuid}: {e}")
        # We return the object even if not persisted correctly, assuming caller handles downstream
        pass

    return db_run


async def update_tunix_run_record(
    db: AsyncSession,
    db_run: TunixRun,
    response: TunixRunResponse,
) -> None:
    """Update Tunix run record with final results."""
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
        logger.error(f"Failed to update run record {db_run.run_id}: {e}")


async def process_tunix_run(
    run_id: str,
    request: TunixRunRequest,
    output_dir: str,
    started_at: str,
    db: AsyncSession,
) -> TunixRunResponse:
    """Process a Tunix run (dry-run or local).

    This function handles the actual execution logic.
    """
    if request.dry_run:
        return await execute_dry_run(
            run_id=run_id,
            request=request,
            output_dir=output_dir,
            started_at=started_at,
            db=db,
        )
    else:
        return await execute_local(
            run_id=run_id,
            request=request,
            output_dir=output_dir,
            started_at=started_at,
            db=db,
        )


async def execute_dry_run(
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
            device=request.device,
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
        from tunix_rt_backend.schemas.tunix import TunixExportRequest

        # Use real Pydantic model for export request
        export_request = TunixExportRequest(
            dataset_key=request.dataset_key, trace_ids=None, limit=5
        )

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


async def execute_local(  # pragma: no cover
    run_id: str,
    request: TunixRunRequest,
    output_dir: str,
    started_at: str,
    db: AsyncSession,
) -> TunixRunResponse:
    """Execute local mode (run training script via subprocess).

    Args:
        run_id: Unique run identifier
        request: Run configuration
        output_dir: Output directory path
        started_at: Start timestamp (ISO-8601)
        db: Database session

    Returns:
        TunixRunResponse with execution results

    Note: This function spawns the training script as a subprocess.
    It is excluded from coverage in base CI where training deps are not installed.
    """
    # M24 Update: We now execute the internal training script (training/train_sft_tunix.py)
    # instead of expecting a 'tunix' CLI to be on the PATH.
    # This allows fallback to PyTorch/Transformers if JAX/Tunix is missing.

    # Step 1: Prepare temporary files (manifest + dataset)
    try:
        from tunix_rt_backend.schemas import TunixManifestRequest
        from tunix_rt_backend.schemas.tunix import TunixExportRequest

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
            device=request.device,
        )
        dataset_path = f"./datasets/{request.dataset_key}.jsonl"
        manifest_yaml = build_sft_manifest(manifest_request, dataset_path)
        manifest_path = temp_dir / "manifest.yaml"
        manifest_path.write_text(manifest_yaml, encoding="utf-8")

        # Export dataset JSONL
        export_request = TunixExportRequest(
            dataset_key=request.dataset_key,
            trace_ids=None,
            limit=request.num_epochs * request.batch_size * 10,
        )

        jsonl_lines = await export_tunix_sft_jsonl(export_request, db)

        dataset_path_str = str(temp_dir / f"{request.dataset_key}.jsonl")
        with open(dataset_path_str, "w", encoding="utf-8") as f:
            if isinstance(jsonl_lines, str):
                f.write(jsonl_lines)
            else:
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

    # Step 2: Execute training script with log streaming (M16)
    log_manager = LogManager(db, uuid.UUID(run_id))

    try:
        # Determine script path (assume relative to backend root or repo root)
        # We are running from 'backend' directory in CI/Dev usually.
        # But uvicorn might be running from repo root or backend.
        # Let's find the script.
        script_path = Path("training/train_sft_tunix.py")
        if not script_path.exists():
            # Try ../training/train_sft_tunix.py if in backend
            script_path = Path("../training/train_sft_tunix.py")

        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found at {script_path}")

        # Use asyncio.create_subprocess_exec for async stream reading
        # Command: python script.py --config ... --data ... --output ...
        process = await asyncio.create_subprocess_exec(
            "python",
            str(script_path),
            "--config",
            str(manifest_path.absolute()),
            "--data",
            str(Path(dataset_path_str).absolute()),
            "--output",
            str(Path(output_dir).absolute()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Run in current directory (backend or repo root) so script can find its imports
            # if needed, or use temp_dir? The script is standalone-ish but imports yaml.
            # Using repo root/backend root is safer for python path.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        # Launch writer task
        writer_task = asyncio.create_task(log_manager.writer_loop())

        # M16: Monitor cancellation task
        async def cancel_monitor() -> None:
            while process.returncode is None:
                try:
                    # Use a fresh session for checking cancellation
                    # to avoid interfering with the main session
                    # (although here main session is unused during wait)
                    # But LogManager uses 'db'. If we use async_session_maker, it's safer.
                    async with async_session_maker() as session:
                        run = await session.get(TunixRun, uuid.UUID(run_id))
                        if run and run.status == "cancel_requested":
                            logger.info(f"Cancellation requested for run {run_id}")
                            process.terminate()
                            # Allow brief grace period then kill?
                            # For now just terminate and break loop
                            return
                except Exception as e:
                    logger.warning(f"Error checking cancellation status: {e}")

                await asyncio.sleep(2)

        monitor_task = asyncio.create_task(cancel_monitor())

        try:
            # Wait for process and streams to complete
            # Ensure stdout/stderr are not None (mypy check)
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Process stdout/stderr not available")

            await asyncio.wait_for(
                asyncio.gather(
                    process.wait(),
                    log_manager.stream_log(process.stdout, "stdout"),
                    log_manager.stream_log(process.stderr, "stderr"),
                ),
                timeout=30,  # 30 second timeout per M13 requirements
            )
        except asyncio.TimeoutError:
            raise  # Handle in outer except
        except asyncio.CancelledError:
            # If execute_local is cancelled (e.g. app shutdown), terminate process
            process.terminate()
            raise
        finally:
            # Clean up tasks
            monitor_task.cancel()
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        completed_at = datetime.now(timezone.utc).isoformat()
        duration = (
            datetime.fromisoformat(completed_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp()
        )

        stdout_summary, stderr_summary = log_manager.get_summary()

        # Determine status
        # If we terminated due to cancellation, returncode might be non-zero (e.g. SIGTERM)
        # We should check if cancellation was requested to set status='cancelled'
        final_status: Literal["completed", "failed", "cancelled"]
        message: str

        # Check if it was cancelled
        async with async_session_maker() as session:
            run = await session.get(TunixRun, uuid.UUID(run_id))
            was_cancelled = run and run.status == "cancel_requested"

        if was_cancelled:
            final_status = "cancelled"
            message = "Run cancelled by user"
        elif process.returncode == 0:
            final_status = "completed"
            message = "Local execution completed successfully"

            # M23: Generate predictions artifact (inference step)
            try:
                await generate_predictions(
                    manifest_path.parent / f"{request.dataset_key}.jsonl", Path(output_dir)
                )
            except Exception as e:
                logger.error(f"Failed to generate predictions: {e}")
                # Don't fail the run, but log error. Judge will catch missing file.
                message += f" (Warning: Inference failed: {e})"

        else:
            final_status = "failed"
            message = f"Local execution failed with exit code {process.returncode}"

        return TunixRunResponse(
            run_id=run_id,
            status=final_status,  # type: ignore[arg-type]
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=process.returncode,
            stdout=stdout_summary,
            stderr=stderr_summary,
            duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            message=message,
        )

    except asyncio.TimeoutError:
        # Kill process on timeout
        try:
            process.terminate()
            await asyncio.sleep(0.1)
            if process.returncode is None:
                process.kill()
        except Exception:
            pass

        completed_at = datetime.now(timezone.utc).isoformat()
        duration = (
            datetime.fromisoformat(completed_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp()
        )

        stdout_summary, stderr_summary = log_manager.get_summary()

        return TunixRunResponse(
            run_id=run_id,
            status="timeout",
            mode="local",
            dataset_key=request.dataset_key,
            model_id=request.model_id,
            output_dir=output_dir,
            exit_code=None,
            stdout=stdout_summary,
            stderr=stderr_summary + "\n\nExecution timeout (exceeded 30 seconds)",
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


async def cancel_tunix_run(run_id: uuid.UUID, db: AsyncSession) -> None:
    """Cancel a Tunix run (M16).

    Args:
        run_id: UUID of the run
        db: Database session

    Raises:
        ValueError: If run not found or cannot be cancelled
    """
    run = await db.get(TunixRun, run_id)
    if not run:
        raise ValueError("Run not found")

    if run.status in ["completed", "failed", "timeout", "cancelled"]:
        raise ValueError(f"Run is already in terminal state: {run.status}")

    if run.status == "pending":
        run.status = "cancelled"
        run.completed_at = datetime.now(timezone.utc)
        run.stderr += "\nRun cancelled before starting."
    elif run.status == "running":
        run.status = "cancel_requested"
        # Worker will pick this up and terminate process

    await db.commit()


def truncate_output(output: str, max_bytes: int | None = None) -> str:
    """Truncate output string to maximum byte length.

    Args:
        output: Output string to truncate
        max_bytes: Maximum byte length (default: use setting)

    Returns:
        Truncated string with notice if truncation occurred
    """
    if max_bytes is None:
        max_bytes = settings.tunix_output_max_bytes

    if not output:
        return ""

    encoded = output.encode("utf-8")
    if len(encoded) <= max_bytes:
        return output

    # Truncate and add notice
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}\n\n[... output truncated to {max_bytes} bytes ...]"


async def generate_predictions(dataset_path: Path, output_dir: Path) -> None:
    """Generate predictions.jsonl from dataset (M23/M24).

    M24: Replaced placeholder with real inference using distilgpt2.
    M27: Use trained model if available.
    """
    logger.info(f"Generating predictions from {dataset_path} to {output_dir}")

    # Ensure dataset exists
    if not dataset_path.exists():
        logger.warning(f"Dataset not found for inference: {dataset_path}")
        return

    # Determine model to use
    model_name = "distilgpt2"
    final_model_dir = output_dir / "final_model"
    if final_model_dir.exists():
        logger.info(f"Found trained model at {final_model_dir}")
        model_name = str(final_model_dir)

    # M24: Run inference in a separate thread to avoid blocking the event loop
    try:
        await asyncio.to_thread(_run_inference_sync, dataset_path, output_dir, model_name)
    except Exception as e:
        logger.error(f"Error during prediction generation: {e}")
        # We don't raise here to avoid crashing the whole run flow,
        # but the Judge will fail if predictions.jsonl is missing/empty.
        raise


def _run_inference_sync(  # pragma: no cover
    dataset_path: Path, output_dir: Path, model_name: str = "distilgpt2"
) -> None:
    """Synchronous inference logic (M24).

    Note: This function requires optional ML dependencies (torch, transformers).
    It is excluded from coverage in base CI where those deps are not installed.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError:
        logger.warning("Transformers/Torch not installed. Skipping real inference.")
        # Fallback to placeholder if dependencies are missing (e.g. in minimal env)
        _generate_placeholder_predictions(dataset_path, output_dir)
        return

    logger.info(f"Loading model {model_name} for inference...")
    try:
        # transformers library is untyped
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Try loading (handling Flax weights if needed)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        except OSError:
            logger.info("Standard load failed, trying from_flax=True...")
            model = AutoModelForCausalLM.from_pretrained(model_name, from_flax=True)  # type: ignore[no-untyped-call]

        # Use CPU for deterministic CI smoke or if no GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # type: ignore[arg-type]
        model.eval()  # type: ignore[no-untyped-call]
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        # Fallback to base model if loading trained model fails?
        # Only if we tried to load a path. If distilgpt2 failed, we are toast.
        if Path(model_name).exists():
            logger.warning("Falling back to distilgpt2")
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # type: ignore[no-untyped-call]
                model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # type: ignore[no-untyped-call]
                model.to(device)  # type: ignore[arg-type]
                model.eval()  # type: ignore[no-untyped-call]
            except Exception:
                raise e
        else:
            raise

    predictions = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                trace_id = item.get("id")
                # M24: Support both 'prompts' (Tunix) and 'prompt' (Legacy) keys
                prompt_text = item.get("prompts") or item.get("prompt") or ""

                if not trace_id or not prompt_text.strip():
                    continue

                # M24: Simple logic to extract user prompt if possible
                # Default Tunix format: <start_of_turn>user\n...\n<start_of_turn>model
                input_text = prompt_text
                if "<start_of_turn>model" in prompt_text:
                    parts = prompt_text.split("<start_of_turn>model")
                    input_text = parts[0] + "<start_of_turn>model"

                # Tokenize
                inputs = tokenizer(input_text, return_tensors="pt").to(device)

                # Generate (Greedy, Deterministic)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode only the new tokens
                new_tokens = outputs[0][inputs.input_ids.shape[1] :]
                prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                predictions.append({"trace_id": trace_id, "prediction": prediction_text})

            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Inference failed for item {trace_id}: {e}")
                continue

    if not predictions:
        logger.error("No predictions generated. Skipping file creation.")
        raise RuntimeError("Inference failed for all items (no predictions generated)")

    output_path = output_dir / "predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # M25: Save inference metadata
    device_str = "cpu"
    # Try to capture device string if available in local scope (from try block above)
    # Using a safer check that doesn't trigger type errors on 'device' variable
    try:
        import torch

        device_type = torch.device  # Cache the type for isinstance check
        if "device" in locals() and isinstance(locals()["device"], device_type):
            device_str = str(locals()["device"])
    except (ImportError, NameError, TypeError):
        pass

    meta = {
        "model_id": model_name,
        "dataset_path": str(dataset_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 50,
            "device": device_str,
        },
    }
    with open(output_dir / "predictions_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Generated {len(predictions)} predictions to {output_path}")


def _generate_placeholder_predictions(dataset_path: Path, output_dir: Path) -> None:
    """Fallback placeholder generation (M23 logic)."""
    predictions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                trace_id = item.get("id")
                if trace_id:
                    predictions.append(
                        {
                            "trace_id": trace_id,
                            "prediction": "Model prediction placeholder (transformers missing)",
                        }
                    )
            except Exception:
                continue

    if not predictions:
        # If no items found, don't write empty file
        return

    output_path = output_dir / "predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
