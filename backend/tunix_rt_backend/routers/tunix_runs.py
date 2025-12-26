"""Tunix run management endpoints.

Domain: Training run lifecycle, logs, artifacts, and metrics

Primary endpoints:
- GET /api/tunix/runs: List runs with pagination/filtering
- GET /api/tunix/runs/{id}: Run details (status, config, metrics)
- GET /api/tunix/runs/{id}/status: Polling endpoint for async runs
- GET /api/tunix/runs/{id}/logs: SSE log streaming
- POST /api/tunix/runs/{id}/cancel: Cancel running job
- GET /api/tunix/runs/{id}/artifacts: List/download output artifacts

Cross-cutting concerns:
- Status transitions: pending → running → completed/failed/cancelled
- Log streaming via Server-Sent Events (SSE)
- Stdout/stderr truncation (10KB limit for storage)
- Artifact file download with proper content types
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.db.models import TunixRun, TunixRunLogChunk
from tunix_rt_backend.schemas import (
    TunixRunListItem,
    TunixRunListResponse,
    TunixRunResponse,
    TunixRunStatusResponse,
)
from tunix_rt_backend.services.tunix_execution import cancel_tunix_run

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/tunix/runs/{run_id}/status", response_model=TunixRunStatusResponse)
async def get_tunix_run_status(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TunixRunStatusResponse:
    """Get status of a Tunix training run (M15).

    Args:
        run_id: UUID of the run
        db: Database session

    Returns:
        TunixRunStatusResponse with status and timestamps

    Raises:
        HTTPException: 404 if run not found
    """
    # Fetch run
    result = await db.execute(select(TunixRun).where(TunixRun.run_id == run_id))
    db_run = result.scalar_one_or_none()

    if db_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tunix run not found: {run_id}",
        )

    return TunixRunStatusResponse(
        run_id=str(db_run.run_id),
        status=db_run.status,  # type: ignore[arg-type]
        queued_at=db_run.created_at.isoformat(),
        started_at=db_run.started_at.isoformat(),
        completed_at=db_run.completed_at.isoformat() if db_run.completed_at else None,
        exit_code=db_run.exit_code,
    )


@router.get("/api/tunix/runs", response_model=TunixRunListResponse)
async def list_tunix_runs(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    dataset_key: str | None = None,
    mode: str | None = None,
) -> TunixRunListResponse:
    """List Tunix training runs with pagination and filtering (M14).

    Returns a paginated list of Tunix run history with optional filtering by
    status, dataset_key, and execution mode.

    Args:
        limit: Maximum number of runs to return (1-100, default 20)
        offset: Pagination offset (default 0)
        status_filter: Filter by execution status (optional, query param: status)
        dataset_key: Filter by dataset identifier (optional)
        mode: Filter by execution mode (dry-run or local, optional)
        db: Database session

    Returns:
        TunixRunListResponse with run summaries and pagination info

    Raises:
        HTTPException: 422 if limit > 100 or offset < 0

    Note:
        - All filters use AND logic (status AND dataset_key AND mode)
        - Runs are sorted by created_at DESC (most recent first)
        - Use GET /api/tunix/runs/{run_id} for full run details including stdout/stderr
    """
    # Validate pagination parameters
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be between 1 and 100",
        )

    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="offset must be non-negative",
        )

    # Build query with filters (AND logic)
    query = select(TunixRun)

    if status_filter:
        query = query.where(TunixRun.status == status_filter)
    if dataset_key:
        query = query.where(TunixRun.dataset_key == dataset_key)
    if mode:
        query = query.where(TunixRun.mode == mode)

    # Order by created_at DESC and apply pagination
    query = query.order_by(TunixRun.created_at.desc()).limit(limit + 1).offset(offset)

    # Execute query
    result = await db.execute(query)
    db_runs = result.scalars().all()

    # Check if there are more results
    has_more = len(db_runs) > limit
    runs_to_return = db_runs[:limit]

    # Build response
    data = [
        TunixRunListItem(
            run_id=str(run.run_id),
            dataset_key=run.dataset_key,
            model_id=run.model_id,
            mode=run.mode,  # type: ignore[arg-type]
            status=run.status,  # type: ignore[arg-type]
            started_at=run.started_at.isoformat(),
            duration_seconds=run.duration_seconds,
            metrics=run.metrics,
        )
        for run in runs_to_return
    ]

    next_offset = offset + limit if has_more else None

    return TunixRunListResponse(
        data=data,
        pagination={
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
        },
    )


@router.get("/api/tunix/runs/{run_id}", response_model=TunixRunResponse)
async def get_tunix_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TunixRunResponse:
    """Get full details of a Tunix training run by ID (M14).

    Returns complete run information including stdout, stderr, and all metadata.

    Args:
        run_id: UUID of the run to retrieve
        db: Database session

    Returns:
        TunixRunResponse with full run details

    Raises:
        HTTPException: 404 if run not found

    Note:
        - Returns the same schema as POST /api/tunix/run for consistency
        - Includes full stdout/stderr (truncated to 10KB at capture time)
        - message field is reconstructed based on final status
    """
    # Fetch run from database
    result = await db.execute(select(TunixRun).where(TunixRun.run_id == run_id))
    db_run = result.scalar_one_or_none()

    if db_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tunix run not found: {run_id}",
        )

    # Reconstruct message based on status
    if db_run.status == "completed":
        message = (
            f"{db_run.mode.capitalize()} execution completed successfully"
            if db_run.mode == "local"
            else "Dry-run validation successful"
        )
    elif db_run.status == "failed":
        message = f"{db_run.mode.capitalize()} execution failed"
        if db_run.exit_code is not None:
            message += f" with exit code {db_run.exit_code}"
    elif db_run.status == "timeout":
        message = f"{db_run.mode.capitalize()} execution timed out after 30 seconds"
    elif db_run.status == "running":
        message = f"{db_run.mode.capitalize()} execution in progress"
    elif db_run.status == "cancel_requested":
        message = f"{db_run.mode.capitalize()} execution cancellation requested"
    elif db_run.status == "cancelled":
        message = f"{db_run.mode.capitalize()} execution cancelled"
    else:  # pending
        message = f"{db_run.mode.capitalize()} execution pending"

    # Build response (reuse TunixRunResponse schema per M14 decision)
    return TunixRunResponse(
        run_id=str(db_run.run_id),
        status=db_run.status,  # type: ignore[arg-type]
        mode=db_run.mode,  # type: ignore[arg-type]
        dataset_key=db_run.dataset_key,
        model_id=db_run.model_id,
        output_dir="",  # Not stored in M14, would require additional schema field
        exit_code=db_run.exit_code,
        stdout=db_run.stdout,
        stderr=db_run.stderr,
        duration_seconds=db_run.duration_seconds,
        started_at=db_run.started_at.isoformat(),
        completed_at=db_run.completed_at.isoformat() if db_run.completed_at else None,
        message=message,
    )


@router.get("/api/tunix/runs/{run_id}/logs")
async def stream_run_logs(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    since_seq: int = 0,
) -> StreamingResponse:
    """Stream Tunix run logs via SSE (M16).

    Args:
        run_id: UUID of the run
        since_seq: Sequence number to start from (default 0)
        db: Database session

    Returns:
        StreamingResponse (SSE) with log events
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        current_seq = since_seq

        while True:
            # Fetch new chunks
            stmt = (
                select(TunixRunLogChunk)
                .where(TunixRunLogChunk.run_id == run_id)
                .where(TunixRunLogChunk.seq > current_seq)
                .order_by(TunixRunLogChunk.seq.asc())
            )
            result = await db.execute(stmt)
            chunks = result.scalars().all()

            if chunks:
                for chunk in chunks:
                    data = json.dumps(
                        {
                            "seq": chunk.seq,
                            "stream": chunk.stream,
                            "chunk": chunk.chunk,
                            "created_at": chunk.created_at.isoformat(),
                        }
                    )
                    yield f"event: log\ndata: {data}\n\n"
                    current_seq = chunk.seq

            # If no new chunks, check if run is finished
            else:
                run_stmt = select(TunixRun).where(TunixRun.run_id == run_id)
                run_result = await db.execute(run_stmt)
                run = run_result.scalar_one_or_none()

                if not run:
                    yield f"event: error\ndata: {json.dumps({'error': 'Run not found'})}\n\n"
                    break

                if run.status in ["completed", "failed", "timeout", "cancelled"]:
                    # Final check for chunks in case race condition
                    final_stmt = (
                        select(TunixRunLogChunk)
                        .where(TunixRunLogChunk.run_id == run_id)
                        .where(TunixRunLogChunk.seq > current_seq)
                        .order_by(TunixRunLogChunk.seq.asc())
                    )
                    final_result = await db.execute(final_stmt)
                    final_chunks = final_result.scalars().all()

                    for chunk in final_chunks:
                        data = json.dumps(
                            {
                                "seq": chunk.seq,
                                "stream": chunk.stream,
                                "chunk": chunk.chunk,
                                "created_at": chunk.created_at.isoformat(),
                            }
                        )
                        yield f"event: log\ndata: {data}\n\n"

                    yield f"event: status\ndata: {json.dumps({'status': run.status})}\n\n"
                    break

                # Send heartbeat
                yield f"event: heartbeat\ndata: {json.dumps({'seq': current_seq})}\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/tunix/runs/{run_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    """Cancel a pending or running Tunix run.

    Args:
        run_id: UUID of the run to cancel
        db: Database session

    Returns:
        Status message

    Raises:
        HTTPException: 404 if run not found
        HTTPException: 400 if run cannot be cancelled (already done)
    """
    try:
        await cancel_tunix_run(run_id, db)
    except ValueError as e:
        if "Run not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return {"message": "Run cancellation requested"}


@router.get("/api/tunix/runs/{run_id}/artifacts")
async def list_artifacts(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, list[dict[str, str | int]]]:
    """List artifacts for a Tunix run.

    Scans the run output directory and returns a list of files.

    Args:
        run_id: UUID of the run
        db: Database session

    Returns:
        List of artifacts with metadata
    """
    # Get run to find output_dir
    run = await db.get(TunixRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    # Check config for output_dir
    if not run.config or "output_dir" not in run.config:
        # Fallback to default convention if not in config
        output_dir = f"./output/tunix_run_{str(run.run_id)[:8]}"
    else:
        output_dir = run.config["output_dir"]
        if not output_dir:
            output_dir = f"./output/tunix_run_{str(run.run_id)[:8]}"

    # Scan directory
    artifacts: list[dict[str, str | int]] = []
    try:
        import os

        base_path = Path(output_dir)
        if base_path.exists() and base_path.is_dir():
            for entry in os.scandir(base_path):
                if entry.is_file():
                    artifacts.append(
                        {
                            "name": entry.name,
                            "size": entry.stat().st_size,
                            "path": entry.name,  # Only return relative name for security
                        }
                    )
    except Exception as e:
        logger.error(f"Error scanning artifacts for {run_id}: {e}")
        # Return empty list or error? Empty list implies no artifacts.

    return {"artifacts": artifacts}


@router.get("/api/tunix/runs/{run_id}/artifacts/{filename}/download")
async def download_artifact(
    run_id: uuid.UUID,
    filename: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    """Download an artifact file.

    Args:
        run_id: UUID of the run
        filename: Name of the file to download
        db: Database session

    Returns:
        FileResponse stream
    """
    # Get run to find output_dir
    run = await db.get(TunixRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    if not run.config or "output_dir" not in run.config:
        output_dir = f"./output/tunix_run_{str(run.run_id)[:8]}"
    else:
        output_dir = run.config["output_dir"] or f"./output/tunix_run_{str(run.run_id)[:8]}"

    # Validate path
    base_path = Path(output_dir).resolve()
    file_path = (base_path / filename).resolve()

    # Path traversal check
    if not str(file_path).startswith(str(base_path)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid artifact path",
        )

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found",
        )

    return FileResponse(path=file_path, filename=filename)


@router.get("/api/tunix/runs/{run_id}/metrics")
async def get_tunix_run_metrics(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[dict[str, Any]]:
    """Get time-series metrics for a Tunix run.

    Reads from the metrics.jsonl artifact in the run's output directory.
    Returns a list of metric points (step, loss, etc.).

    Args:
        run_id: UUID of the run
        db: Database session

    Returns:
        List of metric dictionaries
    """
    run = await db.get(TunixRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    # Determine output_dir (replicate logic from other endpoints)
    if not run.config or "output_dir" not in run.config:
        output_dir = f"./output/tunix_run_{str(run.run_id)[:8]}"
    else:
        output_dir = run.config["output_dir"] or f"./output/tunix_run_{str(run.run_id)[:8]}"

    metrics_path = Path(output_dir) / "metrics.jsonl"

    if not metrics_path.exists():
        return []

    metrics = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        logger.warning(f"Failed to read metrics for {run_id}: {e}")
        # Return what we have
        pass

    return metrics
