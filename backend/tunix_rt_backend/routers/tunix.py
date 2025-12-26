"""Tunix integration endpoints.

Domain: Tunix training orchestration (status, export, run trigger)

Primary endpoints:
- GET /api/tunix/status: Check Tunix availability
- POST /api/tunix/export: Export traces for Tunix training
- POST /api/tunix/manifest: Generate training manifest
- POST /api/tunix/run: Trigger training run (sync or async mode)

Cross-cutting concerns:
- Graceful degradation when Tunix not installed (501)
- Dry-run validation for manifest/config before execution
- Async mode queues runs for background worker
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.schemas import (
    TunixExportRequest,
    TunixManifestRequest,
    TunixManifestResponse,
    TunixRunRequest,
    TunixRunResponse,
)

router = APIRouter()


@router.get("/api/tunix/status")
async def tunix_status() -> dict[str, str | bool | None]:
    """Check Tunix integration status.

    Returns Tunix availability and configuration information. M13 adds
    optional runtime execution capability (dry-run + local modes).

    Returns:
        Status information including availability, version, and runtime requirements

    Note:
        M13: Returns available=True/False based on actual Tunix installation
        Dry-run mode always works (no Tunix required)
        Local execution requires Tunix to be installed
    """
    from tunix_rt_backend.integrations.tunix.availability import (
        tunix_available,
        tunix_version,
    )

    available = tunix_available()

    if available:
        message = (
            "Tunix runtime is available. Supports dry-run validation and local execution. "
            "Use POST /api/tunix/run with dry_run=false to execute training runs locally."
        )
    else:
        message = (
            "Tunix runtime not installed. Dry-run mode available for validation. "
            "Install Tunix for local execution: pip install -e '.[tunix]'"
        )

    return {
        "available": available,
        "version": tunix_version(),
        "runtime_required": not available,  # Runtime required only if not available
        "message": message,
    }


@router.post("/api/tunix/sft/export")
async def tunix_export_sft(
    request: TunixExportRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Export traces in Tunix SFT format (JSONL).

    Exports traces using the tunix_sft format from M09 (Gemma chat template
    with reasoning steps). Supports two modes:
    1. Export from dataset key (uses dataset manifest)
    2. Export from specific trace IDs

    Args:
        request: Export request with dataset_key OR trace_ids
        db: Database session

    Returns:
        NDJSON response with Tunix SFT formatted traces

    Raises:
        HTTPException: 400 if neither dataset_key nor trace_ids provided
        HTTPException: 404 if dataset not found
    """
    from tunix_rt_backend.services.tunix_export import export_tunix_sft_jsonl

    # Validate request has at least one export mode
    if not request.dataset_key and not request.trace_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either dataset_key or trace_ids must be provided",
        )

    # Delegate to service layer
    try:
        jsonl_content = await export_tunix_sft_jsonl(request, db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Return as NDJSON
    return Response(content=jsonl_content, media_type="application/x-ndjson")


@router.post(
    "/api/tunix/sft/manifest",
    response_model=TunixManifestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def tunix_generate_manifest(
    request: TunixManifestRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TunixManifestResponse:
    """Generate a Tunix SFT training run manifest.

    Creates a YAML configuration file that can be used to execute Tunix SFT
    training on a local machine or TPU VM. The manifest references the dataset
    and includes hyperparameters.

    Args:
        request: Manifest generation request with dataset_key, model_id, and hyperparameters
        db: Database session

    Returns:
        Manifest response with YAML content and metadata

    Raises:
        HTTPException: 404 if dataset not found

    Note:
        Generated manifest assumes dataset has been exported to a JSONL file.
        Typical workflow:
        1. POST /api/tunix/sft/export to create dataset JSONL
        2. POST /api/tunix/sft/manifest to create training config
        3. Execute training locally: tunix train --config manifest.yaml
    """
    from tunix_rt_backend.helpers.datasets import load_manifest
    from tunix_rt_backend.integrations.tunix.manifest import build_sft_manifest

    # Verify dataset exists
    try:
        load_manifest(request.dataset_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {request.dataset_key}",
        )

    # Build dataset path (convention: datasets/{dataset_key}.jsonl)
    dataset_path = f"./datasets/{request.dataset_key}.jsonl"

    # Generate manifest
    manifest_yaml = build_sft_manifest(request, dataset_path)

    # Return response
    return TunixManifestResponse(
        manifest_yaml=manifest_yaml,
        dataset_key=request.dataset_key,
        model_id=request.model_id,
        format="tunix_sft",
        message=f"Manifest generated for dataset {request.dataset_key}. "
        f"Save as YAML and execute with Tunix CLI.",
    )


@router.post("/api/tunix/run", status_code=status.HTTP_200_OK)
async def tunix_run(
    request: TunixRunRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    mode: Annotated[str | None, Query()] = None,
) -> TunixRunResponse:
    """Execute a Tunix training run (M13/M15).

    This endpoint supports execution modes:
    - **dry-run** (default): Validates manifest + dataset without executing
    - **local**: Executes tunix CLI via subprocess with 30s timeout

    M15 Update: Supports async execution via `?mode=async`.
    - If `mode=async`, the run is enqueued (status=pending) and returns immediately.
    - Otherwise, execution is synchronous (blocking).

    Args:
        request: Run configuration (dataset_key, model_id, hyperparameters, dry_run flag)
        db: Database session
        mode: Execution strategy ("async" or None)

    Returns:
        Run response with execution status, logs, and metadata

    Raises:
        HTTPException: 501 if Tunix not available and dry_run=False (and not async)

    Note:
        - Dry-run mode always works (no Tunix required)
        - Local mode requires Tunix to be installed (backend[tunix] extra)
        - All runs are persisted to database (M14)
    """
    from tunix_rt_backend.integrations.tunix.availability import tunix_available
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    async_mode = mode == "async"

    # Check Tunix availability for local execution (unless dry-run or async)
    # Note: If async, worker will check availability later.
    # But for now, let's enforce it here too if not dry-run, to fail fast?
    # Actually, if async, we might want to allow enqueueing even if this node doesn't have Tunix?
    # But for simplicity, let's keep the check for now or skip it for async.
    # The plan says "API can enqueue a run and return immediately".
    # So strictly, we shouldn't check runtime here if async.
    # But if dry_run=True, we don't check anyway.
    # If dry_run=False and async=False, we check.

    if not request.dry_run and not async_mode and not tunix_available():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Tunix runtime not available. Install with: pip install -e '.[tunix]' "
            "or set dry_run=true to validate without executing.",
        )

    # Execute run (dry-run, local, or async enqueue)
    return await execute_tunix_run(request, db, async_mode=async_mode)
