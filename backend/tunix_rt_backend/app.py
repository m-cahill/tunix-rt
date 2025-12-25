"""FastAPI application with health endpoints."""

import asyncio
import json
import logging
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# M15: Observability
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore[import-not-found]
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.db.models import Score, Trace, TunixRun, TunixRunLogChunk
from tunix_rt_backend.helpers.traces import get_trace_or_404
from tunix_rt_backend.metrics import (
    TUNIX_DB_WRITE_LATENCY_MS,
)
from tunix_rt_backend.redi_client import MockRediClient, RediClient, RediClientProtocol
from tunix_rt_backend.schemas import (
    CompareResponse,
    DatasetBuildRequest,
    DatasetBuildResponse,
    ExportFormat,
    PaginationInfo,
    ReasoningTrace,
    ScoreRequest,
    ScoreResponse,
    TraceBatchCreateResponse,
    TraceCreateResponse,
    TraceDetail,
    TraceListItem,
    TraceListResponse,
    TraceWithScore,
    TunixExportRequest,
    TunixManifestRequest,
    TunixManifestResponse,
    TunixRunListItem,
    TunixRunListResponse,
    TunixRunRequest,
    TunixRunResponse,
    TunixRunStatusResponse,
    UngarGenerateRequest,
    UngarGenerateResponse,
    UngarStatusResponse,
)

# M17: Evaluation schemas
from tunix_rt_backend.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    LeaderboardResponse,
)
from tunix_rt_backend.schemas.model_registry import (
    ModelArtifactCreate,
    ModelArtifactRead,
    ModelPromotionRequest,
    ModelVersionRead,
)

# M18: Regression schemas
from tunix_rt_backend.schemas.regression import (
    RegressionBaselineCreate,
    RegressionBaselineResponse,
    RegressionCheckRequest,
    RegressionCheckResult,
)

# M19: Tuning schemas
from tunix_rt_backend.schemas.tuning import (
    TuningJobCreate,
    TuningJobRead,
    TuningJobStartResponse,
)
from tunix_rt_backend.scoring import baseline_score
from tunix_rt_backend.services.datasets_export import export_dataset_to_jsonl
from tunix_rt_backend.services.model_registry import ModelRegistryService
from tunix_rt_backend.services.traces_batch import create_traces_batch
from tunix_rt_backend.services.tunix_execution import cancel_tunix_run
from tunix_rt_backend.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tunix RT Backend",
    description="Reasoning-Trace backend with RediAI integration",
    version="0.1.0",
)

# CORS middleware for frontend integration
# M4: Allow both localhost and 127.0.0.1 for dev (5173) and preview (4173) modes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (DNS)
        "http://127.0.0.1:5173",  # Vite dev server (IPv4)
        "http://localhost:4173",  # Vite preview (DNS)
        "http://127.0.0.1:4173",  # Vite preview (IPv4)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple TTL cache for RediAI health
_redi_health_cache: dict[str, tuple[dict[str, str], datetime]] = {}


async def validate_payload_size(request: Request) -> None:
    """Dependency to validate request payload size.

    Raises:
        HTTPException: 413 if payload exceeds TRACE_MAX_BYTES
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.trace_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Payload size exceeds maximum of {settings.trace_max_bytes} bytes",
        )


def get_redi_client() -> RediClientProtocol:
    """Dependency provider for RediAI client.

    Returns MockRediClient in mock mode, RediClient in real mode.
    This allows easy testing via dependency_overrides.
    """
    if settings.rediai_mode == "mock":
        return MockRediClient(simulate_healthy=True)
    return RediClient(base_url=settings.rediai_base_url, health_path=settings.rediai_health_path)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Check tunix-rt application health.

    Returns:
        {"status": "healthy"}
    """
    return {"status": "healthy"}


@app.get("/api/redi/health")
async def redi_health(
    redi_client: Annotated[RediClientProtocol, Depends(get_redi_client)],
) -> dict[str, str]:
    """Check RediAI integration health with TTL caching.

    In mock mode: always returns healthy.
    In real mode: probes actual RediAI instance (with 30s cache).

    Args:
        redi_client: Injected RediAI client (real or mock)

    Returns:
        {"status": "healthy"} if RediAI is reachable
        {"status": "down", "error": "..."} if RediAI is unreachable
    """
    # Check cache
    now = datetime.now(timezone.utc)
    cache_ttl = timedelta(seconds=settings.rediai_health_cache_ttl_seconds)

    if "redi_health" in _redi_health_cache:
        cached_result, cached_time = _redi_health_cache["redi_health"]
        if now - cached_time < cache_ttl:
            return cached_result
        # Implicit else: cache expired, fall through to fetch
    # Implicit else: no cache entry, fall through to fetch

    # Cache miss or expired - fetch fresh result
    result = await redi_client.health()
    _redi_health_cache["redi_health"] = (result, now)
    return result


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics (M15)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/api/traces",
    response_model=TraceCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(validate_payload_size)],
)
async def create_trace(
    trace: ReasoningTrace,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraceCreateResponse:
    """Create a new reasoning trace.

    Args:
        trace: ReasoningTrace payload (validated by Pydantic)
        db: Database session

    Returns:
        TraceCreateResponse with id, created_at, trace_version

    Raises:
        HTTPException: 413 if payload size exceeds limit
        HTTPException: 422 if validation fails
    """
    # Create trace model
    db_trace = Trace(
        trace_version=trace.trace_version,
        payload=trace.model_dump(),
    )

    # Save to database
    start_time = time.perf_counter()
    db.add(db_trace)
    await db.commit()
    await db.refresh(db_trace)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"DB commit latency for create_trace: {duration_ms:.2f}ms")
    TUNIX_DB_WRITE_LATENCY_MS.labels(operation="create_trace").observe(duration_ms)

    return TraceCreateResponse(
        id=db_trace.id,
        created_at=db_trace.created_at,
        trace_version=db_trace.trace_version,
    )


@app.post(
    "/api/traces/batch",
    response_model=TraceBatchCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_traces_batch_endpoint(
    traces: list[ReasoningTrace],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraceBatchCreateResponse:
    """Create multiple reasoning traces in a single transaction.

    Useful for bulk importing traces from evaluation runs or other sources.
    All traces are validated before any are inserted. If any trace is invalid,
    the entire batch is rejected.

    Args:
        traces: List of ReasoningTrace payloads (validated by Pydantic)
        db: Database session

    Returns:
        TraceBatchCreateResponse with created_count and list of created traces

    Raises:
        HTTPException: 422 if any trace fails validation
        HTTPException: 400 if batch is empty or exceeds max size

    Note:
        Maximum batch size is 1000 traces per request.
    """
    # Delegate to service layer (optimized version with bulk refresh)
    try:
        return await create_traces_batch(traces, db)
    except ValueError as e:
        # Convert service-level ValueError to HTTP 400
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.get("/api/traces/compare", response_model=CompareResponse)
async def compare_traces(
    db: Annotated[AsyncSession, Depends(get_db)],
    base: uuid.UUID,
    other: uuid.UUID,
) -> CompareResponse:
    """Compare two traces side-by-side with scores.

    Args:
        base: UUID of the base trace
        other: UUID of the other trace
        db: Database session

    Returns:
        CompareResponse with both traces and their scores

    Raises:
        HTTPException: 404 if either trace not found
    """
    # Fetch both traces using helper with labels for clear error messages
    base_trace = await get_trace_or_404(db, base, label="Base")
    other_trace = await get_trace_or_404(db, other, label="Other")

    # Parse trace payloads
    base_payload = ReasoningTrace(**base_trace.payload)
    other_payload = ReasoningTrace(**other_trace.payload)

    # Compute scores for both traces using baseline scorer
    base_score_value, _ = baseline_score(base_payload)
    other_score_value, _ = baseline_score(other_payload)

    # Build response
    return CompareResponse(
        base=TraceWithScore(
            id=base_trace.id,
            created_at=base_trace.created_at,
            score=base_score_value,
            trace_version=base_trace.trace_version,
            payload=base_payload,
        ),
        other=TraceWithScore(
            id=other_trace.id,
            created_at=other_trace.created_at,
            score=other_score_value,
            trace_version=other_trace.trace_version,
            payload=other_payload,
        ),
    )


@app.get("/api/traces/{trace_id}", response_model=TraceDetail)
async def get_trace(
    trace_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TraceDetail:
    """Get a trace by ID.

    Args:
        trace_id: UUID of the trace
        db: Database session

    Returns:
        TraceDetail with full payload

    Raises:
        HTTPException: 404 if trace not found
    """
    db_trace = await get_trace_or_404(db, trace_id)

    return TraceDetail(
        id=db_trace.id,
        created_at=db_trace.created_at,
        trace_version=db_trace.trace_version,
        payload=ReasoningTrace(**db_trace.payload),
    )


@app.get("/api/traces", response_model=TraceListResponse)
async def list_traces(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> TraceListResponse:
    """List traces with pagination.

    Args:
        limit: Maximum number of traces to return (max 100, default 20)
        offset: Offset for pagination (default 0)
        db: Database session

    Returns:
        TraceListResponse with data and pagination info

    Raises:
        HTTPException: 422 if limit > 100 or offset < 0
    """
    # Validate pagination parameters
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be between 1 and 100",
        )
    # limit is valid - continue

    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="offset must be non-negative",
        )
    # offset is valid - continue

    # Query traces with limit + 1 to check if there are more
    result = await db.execute(
        select(Trace).order_by(Trace.created_at.desc()).limit(limit + 1).offset(offset)
    )
    db_traces = result.scalars().all()

    # Check if there are more results
    has_more = len(db_traces) > limit
    traces_to_return = db_traces[:limit]

    # Build response
    data = [
        TraceListItem(
            id=trace.id,
            created_at=trace.created_at,
            trace_version=trace.trace_version,
        )
        for trace in traces_to_return
    ]

    next_offset = offset + limit if has_more else None

    return TraceListResponse(
        data=data,
        pagination=PaginationInfo(
            limit=limit,
            offset=offset,
            next_offset=next_offset,
        ),
    )


@app.post(
    "/api/traces/{trace_id}/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_201_CREATED,
)
async def score_trace(
    trace_id: uuid.UUID,
    score_request: ScoreRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ScoreResponse:
    """Score a trace using specified criteria.

    Args:
        trace_id: UUID of the trace to score
        score_request: Scoring request with criteria
        db: Database session

    Returns:
        ScoreResponse with score value and details

    Raises:
        HTTPException: 404 if trace not found
    """
    # Fetch the trace using helper
    db_trace = await get_trace_or_404(db, trace_id)

    # Parse the trace payload
    trace = ReasoningTrace(**db_trace.payload)

    # Compute score based on criteria
    if score_request.criteria == "baseline":
        score_value, details = baseline_score(trace)
    else:  # pragma: no cover
        # This shouldn't happen due to Literal type, but defensive check
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported criteria: {score_request.criteria}",
        )

    # Store score in database
    db_score = Score(
        trace_id=trace_id,
        criteria=score_request.criteria,
        score=score_value,
        details=details.model_dump(mode="json"),  # Use mode="json" for proper serialization
    )
    db.add(db_score)
    await db.commit()
    await db.refresh(db_score)

    return ScoreResponse(
        trace_id=trace_id,
        score=score_value,
        details=details,
    )


# ==================== UNGAR Integration Endpoints ====================


@app.get("/api/ungar/status", response_model=UngarStatusResponse)
async def ungar_status() -> UngarStatusResponse:
    """Check UNGAR availability and version.

    Returns:
        UngarStatusResponse with availability status and version (if available)

    Note:
        This endpoint always succeeds (200 OK) but the 'available' field
        indicates whether UNGAR is actually installed.
    """
    from tunix_rt_backend.services.ungar_generator import check_ungar_status

    return check_ungar_status()


@app.post(
    "/api/ungar/high-card-duel/generate",
    response_model=UngarGenerateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ungar_generate_high_card_duel(
    request: UngarGenerateRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UngarGenerateResponse:
    """Generate High Card Duel traces from UNGAR episodes.

    Args:
        request: Generation parameters (count, seed, persist)
        db: Database session

    Returns:
        UngarGenerateResponse with created trace IDs and preview

    Raises:
        HTTPException: 501 if UNGAR is not installed
    """
    from tunix_rt_backend.services.ungar_generator import (
        generate_high_card_duel_traces as generate_traces,
    )

    # Delegate to service layer
    try:
        trace_ids, preview = await generate_traces(request, db)
    except ValueError as e:
        # Convert service-level ValueError to HTTP 501
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )

    return UngarGenerateResponse(trace_ids=trace_ids, preview=preview)


@app.get("/api/ungar/high-card-duel/export.jsonl")
async def ungar_export_high_card_duel_jsonl(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 100,
    trace_ids: str | None = None,
) -> Response:
    """Export High Card Duel traces in JSONL format (Tunix-friendly).

    Args:
        limit: Maximum number of traces to export (default 100)
        trace_ids: Comma-separated list of trace IDs to export (optional)
        db: Database session

    Returns:
        Response with Content-Type: application/x-ndjson

    Note:
        If trace_ids is provided, only those traces are exported.
        Otherwise, exports the most recent traces up to limit.
    """
    from tunix_rt_backend.services.ungar_generator import export_high_card_duel_jsonl

    # Delegate to service layer
    content = await export_high_card_duel_jsonl(db, limit, trace_ids)
    return Response(content=content, media_type="application/x-ndjson")


# ================================================================================
# Dataset Endpoints
# ================================================================================


@app.post(
    "/api/datasets/build",
    response_model=DatasetBuildResponse,
    status_code=status.HTTP_201_CREATED,
)
async def build_dataset(
    request: DatasetBuildRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DatasetBuildResponse:
    """Build a new dataset from traces.

    Creates a dataset manifest and saves it to disk. The dataset can then be
    exported using the export endpoint.

    Args:
        request: Dataset build parameters
        db: Database session

    Returns:
        Dataset build response with dataset_key and build_id

    Raises:
        HTTPException: 422 if validation fails (e.g., random strategy without seed)
    """
    from tunix_rt_backend.services.datasets_builder import build_dataset_manifest

    # Delegate to service layer
    try:
        (
            dataset_key,
            build_id,
            trace_count,
            manifest_path,
        ) = await build_dataset_manifest(request, db)
    except ValueError as e:
        # Convert service-level ValueError to HTTP 422
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return DatasetBuildResponse(
        dataset_key=dataset_key,
        build_id=build_id,
        trace_count=trace_count,
        manifest_path=str(manifest_path),
    )


@app.get("/api/datasets/{dataset_key}/export.jsonl")
async def export_dataset(
    dataset_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    format: ExportFormat = "trace",
) -> Response:
    """Export a dataset as JSONL.

    Loads the dataset manifest and exports all included traces in JSONL format.
    Supports three formats:
    - 'trace': Raw trace data (default)
    - 'tunix_sft': Formatted for Tunix SFT training with rendered prompts
    - 'training_example': TrainingExample objects (prompt/response pairs)

    Args:
        dataset_key: Dataset identifier (name-version)
        db: Database session
        format: Export format ('trace', 'tunix_sft', or 'training_example', default: 'trace')

    Returns:
        NDJSON response with one trace per line

    Raises:
        HTTPException: 404 if dataset not found
        HTTPException: 422 if format is invalid (FastAPI validates automatically)
    """
    from tunix_rt_backend.helpers.datasets import load_manifest

    # Load manifest
    try:
        manifest = load_manifest(dataset_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_key}",
        )

    # Delegate to service layer for export formatting
    content = await export_dataset_to_jsonl(manifest, db, format)

    # Return as NDJSON
    return Response(content=content, media_type="application/x-ndjson")


# ================================================================================
# TUNIX INTEGRATION ENDPOINTS (M12)
#
# M12 Design: Mock-first, artifact-based integration
# - No Tunix runtime dependency required
# - Generates JSONL exports + YAML manifests
# - Reuses existing tunix_sft export format from M09
# ================================================================================


@app.get("/api/tunix/status")
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


@app.post("/api/tunix/sft/export")
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


@app.post(
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


@app.post("/api/tunix/run", status_code=status.HTTP_200_OK)
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


# ================================================================================
# M14: Tunix Run Registry Endpoints
# ================================================================================


@app.get("/api/tunix/runs/{run_id}/status", response_model=TunixRunStatusResponse)
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
    from sqlalchemy import select

    from tunix_rt_backend.db.models import TunixRun

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


@app.get("/api/tunix/runs", response_model=TunixRunListResponse)
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
    from sqlalchemy import select

    from tunix_rt_backend.db.models import TunixRun

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


@app.get("/api/tunix/runs/{run_id}", response_model=TunixRunResponse)
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
    from sqlalchemy import select

    from tunix_rt_backend.db.models import TunixRun

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


@app.get("/api/tunix/runs/{run_id}/logs")
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


@app.post("/api/tunix/runs/{run_id}/cancel", status_code=status.HTTP_200_OK)
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


@app.get("/api/tunix/runs/{run_id}/artifacts")
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
        from pathlib import Path

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


@app.get("/api/tunix/runs/{run_id}/artifacts/{filename}/download")
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
    from pathlib import Path

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


# ================================================================================
# M17: Evaluation Endpoints
# ================================================================================


@app.post(
    "/api/tunix/runs/{run_id}/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def evaluate_tunix_run(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    redi_client: Annotated[RediClientProtocol, Depends(get_redi_client)],
    request: EvaluationRequest | None = None,
) -> EvaluationResponse:
    """Trigger evaluation for a completed run (M17).

    Args:
        run_id: UUID of the run
        request: Optional evaluation parameters (judge_override)
        db: Database session
        redi_client: RediAI client (injected)

    Returns:
        EvaluationResponse with results

    Raises:
        HTTPException: 404 if run not found
        HTTPException: 400 if run not in completed state
    """
    from tunix_rt_backend.services.evaluation import EvaluationService

    service = EvaluationService(db, redi_client)
    try:
        judge_override = request.judge_override if request else None
        return await service.evaluate_run(run_id, judge_override)
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.get("/api/tunix/runs/{run_id}/evaluation", response_model=EvaluationResponse)
async def get_tunix_run_evaluation(
    run_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EvaluationResponse:
    """Get evaluation details for a run (M17).

    Args:
        run_id: UUID of the run
        db: Database session

    Returns:
        EvaluationResponse

    Raises:
        HTTPException: 404 if evaluation not found
    """
    from tunix_rt_backend.services.evaluation import EvaluationService

    service = EvaluationService(db)
    evaluation = await service.get_evaluation(run_id)

    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation not found",
        )

    return evaluation


@app.get("/api/tunix/evaluations", response_model=LeaderboardResponse)
async def get_leaderboard(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
    offset: int = 0,
) -> LeaderboardResponse:
    """Get leaderboard data (M17).

    Returns:
        LeaderboardResponse with sorted evaluation results
    """
    from tunix_rt_backend.services.evaluation import EvaluationService

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

    service = EvaluationService(db)
    return await service.get_leaderboard(limit=limit, offset=offset)


# ================================================================================
# M18: Regression Endpoints
# ================================================================================


@app.post(
    "/api/regression/baselines",
    response_model=RegressionBaselineResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_regression_baseline(
    request: RegressionBaselineCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegressionBaselineResponse:
    """Create or update a named regression baseline."""
    from tunix_rt_backend.services.regression import RegressionService

    service = RegressionService(db)
    try:
        return await service.create_baseline(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post(
    "/api/regression/check",
    response_model=RegressionCheckResult,
)
async def check_regression(
    request: RegressionCheckRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegressionCheckResult:
    """Check for regression against a baseline."""
    from tunix_rt_backend.services.regression import RegressionService

    service = RegressionService(db)
    try:
        return await service.check_regression(request.run_id, request.baseline_name)
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ================================================================================
# M19: Tuning Endpoints
# ================================================================================


@app.post(
    "/api/tuning/jobs",
    response_model=TuningJobRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_tuning_job(
    request: TuningJobCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TuningJobRead:
    """Create a new hyperparameter tuning job."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    job = await service.create_job(request)

    # Convert DB model to Read schema
    return TuningJobRead(
        id=job.id,
        name=job.name,
        status=job.status,
        dataset_key=job.dataset_key,
        base_model_id=job.base_model_id,
        mode=job.mode,
        metric_name=job.metric_name,
        metric_mode=job.metric_mode,
        num_samples=job.num_samples,
        max_concurrent_trials=job.max_concurrent_trials,
        search_space_json=job.search_space_json,
        best_run_id=job.best_run_id,
        best_params_json=job.best_params_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.post(
    "/api/tuning/jobs/{job_id}/start",
    response_model=TuningJobStartResponse,
    status_code=status.HTTP_200_OK,
)
async def start_tuning_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TuningJobStartResponse:
    """Start a tuning job."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    try:
        await service.start_job(job_id)
    except RuntimeError as e:
        # Ray not installed
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return TuningJobStartResponse(
        job_id=job_id,
        status="running",
        message="Tuning job started",
    )


@app.get("/api/tuning/jobs", response_model=list[TuningJobRead])
async def list_tuning_jobs(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> list[TuningJobRead]:
    """List tuning jobs."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    jobs = await service.list_jobs(limit, offset)

    return [
        TuningJobRead(
            id=job.id,
            name=job.name,
            status=job.status,
            dataset_key=job.dataset_key,
            base_model_id=job.base_model_id,
            mode=job.mode,
            metric_name=job.metric_name,
            metric_mode=job.metric_mode,
            num_samples=job.num_samples,
            max_concurrent_trials=job.max_concurrent_trials,
            search_space_json=job.search_space_json,
            best_run_id=job.best_run_id,
            best_params_json=job.best_params_json,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@app.get("/api/tuning/jobs/{job_id}", response_model=TuningJobRead)
async def get_tuning_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TuningJobRead:
    """Get tuning job details."""
    from sqlalchemy import select

    from tunix_rt_backend.db.models import TunixTuningTrial
    from tunix_rt_backend.schemas.tuning import TuningTrialRead
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    job = await service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Fetch trials
    trials_result = await db.execute(
        select(TunixTuningTrial)
        .where(TunixTuningTrial.tuning_job_id == job_id)
        .order_by(TunixTuningTrial.created_at)
    )
    trials = trials_result.scalars().all()

    trial_reads = [
        TuningTrialRead(
            id=t.id,
            tuning_job_id=t.tuning_job_id,
            run_id=t.run_id,
            params_json=t.params_json,
            metric_value=t.metric_value,
            status=t.status,
            error=t.error,
            created_at=t.created_at,
            completed_at=t.completed_at,
        )
        for t in trials
    ]

    return TuningJobRead(
        id=job.id,
        name=job.name,
        status=job.status,
        dataset_key=job.dataset_key,
        base_model_id=job.base_model_id,
        mode=job.mode,
        metric_name=job.metric_name,
        metric_mode=job.metric_mode,
        num_samples=job.num_samples,
        max_concurrent_trials=job.max_concurrent_trials,
        search_space_json=job.search_space_json,
        best_run_id=job.best_run_id,
        best_params_json=job.best_params_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        trials=trial_reads,
    )


# ================================================================================
# M20: Model Registry
# ================================================================================


@app.post("/api/models", response_model=ModelArtifactRead, status_code=status.HTTP_201_CREATED)
async def create_model_artifact(
    request: ModelArtifactCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelArtifactRead:
    """Create a new model artifact family."""
    service = ModelRegistryService(db)
    try:
        artifact = await service.create_artifact(request)
        return ModelArtifactRead(
            id=artifact.id,
            name=artifact.name,
            description=artifact.description,
            task_type=artifact.task_type,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@app.get("/api/models", response_model=list[ModelArtifactRead])
async def list_model_artifacts(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelArtifactRead]:
    """List all model artifacts."""
    service = ModelRegistryService(db)
    artifacts = await service.list_artifacts()
    return [
        ModelArtifactRead(
            id=a.id,
            name=a.name,
            description=a.description,
            task_type=a.task_type,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in artifacts
    ]


@app.get("/api/models/{artifact_id}", response_model=ModelArtifactRead)
async def get_model_artifact(
    artifact_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelArtifactRead:
    """Get details of a model artifact."""
    service = ModelRegistryService(db)
    artifact = await service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    return ModelArtifactRead(
        id=artifact.id,
        name=artifact.name,
        description=artifact.description,
        task_type=artifact.task_type,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


@app.post("/api/models/{artifact_id}/versions/promote", response_model=ModelVersionRead)
async def promote_run_to_version(
    artifact_id: uuid.UUID,
    request: ModelPromotionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelVersionRead:
    """Promote a TunixRun to a ModelVersion."""
    service = ModelRegistryService(db)
    try:
        version = await service.promote_run(artifact_id, request)
        return ModelVersionRead(
            id=version.id,
            artifact_id=version.artifact_id,
            version=version.version,
            source_run_id=version.source_run_id,
            status=version.status,
            metrics_json=version.metrics_json,
            config_json=version.config_json,
            provenance_json=version.provenance_json,
            storage_uri=version.storage_uri,
            sha256=version.sha256,
            size_bytes=version.size_bytes,
            created_at=version.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/models/versions/{version_id}", response_model=ModelVersionRead)
async def get_model_version(
    version_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelVersionRead:
    """Get details of a model version."""
    service = ModelRegistryService(db)
    version = await service.get_version(version_id)
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    return ModelVersionRead(
        id=version.id,
        artifact_id=version.artifact_id,
        version=version.version,
        source_run_id=version.source_run_id,
        status=version.status,
        metrics_json=version.metrics_json,
        config_json=version.config_json,
        provenance_json=version.provenance_json,
        storage_uri=version.storage_uri,
        sha256=version.sha256,
        size_bytes=version.size_bytes,
        created_at=version.created_at,
    )


@app.get("/api/models/versions/{version_id}/download")
async def download_model_version(
    version_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    """Download a model version (as zip stream)."""
    service = ModelRegistryService(db)
    version = await service.get_version(version_id)
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    # Locate directory
    path = service.storage.get(version.storage_uri)

    # Create temp file for zip
    temp_dir = tempfile.mkdtemp()
    base_name = str(Path(temp_dir) / f"{version.version}")

    try:
        # Run zip in thread to avoid blocking
        zip_path = await asyncio.to_thread(shutil.make_archive, base_name, "zip", root_dir=path)

        # Cleanup temp dir after response
        return FileResponse(
            zip_path,
            filename=f"{version.artifact_id}_{version.version}.zip",
            background=BackgroundTask(shutil.rmtree, temp_dir),
        )
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to zip artifact: {e}")
