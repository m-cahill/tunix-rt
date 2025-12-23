"""FastAPI application with health endpoints."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.db.models import Score, Trace
from tunix_rt_backend.helpers.traces import get_trace_or_404
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
    TunixRunRequest,
    TunixRunResponse,
    UngarGenerateRequest,
    UngarGenerateResponse,
    UngarStatusResponse,
)
from tunix_rt_backend.scoring import baseline_score
from tunix_rt_backend.services.datasets_export import export_dataset_to_jsonl
from tunix_rt_backend.services.traces_batch import create_traces_batch_optimized
from tunix_rt_backend.settings import settings

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
    db.add(db_trace)
    await db.commit()
    await db.refresh(db_trace)

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
        return await create_traces_batch_optimized(traces, db)
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
) -> TunixRunResponse:
    """Execute a Tunix training run (M13).

    This endpoint supports two execution modes:
    - **dry-run** (default): Validates manifest + dataset without executing
    - **local**: Executes tunix CLI via subprocess with 30s timeout

    Args:
        request: Run configuration (dataset_key, model_id, hyperparameters, dry_run flag)
        db: Database session

    Returns:
        Run response with execution status, logs, and metadata

    Raises:
        HTTPException: 501 if Tunix not available and dry_run=False

    Note:
        - Dry-run mode always works (no Tunix required)
        - Local mode requires Tunix to be installed (backend[tunix] extra)
        - M13 does NOT persist run metadata to database (deferred to M14)
        - Execution is synchronous (blocking) with 30s timeout

    Example (dry-run):
        ```
        POST /api/tunix/run
        {
          "dataset_key": "my_dataset-v1",
          "model_id": "google/gemma-2b-it",
          "dry_run": true
        }
        ```

    Example (local execution):
        ```
        POST /api/tunix/run
        {
          "dataset_key": "my_dataset-v1",
          "model_id": "google/gemma-2b-it",
          "dry_run": false
        }
        ```
    """
    from tunix_rt_backend.integrations.tunix.availability import tunix_available
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    # Check Tunix availability for local execution
    if not request.dry_run and not tunix_available():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Tunix runtime not available. Install with: pip install -e '.[tunix]' "
            "or set dry_run=true to validate without executing.",
        )

    # Execute run (dry-run or local)
    return await execute_tunix_run(request, db)
