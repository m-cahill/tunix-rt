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
    PaginationInfo,
    ReasoningTrace,
    ScoreRequest,
    ScoreResponse,
    TraceCreateResponse,
    TraceDetail,
    TraceListItem,
    TraceListResponse,
    TraceWithScore,
    UngarGenerateRequest,
    UngarGenerateResponse,
    UngarStatusResponse,
)
from tunix_rt_backend.scoring import baseline_score
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
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="limit must be between 1 and 100",
        )
    # limit is valid - continue

    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
    from tunix_rt_backend.integrations.ungar.availability import (
        ungar_available,
        ungar_version,
    )

    available = ungar_available()
    version = ungar_version() if available else None

    return UngarStatusResponse(available=available, version=version)


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
    from tunix_rt_backend.integrations.ungar.availability import ungar_available

    # Check if UNGAR is available
    if not ungar_available():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="UNGAR is not installed. Install with: pip install -e '.[ungar]'",
        )

    # Import generator (lazy to avoid import errors when UNGAR not installed)
    from tunix_rt_backend.integrations.ungar.high_card_duel import (
        generate_high_card_duel_traces,
    )

    # Generate traces
    traces = generate_high_card_duel_traces(count=request.count, seed=request.seed)

    # Persist to database if requested
    trace_ids: list[uuid.UUID] = []
    if request.persist:
        for trace in traces:
            db_trace = Trace(
                trace_version=trace.trace_version,
                payload=trace.model_dump(),
            )
            db.add(db_trace)
            await db.commit()
            await db.refresh(db_trace)
            trace_ids.append(db_trace.id)
    else:
        # Generate placeholder IDs for preview
        trace_ids = [uuid.uuid4() for _ in traces]

    # Build preview (first 3 traces, metadata only)
    preview = [
        {
            "trace_id": str(trace_ids[i]),
            "game": trace.meta.get("game") if trace.meta else None,
            "result": trace.meta.get("result") if trace.meta else None,
            "my_card": trace.meta.get("my_card") if trace.meta else None,
        }
        for i, trace in enumerate(traces[:3])
    ]

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
    # Build query
    if trace_ids:
        # Parse comma-separated UUIDs
        ids = [uuid.UUID(tid.strip()) for tid in trace_ids.split(",")]
        result = await db.execute(select(Trace).where(Trace.id.in_(ids)))
    else:
        # Get most recent traces with source="ungar" in metadata
        # Note: We fetch all traces and filter in Python for compatibility
        # (JSON querying syntax varies between SQLite and PostgreSQL)
        result = await db.execute(
            select(Trace)
            .order_by(Trace.created_at.desc())
            .limit(limit * 10)  # Fetch more to account for filtering
        )

    all_traces = result.scalars().all()

    # Filter for UNGAR traces (Python-level filtering for DB compatibility)
    db_traces = [t for t in all_traces if t.payload.get("meta", {}).get("source") == "ungar"][
        :limit
    ]

    # Convert to JSONL
    import json

    lines = []
    for db_trace in db_traces:
        trace_payload = ReasoningTrace(**db_trace.payload)

        # Build Tunix-friendly JSONL record
        record = {
            "id": str(db_trace.id),
            "prompts": trace_payload.prompt,  # Use 'prompts' for Tunix compatibility
            "trace_steps": [step.content for step in trace_payload.steps],
            "final_answer": trace_payload.final_answer,
            "metadata": {
                "created_at": db_trace.created_at.isoformat(),
                "trace_version": db_trace.trace_version,
                **(trace_payload.meta or {}),
            },
        }
        lines.append(json.dumps(record))

    # Return as NDJSON
    content = "\n".join(lines) + "\n" if lines else ""
    return Response(content=content, media_type="application/x-ndjson")
