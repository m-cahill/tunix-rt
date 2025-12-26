"""Traces CRUD endpoints.

Domain: Reasoning trace storage and retrieval

Primary endpoints:
- POST /api/traces: Create trace with validation
- GET /api/traces/{id}: Retrieve single trace
- GET /api/traces: List traces with pagination
- POST /api/traces/compare: Compare two traces with scoring
- POST /api/traces/batch: Bulk import traces

Cross-cutting concerns:
- Payload size validation (enforced via dependency)
- Automatic scoring on creation
- Pagination for list endpoint (limit/offset)
"""

import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.db.models import Score, Trace
from tunix_rt_backend.dependencies import validate_payload_size
from tunix_rt_backend.helpers.traces import get_trace_or_404
from tunix_rt_backend.metrics import TUNIX_DB_WRITE_LATENCY_MS
from tunix_rt_backend.schemas import (
    CompareResponse,
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
)
from tunix_rt_backend.scoring import baseline_score
from tunix_rt_backend.services.traces_batch import create_traces_batch

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
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


@router.post(
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


@router.get("/api/traces/compare", response_model=CompareResponse)
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


@router.get("/api/traces/{trace_id}", response_model=TraceDetail)
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


@router.get("/api/traces", response_model=TraceListResponse)
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


@router.post(
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
