"""Trace batch import service.

This module handles bulk trace creation logic, including validation,
model creation, and transaction management.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas import ReasoningTrace, TraceBatchCreateResponse, TraceCreateResponse


async def create_traces_batch(
    traces: list[ReasoningTrace],
    db: AsyncSession,
) -> TraceBatchCreateResponse:
    """Create multiple reasoning traces in a single transaction.

    This service function handles:
    - Batch size validation
    - Trace model creation
    - Transactional insert (all-or-nothing)
    - Response construction

    Args:
        traces: List of ReasoningTrace payloads (already validated by Pydantic)
        db: Database session

    Returns:
        TraceBatchCreateResponse with created_count and list of created traces

    Raises:
        ValueError: If batch is empty or exceeds max size

    Note:
        Maximum batch size is 1000 traces per request.
        All validation happens before any database operations.
    """
    # Validate batch size
    if not traces:
        raise ValueError("Batch must contain at least one trace")

    max_batch_size = 1000
    if len(traces) > max_batch_size:
        raise ValueError(f"Batch size ({len(traces)}) exceeds maximum ({max_batch_size})")

    # Create all trace models
    db_traces = []
    for trace in traces:
        db_trace = Trace(
            trace_version=trace.trace_version,
            payload=trace.model_dump(),
        )
        db_traces.append(db_trace)

    # Insert all traces in a single transaction
    db.add_all(db_traces)
    await db.commit()

    # Refresh all traces to get generated IDs and timestamps
    # NOTE: This currently does N individual SELECT queries.
    # For large batches (>500 traces), consider bulk refresh optimization.
    # See M10 Phase 3 for potential improvement.
    for db_trace in db_traces:
        await db.refresh(db_trace)

    # Build response
    created_traces = [
        TraceCreateResponse(
            id=db_trace.id,
            created_at=db_trace.created_at,
            trace_version=db_trace.trace_version,
        )
        for db_trace in db_traces
    ]

    return TraceBatchCreateResponse(
        created_count=len(created_traces),
        traces=created_traces,
    )


async def create_traces_batch_optimized(
    traces: list[ReasoningTrace],
    db: AsyncSession,
) -> TraceBatchCreateResponse:
    """Optimized version of create_traces_batch with bulk refresh.

    This version uses a single bulk SELECT after commit to refresh all traces,
    avoiding N individual queries.

    WARNING: Do NOT use concurrent operations on the same AsyncSession.
    AsyncSession is mutable/stateful and not safe for concurrent access.
    See: https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html

    Args:
        traces: List of ReasoningTrace payloads
        db: Database session

    Returns:
        TraceBatchCreateResponse with created_count and list of created traces

    Raises:
        ValueError: If batch is empty or exceeds max size
    """
    from sqlalchemy import select

    # Validate batch size
    if not traces:
        raise ValueError("Batch must contain at least one trace")

    max_batch_size = 1000
    if len(traces) > max_batch_size:
        raise ValueError(f"Batch size ({len(traces)}) exceeds maximum ({max_batch_size})")

    # Create all trace models
    db_traces = []
    for trace in traces:
        db_trace = Trace(
            trace_version=trace.trace_version,
            payload=trace.model_dump(),
        )
        db_traces.append(db_trace)

    # Insert all traces in a single transaction
    db.add_all(db_traces)
    await db.commit()

    # OPTIMIZATION: Bulk refresh via single SELECT query
    # Extract IDs from committed traces
    trace_ids = [db_trace.id for db_trace in db_traces]

    # Fetch all traces in one query
    result = await db.execute(select(Trace).where(Trace.id.in_(trace_ids)))
    refreshed_traces = result.scalars().all()

    # Create mapping for fast lookup
    trace_map: dict[uuid.UUID, Trace] = {t.id: t for t in refreshed_traces}

    # Build response using refreshed data
    created_traces = [
        TraceCreateResponse(
            id=db_trace.id,
            created_at=trace_map[db_trace.id].created_at,
            trace_version=trace_map[db_trace.id].trace_version,
        )
        for db_trace in db_traces
    ]

    return TraceBatchCreateResponse(
        created_count=len(created_traces),
        traces=created_traces,
    )
