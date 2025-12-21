"""Helper functions for trace-related operations."""

import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace


async def get_trace_or_404(
    db: AsyncSession,
    trace_id: uuid.UUID,
    label: str | None = None,
) -> Trace:
    """Fetch a trace by ID, raising 404 if not found.

    This helper centralizes the fetch-and-validate pattern used across
    multiple endpoints, ensuring consistent error messages and reducing
    code duplication.

    Args:
        db: Database session
        trace_id: UUID of the trace to fetch
        label: Optional label for error message context (e.g., "Base", "Other")

    Returns:
        Trace object if found

    Raises:
        HTTPException: 404 if trace not found

    Examples:
        >>> # Simple fetch
        >>> trace = await get_trace_or_404(db, trace_id)
        >>> # With context label for comparison
        >>> base = await get_trace_or_404(db, base_id, label="Base")
        >>> other = await get_trace_or_404(db, other_id, label="Other")
    """
    result = await db.execute(select(Trace).where(Trace.id == trace_id))
    trace = result.scalar_one_or_none()

    if trace is None:
        # Build error message with optional label
        if label:
            message = f"{label} trace {trace_id} not found"
        else:
            message = f"Trace {trace_id} not found"

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message,
        )

    return trace
