"""UNGAR trace generation service.

This module provides business logic for UNGAR integration endpoints:
- Checking UNGAR availability
- Generating High Card Duel traces
- Exporting traces to JSONL format

The service handles optional UNGAR dependency gracefully via lazy imports.
"""

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas import ReasoningTrace, UngarGenerateRequest, UngarStatusResponse


def check_ungar_status() -> UngarStatusResponse:
    """Check UNGAR availability and version.
    
    Returns:
        UngarStatusResponse with availability status and version (if available)
    
    Note:
        This function uses lazy imports to avoid errors when UNGAR is not installed.
    """
    from tunix_rt_backend.integrations.ungar.availability import (
        ungar_available,
        ungar_version,
    )

    available = ungar_available()
    version = ungar_version() if available else None

    return UngarStatusResponse(available=available, version=version)


async def generate_high_card_duel_traces(
    request: UngarGenerateRequest,
    db: AsyncSession,
) -> tuple[list[uuid.UUID], list[dict[str, Any]]]:
    """Generate High Card Duel traces from UNGAR episodes.
    
    Args:
        request: Generation parameters (count, seed, persist)
        db: Database session for persisting traces
    
    Returns:
        Tuple of (trace_ids, preview) where:
        - trace_ids: List of created trace UUIDs
        - preview: List of preview dictionaries for first 3 traces
    
    Raises:
        ValueError: If UNGAR is not installed
    """
    from tunix_rt_backend.integrations.ungar.availability import ungar_available

    # Check if UNGAR is available
    if not ungar_available():
        raise ValueError("UNGAR is not installed. Install with: pip install -e '.[ungar]'")

    # Import generator (lazy to avoid import errors when UNGAR not installed)
    from tunix_rt_backend.integrations.ungar.high_card_duel import (
        generate_high_card_duel_traces as ungar_generate,
    )

    # Generate traces
    traces = ungar_generate(count=request.count, seed=request.seed)

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

    return trace_ids, preview


async def export_high_card_duel_jsonl(
    db: AsyncSession,
    limit: int = 100,
    trace_ids_str: str | None = None,
) -> str:
    """Export High Card Duel traces in JSONL format (Tunix-friendly).
    
    Args:
        db: Database session
        limit: Maximum number of traces to export (default 100)
        trace_ids_str: Comma-separated list of trace IDs to export (optional)
    
    Returns:
        JSONL string with one trace per line
    
    Note:
        If trace_ids_str is provided, only those traces are exported.
        Otherwise, exports the most recent traces with source="ungar" up to limit.
    """
    # Build query
    if trace_ids_str:
        # Parse comma-separated UUIDs
        ids = [uuid.UUID(tid.strip()) for tid in trace_ids_str.split(",")]
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

    # Return as NDJSON string
    return "\n".join(lines) + "\n" if lines else ""

