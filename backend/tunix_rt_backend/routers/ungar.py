"""UNGAR integration endpoints.

Domain: Synthetic trace generation via UNGAR game engine

Primary endpoints:
- GET /api/ungar/status: Check UNGAR availability and version
- POST /api/ungar/generate: Generate traces from game simulations

Cross-cutting concerns:
- UNGAR is optional; endpoints degrade gracefully (503) when not installed
- Supports High Card Duel game for synthetic reasoning traces
- Generated traces include game metadata in payload
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.schemas import (
    UngarGenerateRequest,
    UngarGenerateResponse,
    UngarStatusResponse,
)

router = APIRouter()


@router.get("/api/ungar/status", response_model=UngarStatusResponse)
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


@router.post(
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


@router.get("/api/ungar/high-card-duel/export.jsonl")
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
