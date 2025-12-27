"""Evaluation endpoints (M17, M35).

Domain: Model evaluation, scoring, and leaderboard

Primary endpoints:
- POST /api/evaluation/evaluate: Trigger evaluation for a run
- GET /api/evaluation/leaderboard: Paginated + filtered leaderboard by score

M35 additions:
- Leaderboard filtering by dataset, model_id, config, date range
- Scorecard summary inline with leaderboard items

Cross-cutting concerns:
- Pluggable judge interface (MockJudge, GemmaJudge, AnswerCorrectnessJudge)
- Evaluations stored in tunix_run_evaluations table
- Leaderboard excludes dry-run entries
- Pagination with limit/offset
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.dependencies import get_redi_client
from tunix_rt_backend.redi_client import RediClientProtocol
from tunix_rt_backend.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    LeaderboardFilters,
    LeaderboardResponse,
)

router = APIRouter()


@router.post(
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


@router.get("/api/tunix/runs/{run_id}/evaluation", response_model=EvaluationResponse)
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


@router.get("/api/tunix/evaluations", response_model=LeaderboardResponse)
async def get_leaderboard(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=100, description="Results per page"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
    # M35: Filter parameters (all optional, AND logic)
    dataset_key: str | None = Query(default=None, description="Filter by dataset (exact)"),
    model_id: str | None = Query(default=None, description="Filter by model (contains)"),
    config_path: str | None = Query(default=None, description="Filter by config (contains)"),
    date_from: datetime | None = Query(default=None, description="Filter by date (>=)"),
    date_to: datetime | None = Query(default=None, description="Filter by date (<=)"),
) -> LeaderboardResponse:
    """Get leaderboard data with optional filtering (M17, M35).

    Supports filtering by dataset, model, config, and date range.
    All filters use AND logic. Empty values are ignored.

    Returns:
        LeaderboardResponse with sorted evaluation results and applied filters
    """
    from tunix_rt_backend.services.evaluation import EvaluationService

    # Build filters object
    filters = LeaderboardFilters(
        dataset_key=dataset_key,
        model_id=model_id,
        config_path=config_path,
        date_from=date_from,
        date_to=date_to,
    )

    service = EvaluationService(db)
    return await service.get_leaderboard(limit=limit, offset=offset, filters=filters)
