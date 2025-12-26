"""Evaluation endpoints.

Domain: Model evaluation, scoring, and leaderboard

Primary endpoints:
- POST /api/evaluation/evaluate: Trigger evaluation for a run
- GET /api/evaluation/leaderboard: Paginated leaderboard by score

Cross-cutting concerns:
- Pluggable judge interface (MockJudge, GemmaJudge, AnswerCorrectnessJudge)
- Evaluations stored in tunix_run_evaluations table
- Leaderboard excludes dry-run entries
- Pagination with limit/offset
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.dependencies import get_redi_client
from tunix_rt_backend.redi_client import RediClientProtocol
from tunix_rt_backend.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
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
