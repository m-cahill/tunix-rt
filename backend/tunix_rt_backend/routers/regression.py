"""Regression testing endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.schemas.regression import (
    RegressionBaselineCreate,
    RegressionBaselineResponse,
    RegressionCheckRequest,
    RegressionCheckResult,
)

router = APIRouter()


@router.post(
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


@router.post(
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
