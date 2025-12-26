"""Hyperparameter tuning endpoints.

Domain: Hyperparameter optimization via Ray Tune

Primary endpoints:
- POST /api/tuning/jobs: Create tuning job with search space
- GET /api/tuning/jobs: List all tuning jobs
- GET /api/tuning/jobs/{id}: Job details with trials
- POST /api/tuning/jobs/{id}/start: Launch optimization sweep
- GET /api/tuning/jobs/{id}/trials: List trials with metrics
- POST /api/tuning/jobs/{id}/promote-best: Promote best trial to registry

Cross-cutting concerns:
- Requires Ray Tune (optional dependency)
- Search space validation (numeric ranges, choices)
- Trials linked to TunixRun records for persistence
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.db.models import TunixTuningTrial
from tunix_rt_backend.schemas.tuning import (
    TuningJobCreate,
    TuningJobRead,
    TuningJobStartResponse,
    TuningTrialRead,
)

router = APIRouter()


@router.post(
    "/api/tuning/jobs",
    response_model=TuningJobRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_tuning_job(
    request: TuningJobCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TuningJobRead:
    """Create a new hyperparameter tuning job."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    job = await service.create_job(request)

    # Convert DB model to Read schema
    return TuningJobRead(
        id=job.id,
        name=job.name,
        status=job.status,
        dataset_key=job.dataset_key,
        base_model_id=job.base_model_id,
        mode=job.mode,
        metric_name=job.metric_name,
        metric_mode=job.metric_mode,
        num_samples=job.num_samples,
        max_concurrent_trials=job.max_concurrent_trials,
        search_space_json=job.search_space_json,
        best_run_id=job.best_run_id,
        best_params_json=job.best_params_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post(
    "/api/tuning/jobs/{job_id}/start",
    response_model=TuningJobStartResponse,
    status_code=status.HTTP_200_OK,
)
async def start_tuning_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TuningJobStartResponse:
    """Start a tuning job."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    try:
        await service.start_job(job_id)
    except RuntimeError as e:
        # Ray not installed
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )
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

    return TuningJobStartResponse(
        job_id=job_id,
        status="running",
        message="Tuning job started",
    )


@router.get("/api/tuning/jobs", response_model=list[TuningJobRead])
async def list_tuning_jobs(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> list[TuningJobRead]:
    """List tuning jobs."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    jobs = await service.list_jobs(limit, offset)

    return [
        TuningJobRead(
            id=job.id,
            name=job.name,
            status=job.status,
            dataset_key=job.dataset_key,
            base_model_id=job.base_model_id,
            mode=job.mode,
            metric_name=job.metric_name,
            metric_mode=job.metric_mode,
            num_samples=job.num_samples,
            max_concurrent_trials=job.max_concurrent_trials,
            search_space_json=job.search_space_json,
            best_run_id=job.best_run_id,
            best_params_json=job.best_params_json,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@router.get("/api/tuning/jobs/{job_id}", response_model=TuningJobRead)
async def get_tuning_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TuningJobRead:
    """Get tuning job details."""
    from tunix_rt_backend.services.tuning_service import TuningService

    service = TuningService(db)
    job = await service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Fetch trials
    trials_result = await db.execute(
        select(TunixTuningTrial)
        .where(TunixTuningTrial.tuning_job_id == job_id)
        .order_by(TunixTuningTrial.created_at)
    )
    trials = trials_result.scalars().all()

    trial_reads = [
        TuningTrialRead(
            id=t.id,
            tuning_job_id=t.tuning_job_id,
            run_id=t.run_id,
            params_json=t.params_json,
            metric_value=t.metric_value,
            status=t.status,
            error=t.error,
            created_at=t.created_at,
            completed_at=t.completed_at,
        )
        for t in trials
    ]

    return TuningJobRead(
        id=job.id,
        name=job.name,
        status=job.status,
        dataset_key=job.dataset_key,
        base_model_id=job.base_model_id,
        mode=job.mode,
        metric_name=job.metric_name,
        metric_mode=job.metric_mode,
        num_samples=job.num_samples,
        max_concurrent_trials=job.max_concurrent_trials,
        search_space_json=job.search_space_json,
        best_run_id=job.best_run_id,
        best_params_json=job.best_params_json,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        trials=trial_reads,
    )
