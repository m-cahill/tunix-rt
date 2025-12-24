"""Integration tests for Tuning Service."""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.schemas.tuning import TuningJobCreate
from tunix_rt_backend.services.tuning_service import TuningService, _convert_search_space


@pytest.mark.asyncio
async def test_create_and_get_job(test_db: AsyncSession) -> None:
    service = TuningService(test_db)

    job_in = TuningJobCreate(
        name="Test Job",
        dataset_key="test-v1",
        base_model_id="gemma-2b",
        search_space={"lr": {"type": "uniform", "min": 0.001, "max": 0.01}},
    )

    # Create
    job = await service.create_job(job_in)
    assert job.id is not None
    assert job.name == "Test Job"
    assert job.status == "created"

    # Get
    fetched = await service.get_job(job.id)
    assert fetched is not None
    assert fetched.id == job.id
    assert fetched.search_space_json["lr"]["type"] == "uniform"

    # List
    jobs = await service.list_jobs()
    assert len(jobs) >= 1
    assert any(j.id == job.id for j in jobs)


def test_convert_search_space() -> None:
    # Mock Ray availability and objects
    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", True):
        with patch("tunix_rt_backend.services.tuning_service.tune") as mock_tune:
            space = _convert_search_space(
                {
                    "lr": {"type": "loguniform", "min": 1e-4, "max": 1e-2},
                    "bs": {"type": "choice", "values": [8, 16]},
                }
            )

            mock_tune.loguniform.assert_called_with(1e-4, 1e-2)
            mock_tune.choice.assert_called_with([8, 16])
            assert "lr" in space
            assert "bs" in space


@pytest.mark.asyncio
async def test_start_job_not_found(test_db: AsyncSession) -> None:
    service = TuningService(test_db)
    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", True):
        with pytest.raises(ValueError, match="not found"):
            await service.start_job(uuid.uuid4())


@pytest.mark.asyncio
async def test_start_job_ray_not_installed(test_db: AsyncSession) -> None:
    service = TuningService(test_db)
    # Create valid job
    job = await service.create_job(
        TuningJobCreate(
            name="J",
            dataset_key="d",
            base_model_id="m",
            search_space={"x": {"type": "choice", "values": [1]}},
        )
    )

    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="Ray Tune is not available"):
            await service.start_job(job.id)
