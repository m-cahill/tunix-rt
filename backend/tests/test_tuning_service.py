"""Integration tests for Tuning Service."""

import uuid
from unittest.mock import MagicMock, patch

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


def test_convert_search_space_invalid_type() -> None:
    """Test validation of unknown parameter types."""
    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", True):
        with pytest.raises(ValueError, match="Unknown param type"):
            _convert_search_space({"invalid": {"type": "magic_hyperparam", "min": 1, "max": 10}})


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


@pytest.mark.asyncio
async def test_start_job_wrong_status(test_db: AsyncSession) -> None:
    """Test starting a job that is not in created/failed state."""
    service = TuningService(test_db)
    job = await service.create_job(
        TuningJobCreate(
            name="Running Job",
            dataset_key="d",
            base_model_id="m",
            search_space={"x": {"type": "choice", "values": [1]}},
        )
    )

    # Manually set status to running
    job.status = "running"
    await test_db.commit()

    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", True):
        with pytest.raises(ValueError, match="cannot start"):
            await service.start_job(job.id)


@pytest.mark.asyncio
async def test_run_ray_tune_failure_handling(test_db: AsyncSession) -> None:
    """Test that job status is updated to failed if Ray Tune throws exception."""
    service = TuningService(test_db)
    job = await service.create_job(
        TuningJobCreate(
            name="Failing Job",
            dataset_key="d",
            base_model_id="m",
            search_space={"x": {"type": "choice", "values": [1]}},
        )
    )

    # Mock Ray interactions
    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", True):
        with patch("tunix_rt_backend.services.tuning_service.ray", create=True) as mock_ray:
            mock_ray.is_initialized.return_value = True

            with patch("tunix_rt_backend.services.tuning_service.tune") as mock_tune:
                # Mock Tuner to raise exception
                mock_tuner_instance = MagicMock()
                mock_tuner_instance.fit.side_effect = RuntimeError("Ray crashed")
                mock_tune.Tuner.return_value = mock_tuner_instance

                # Mock async_session_maker to return test_db
                # We use a mock context manager that yields test_db but prevents closing it
                mock_ctx = MagicMock()
                mock_ctx.__aenter__.return_value = test_db
                mock_ctx.__aexit__.return_value = None

                with patch(
                    "tunix_rt_backend.services.tuning_service.async_session_maker",
                    return_value=mock_ctx,
                ):
                    # Call internal method directly since start_job spawns a task
                    # We want to await it to ensure DB updates happen
                    await service._run_ray_tune(job.id)

                # Refresh job from DB
                await test_db.refresh(job)
                assert job.status == "failed"
