"""M28 Tuning Sweep Verification Tests.

Verifies that the TuningService supports the M28 specific search space
(LR, batch_size, weight_decay) and that the trainable function correctly
propagates these parameters to the execution engine.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.schemas.tuning import TuningJobCreate
from tunix_rt_backend.services.tuning_service import (
    TuningService,
    _convert_search_space,
    tunix_trainable,
)


@pytest.mark.asyncio
async def test_m28_schema_support(test_db: AsyncSession) -> None:
    """Verify that M28 search space parameters are supported by the schema."""
    service = TuningService(test_db)

    # M28: LR (loguniform), Batch Size (choice), Weight Decay (uniform)
    m28_search_space = {
        "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-3},
        "batch_size": {"type": "choice", "values": [4, 8, 16]},
        "weight_decay": {"type": "uniform", "min": 0.0, "max": 0.1},
        "warmup_steps": {"type": "randint", "min": 0, "max": 100},
    }

    job_in = TuningJobCreate(
        name="M28 Sweep",
        dataset_key="golden-v2",
        base_model_id="google/gemma-2b",
        metric_name="answer_correctness",
        search_space=m28_search_space,
        num_samples=3,
    )

    # Should create without error
    job = await service.create_job(job_in)
    assert job.id is not None
    assert job.search_space_json == m28_search_space


def test_m28_ray_conversion() -> None:
    """Verify conversion of M28 parameters to Ray Tune objects."""
    m28_search_space = {
        "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-3},
        "batch_size": {"type": "choice", "values": [4, 8, 16]},
        "weight_decay": {"type": "uniform", "min": 0.0, "max": 0.1},
    }

    with patch("tunix_rt_backend.services.tuning_service.RAY_AVAILABLE", True):
        with patch("tunix_rt_backend.services.tuning_service.tune") as mock_tune:
            space = _convert_search_space(m28_search_space)

            # Verify calls
            mock_tune.loguniform.assert_called_with(1e-5, 1e-3)
            mock_tune.choice.assert_called_with([4, 8, 16])
            mock_tune.uniform.assert_called_with(0.0, 0.1)

            assert "learning_rate" in space
            assert "batch_size" in space
            assert "weight_decay" in space


def test_tunix_trainable_propagates_params() -> None:
    """Verify that the trainable function passes hyperparameters to execute_tunix_run."""

    # Mock config that Ray would pass
    config = {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "weight_decay": 0.01,
        # Default override
        "num_epochs": 3,
    }

    tunix_job_id = str(uuid.uuid4())
    tunix_dataset_key = "golden-v2"
    tunix_model_id = "google/gemma-2b"
    tunix_metric_name = "answer_correctness"

    # Mock DB Session
    mock_db = AsyncMock()

    # Mock dependencies
    # Patch the SOURCE of async_session_maker because it is imported locally in tunix_trainable
    with patch("tunix_rt_backend.db.base.async_session_maker") as mock_maker:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_db
        mock_ctx.__aexit__.return_value = None
        mock_maker.return_value = mock_ctx

        with (
            patch("tunix_rt_backend.services.tuning_service.train") as mock_train,
            patch("tunix_rt_backend.services.tuning_service.tune") as mock_tune,
        ):
            # Mock tune.get_trial_id
            mock_tune.get_trial_id.return_value = "trial_123"
            # Also mock get_context just in case
            mock_context = MagicMock()
            mock_context.get_trial_id.return_value = "trial_123"
            mock_tune.get_context.return_value = mock_context
            # train context might not be used anymore but keeping it safe
            mock_train.get_context.return_value = mock_context

            # Patch where it is imported FROM, since it is a local import in the
            # function. Patching the source module should work if we patch
            # before the function runs.
            with patch(
                "tunix_rt_backend.services.tunix_execution.execute_tunix_run",
                new_callable=AsyncMock,
            ) as mock_execute:
                # Setup mock response
                mock_response = MagicMock()
                mock_response.status = "completed"
                mock_response.run_id = str(uuid.uuid4())
                mock_execute.return_value = mock_response

                with patch(
                    "tunix_rt_backend.services.evaluation.EvaluationService"
                ) as MockEvalService:
                    mock_eval_svc = MockEvalService.return_value
                    mock_eval_res = MagicMock()
                    mock_eval_res.score = 0.85
                    mock_eval_res.metrics = {"answer_correctness": 0.85}
                    mock_eval_svc.evaluate_run = AsyncMock(return_value=mock_eval_res)

                    # Run trainable directly
                    # It calls asyncio.run(_execute_trial())
                    # Since we are in a sync test (no loop running), asyncio.run should work fine
                    # and create a temporary loop to run the coroutine.
                    tunix_trainable(
                        config=config,
                        tunix_job_id=tunix_job_id,
                        tunix_dataset_key=tunix_dataset_key,
                        tunix_model_id=tunix_model_id,
                        tunix_metric_name=tunix_metric_name,
                    )

                    # Verify arguments passed to execute_tunix_run
                    assert mock_execute.called
                    call_args = mock_execute.call_args
                    run_req = call_args[0][0]  # First arg is request

                    # Check that params from config override defaults
                    assert run_req.dataset_key == tunix_dataset_key
                    assert run_req.model_id == tunix_model_id
                    assert run_req.learning_rate == 5e-5
                    assert run_req.batch_size == 16
                    assert run_req.weight_decay == 0.01
                    assert run_req.num_epochs == 3
