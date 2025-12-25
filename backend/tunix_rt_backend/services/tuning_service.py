"""Tuning service (M19).

Handles creation and execution of Ray Tune experiments.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import async_session_maker
from tunix_rt_backend.db.models import TunixTuningJob, TunixTuningTrial
from tunix_rt_backend.schemas.tuning import TuningJobCreate

logger = logging.getLogger(__name__)

# Check for Ray availability
try:
    import ray  # type: ignore
    from ray import train, tune

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    train = None
    tune = None
    logger.warning("Ray Tune not installed. Tuning features will be disabled.")

# M23: Guardrail - only allow trusted metrics for tuning
LOCKED_METRICS = {"answer_correctness"}


def _convert_search_space(search_space_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON schema to Ray Tune search space objects."""
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray Tune not available")

    space = {}
    for name, param in search_space_json.items():
        p_type = param.get("type")
        if p_type == "choice":
            space[name] = tune.choice(param["values"])
        elif p_type == "uniform":
            space[name] = tune.uniform(param["min"], param["max"])
        elif p_type == "loguniform":
            space[name] = tune.loguniform(param["min"], param["max"])
        elif p_type == "randint":
            space[name] = tune.randint(param["min"], param["max"])
        else:
            raise ValueError(f"Unknown param type: {p_type}")
    return space


def tunix_trainable(
    config: Dict[str, Any],
    tunix_job_id: str = "",
    tunix_dataset_key: str = "",
    tunix_model_id: str = "",
    tunix_metric_name: str = "",
) -> None:
    """Ray Tune trainable function.

    Executes a single trial:
    1. Initializes DB session
    2. Runs Tunix training (TunixRun)
    3. Evaluates the run
    4. Reports metrics to Ray
    """
    # Import here to ensure worker process has them
    import asyncio  # noqa: F401

    from tunix_rt_backend.db.base import async_session_maker
    from tunix_rt_backend.schemas import TunixRunRequest
    from tunix_rt_backend.services.evaluation import EvaluationService
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    # Remaining config items are hyperparameters
    hyperparameters = config

    async def _execute_trial() -> None:
        async with async_session_maker() as db:
            # Create Trial Record
            # M28 Fix: Use tune.get_context() or tune.get_trial_id()
            try:
                trial_id = tune.get_trial_id()
            except Exception:
                # Fallback or older ray versions
                # In Ray 2.9+, tune.get_trial_id() works
                # If not, try tune.get_context().get_trial_id()
                try:
                    trial_id = tune.get_context().get_trial_id()
                except Exception:
                    trial_id = "unknown_trial"

            job_id = uuid.UUID(tunix_job_id)

            # Create or update trial record
            # Note: Ray might restart trials, so we check if exists or just log
            # For simplicity, we create a record here.

            # Construct TunixRunRequest
            run_req_data = {
                "dataset_key": tunix_dataset_key,
                "model_id": tunix_model_id,
                "dry_run": False,  # Always real execution for tuning
                # Defaults
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "batch_size": 8,
                "max_seq_length": 2048,
            }

            # Override with hyperparameters
            run_req_data.update(hyperparameters)

            try:
                run_request = TunixRunRequest(**run_req_data)  # type: ignore[arg-type]
            except Exception as e:
                # Invalid params
                tune.report({tunix_metric_name: 0.0, "error": str(e), "done": True})
                return

            # Execute Run
            try:
                # We use execute_tunix_run directly (sync wait)
                response = await execute_tunix_run(run_request, db, async_mode=False)

                if response.status != "completed":
                    tune.report(
                        {tunix_metric_name: 0.0, "done": True, "run_status": response.status}
                    )
                    return

                # Evaluate Run
                eval_service = EvaluationService(db)
                # We need the run_id from response
                run_uuid = uuid.UUID(response.run_id)

                # M22: Select judge based on metric name
                judge_override = None
                if tunix_metric_name == "answer_correctness":
                    judge_override = "answer_correctness"

                eval_response = await eval_service.evaluate_run(
                    run_uuid, judge_override=judge_override
                )

                # Extract metric
                # metric_name typically "score" or something from metrics dict
                score = eval_response.score  # Default primary score

                # Check if user asked for a specific metric from details
                if tunix_metric_name != "score" and tunix_metric_name in eval_response.metrics:
                    score = eval_response.metrics[tunix_metric_name]

                # Log trial info to DB
                # Ideally we want to link the trial_id to this run
                # We can try to insert TunixTuningTrial here
                trial = TunixTuningTrial(
                    id=trial_id,
                    tuning_job_id=job_id,
                    run_id=run_uuid,
                    params_json=hyperparameters,
                    metric_value=score,
                    status="completed",
                    created_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                db.add(trial)
                await db.commit()

                # Report to Ray
                tune.report({tunix_metric_name: score, "run_id": response.run_id, "done": True})

            except Exception as e:
                logger.error(f"Trial execution failed: {e}")
                tune.report({tunix_metric_name: 0.0, "error": str(e), "done": True})

                # Log failure to DB
                trial = TunixTuningTrial(
                    id=trial_id,
                    tuning_job_id=job_id,
                    run_id=None,
                    params_json=hyperparameters,
                    metric_value=None,
                    status="failed",
                    error=str(e),
                    created_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                db.add(trial)
                await db.commit()

    # Run async function in sync wrapper
    asyncio.run(_execute_trial())


class TuningService:
    """Service for managing tuning jobs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(self, request: TuningJobCreate) -> TunixTuningJob:
        """Create a new tuning job record."""
        # Dump Pydantic model to dict for storage
        search_space = {k: v.model_dump() for k, v in request.search_space.items()}

        job = TunixTuningJob(
            name=request.name,
            dataset_key=request.dataset_key,
            base_model_id=request.base_model_id,
            metric_name=request.metric_name,
            metric_mode=request.metric_mode,
            num_samples=request.num_samples,
            max_concurrent_trials=request.max_concurrent_trials,
            search_space_json=search_space,
            created_at=datetime.now(timezone.utc),
            status="created",
        )
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def get_job(self, job_id: uuid.UUID) -> TunixTuningJob | None:
        """Get a tuning job by ID."""
        stmt = select(TunixTuningJob).where(TunixTuningJob.id == job_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_jobs(self, limit: int = 20, offset: int = 0) -> list[TunixTuningJob]:
        """List tuning jobs."""
        stmt = (
            select(TunixTuningJob)
            .order_by(TunixTuningJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def start_job(self, job_id: uuid.UUID) -> None:
        """Start a tuning job (runs Ray Tune).

        Note: This is a blocking call in the current design (runs in thread/process?).
        Ideally this should run in a background task (FastAPI BackgroundTasks).
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray Tune is not available")

        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status not in ["created", "failed"]:
            raise ValueError(f"Job is in {job.status} state, cannot start")

        # Guardrail (M23): Check locked metrics BEFORE starting
        if job.metric_name not in LOCKED_METRICS:
            raise ValueError(f"Only locked metrics allowed: {', '.join(sorted(LOCKED_METRICS))}")

        # Update status
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await self.db.commit()

        asyncio.create_task(self._run_ray_tune(job.id))

    async def _run_ray_tune(self, job_id: uuid.UUID) -> None:
        """Internal method to run Ray Tune and update job status."""
        async with async_session_maker() as db:
            job = await db.get(TunixTuningJob, job_id)
            if not job:
                return

            try:
                # Setup Ray Search Space
                search_space = _convert_search_space(job.search_space_json)

                # Inject constants
                # We use tune.with_parameters to pass fixed config
                trainable_with_params = tune.with_parameters(
                    tunix_trainable,
                    tunix_job_id=str(job.id),
                    tunix_dataset_key=job.dataset_key,
                    tunix_model_id=job.base_model_id,
                    tunix_metric_name=job.metric_name,
                )

                # Configure Tuner
                import os

                storage_root = os.path.abspath("./artifacts/tuning")

                tuner = tune.Tuner(
                    trainable_with_params,
                    param_space=search_space,
                    tune_config=tune.TuneConfig(
                        mode=job.metric_mode,
                        metric=job.metric_name,
                        num_samples=job.num_samples,
                        max_concurrent_trials=job.max_concurrent_trials,
                    ),
                    run_config=tune.RunConfig(
                        name=f"tunix_job_{job.id}",
                        storage_path=storage_root,
                    ),
                )

                def run_tuner_sync() -> Any:
                    if not ray.is_initialized():
                        ray.init(ignore_reinit_error=True)
                    return tuner.fit()

                result_grid = await asyncio.to_thread(run_tuner_sync)

                # Process Results
                best_result = result_grid.get_best_result(
                    metric=job.metric_name, mode=job.metric_mode
                )

                # Update Job
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.best_params_json = best_result.config
                job.ray_storage_path = best_result.path

                # Link best run
                # best_result.metrics contains "run_id" if we reported it
                if best_result.metrics and "run_id" in best_result.metrics:
                    job.best_run_id = uuid.UUID(best_result.metrics["run_id"])

                await db.commit()

            except Exception as e:
                import traceback

                logger.error(f"Tuning job {job_id} failed: {e}\n{traceback.format_exc()}")
                job.status = "failed"
                await db.commit()
