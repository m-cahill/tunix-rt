"""Sweep runner for hyperparameter tuning (M34).

This module provides a reusable SweepRunner class for creating and monitoring
Ray Tune tuning jobs via the Tunix RT API. It abstracts the common workflow:
1. Create a tuning job with search space
2. Start the job
3. Poll for completion
4. Return best params and results

Usage:
    from tunix_rt_backend.tuning import SweepRunner, SweepConfig

    config = SweepConfig(
        name="M34 Optimization Sweep",
        dataset_key="dev-reasoning-v2",
        num_samples=10,
    )
    runner = SweepRunner(api_url="http://127.0.0.1:8000")
    result = runner.run(config)
    if result.success:
        print(f"Best params: {result.best_params}")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


@dataclass
class SweepConfig:
    """Configuration for a tuning sweep.

    Attributes:
        name: Human-readable name for the experiment
        dataset_key: Dataset to train on (e.g., "dev-reasoning-v2")
        base_model_id: HuggingFace model ID (default: gemma-3-1b-it)
        metric_name: Metric to optimize (must be in LOCKED_METRICS)
        metric_mode: "max" or "min"
        num_samples: Number of trials to run
        max_concurrent_trials: Parallelism limit
        search_space: Ray Tune search space dict
        poll_interval: Seconds between status polls
        timeout_seconds: Max wait time (0 = infinite)
    """

    name: str = "Tuning Sweep"
    dataset_key: str = "dev-reasoning-v2"
    base_model_id: str = "google/gemma-3-1b-it"
    metric_name: str = "answer_correctness"
    metric_mode: str = "max"
    num_samples: int = 5
    max_concurrent_trials: int = 1
    poll_interval: float = 5.0
    timeout_seconds: float = 0  # 0 = no timeout

    # Search space with M34 defaults
    # Per M34 answers: learning_rate, per_device_batch_size, weight_decay, warmup_steps
    search_space: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": {
                "type": "loguniform",
                "min": 1e-5,
                "max": 1e-4,
            },
            "per_device_batch_size": {
                "type": "choice",
                "values": [1, 2, 4],
            },
            "weight_decay": {
                "type": "uniform",
                "min": 0.0,
                "max": 0.1,
            },
            "warmup_steps": {
                "type": "choice",
                "values": [0, 10, 20],
            },
        }
    )

    def to_api_payload(self) -> dict[str, Any]:
        """Convert config to API payload format."""
        return {
            "name": self.name,
            "dataset_key": self.dataset_key,
            "base_model_id": self.base_model_id,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
            "num_samples": self.num_samples,
            "max_concurrent_trials": self.max_concurrent_trials,
            "search_space": self.search_space,
        }


@dataclass
class SweepResult:
    """Result of a tuning sweep.

    Attributes:
        success: Whether the sweep completed successfully
        job_id: UUID of the tuning job
        status: Final job status
        best_params: Best hyperparameters found (if successful)
        best_run_id: UUID of the best run (if successful)
        error: Error message (if failed)
        trials: List of trial results (if available)
    """

    success: bool
    job_id: str | None = None
    status: str | None = None
    best_params: dict[str, Any] | None = None
    best_run_id: str | None = None
    error: str | None = None
    trials: list[dict[str, Any]] | None = None


# ============================================================
# Sweep Runner
# ============================================================


class SweepRunner:
    """Runs tuning sweeps via the Tunix RT API.

    This class encapsulates the workflow of creating, starting, and monitoring
    a tuning job. It communicates with the backend via HTTP API calls.

    Example:
        runner = SweepRunner(api_url="http://localhost:8000")
        config = SweepConfig(name="My Sweep", num_samples=10)
        result = runner.run(config)
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
        """Initialize the sweep runner.

        Args:
            api_url: Base URL of the Tunix RT API
        """
        self.api_url = api_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def run(self, config: SweepConfig) -> SweepResult:
        """Run a tuning sweep with the given configuration.

        This method:
        1. Creates a tuning job via POST /api/tuning/jobs
        2. Starts the job via POST /api/tuning/jobs/{id}/start
        3. Polls for completion via GET /api/tuning/jobs
        4. Returns the best parameters on success

        Args:
            config: SweepConfig with sweep parameters

        Returns:
            SweepResult with success status and best params
        """
        logger.info(f"Starting sweep: {config.name}")

        # Step 1: Create Job
        try:
            job_id = self._create_job(config)
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return SweepResult(success=False, error=str(e))

        # Step 2: Start Job
        try:
            self._start_job(job_id)
        except Exception as e:
            logger.error(f"Failed to start job: {e}")
            return SweepResult(success=False, job_id=job_id, error=str(e))

        # Step 3: Poll for Completion
        try:
            final_job = self._poll_until_done(job_id, config)
        except Exception as e:
            logger.error(f"Polling failed: {e}")
            return SweepResult(success=False, job_id=job_id, error=str(e))

        # Step 4: Build Result
        if final_job["status"] == "completed":
            return SweepResult(
                success=True,
                job_id=job_id,
                status="completed",
                best_params=final_job.get("best_params_json"),
                best_run_id=final_job.get("best_run_id"),
                trials=final_job.get("trials"),
            )
        else:
            return SweepResult(
                success=False,
                job_id=job_id,
                status=final_job["status"],
                error=f"Job ended with status: {final_job['status']}",
            )

    def _create_job(self, config: SweepConfig) -> str:
        """Create a tuning job and return its ID."""
        payload = config.to_api_payload()
        logger.info(f"Creating job: {json.dumps(payload, indent=2)}")

        resp = self.client.post(f"{self.api_url}/api/tuning/jobs", json=payload)
        resp.raise_for_status()

        job = resp.json()
        job_id: str = job["id"]
        logger.info(f"Job created: {job_id}")
        return job_id

    def _start_job(self, job_id: str) -> None:
        """Start a tuning job."""
        logger.info(f"Starting job {job_id}...")
        resp = self.client.post(f"{self.api_url}/api/tuning/jobs/{job_id}/start")

        if resp.status_code == 501:
            raise RuntimeError("Ray Tune not installed or not available on backend")

        resp.raise_for_status()
        logger.info("Job started successfully")

    def _poll_until_done(self, job_id: str, config: SweepConfig) -> dict[str, Any]:
        """Poll job status until completion or timeout."""
        logger.info("Polling for completion...")
        start_time = time.time()

        while True:
            # Check timeout
            if config.timeout_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed > config.timeout_seconds:
                    raise TimeoutError(f"Sweep timed out after {config.timeout_seconds}s")

            # Get job status
            resp = self.client.get(f"{self.api_url}/api/tuning/jobs")
            resp.raise_for_status()
            jobs = resp.json()

            # Find our job
            current_job: dict[str, Any] | None = next((j for j in jobs if j["id"] == job_id), None)
            if not current_job:
                raise RuntimeError(f"Job {job_id} not found in job list")

            status = current_job["status"]
            logger.info(f"Job status: {status}")

            if status in ["completed", "failed"]:
                return current_job

            time.sleep(config.poll_interval)

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "SweepRunner":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
