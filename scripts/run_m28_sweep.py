#!/usr/bin/env python3
"""M28 Tuning Sweep Script (legacy).

This script runs a hyperparameter tuning sweep using the M28 configuration.
For newer sweeps, use backend/tools/run_tune_m34.py which has updated defaults.

Usage:
    python scripts/run_m28_sweep.py

Requirements:
    - Backend must be running: uvicorn tunix_rt_backend.app:app --reload
    - Ray Tune must be installed: pip install "ray[tune]"
"""

import json
import logging
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from tunix_rt_backend.tuning import SweepConfig, SweepRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000"


def run_sweep() -> int:
    """Run M28 tuning sweep."""
    logger.info("Starting M28 Tuning Sweep...")

    # M28-specific configuration (legacy defaults)
    config = SweepConfig(
        name="M28 Tuning Sweep",
        dataset_key="golden-v2",
        base_model_id="google/gemma-2b-it",
        metric_name="answer_correctness",
        metric_mode="max",
        num_samples=3,
        max_concurrent_trials=1,
        search_space={
            "learning_rate": {
                "type": "loguniform",
                "min": 1e-5,
                "max": 1e-4,
            },
            "batch_size": {
                "type": "choice",
                "values": [2, 4],
            },
            "weight_decay": {
                "type": "uniform",
                "min": 0.0,
                "max": 0.1,
            },
        },
    )

    # Run sweep using shared runner
    with SweepRunner(api_url=API_URL) as runner:
        result = runner.run(config)

    if result.success:
        logger.info(f"Job finished with status: completed")
        logger.info(f"Best Params: {json.dumps(result.best_params, indent=2)}")
        logger.info(f"Best Run ID: {result.best_run_id}")
        return 0
    else:
        logger.error(f"Job failed: {result.error}")
        return 1


if __name__ == "__main__":
    sys.exit(run_sweep())
