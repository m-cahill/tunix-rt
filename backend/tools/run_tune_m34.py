#!/usr/bin/env python3
"""M34 Optimization Sweep Script.

This script runs a hyperparameter tuning sweep using the M34 configuration:
- Dataset: dev-reasoning-v2 (550 traces)
- Model: google/gemma-3-1b-it
- Metric: answer_correctness (maximize)
- Search space: learning_rate, per_device_batch_size, weight_decay, warmup_steps

Usage:
    # From backend directory
    python tools/run_tune_m34.py

    # With custom options
    python tools/run_tune_m34.py --num-samples 10 --dataset dev-reasoning-v2

Requirements:
    - Backend must be running: uvicorn tunix_rt_backend.app:app --reload
    - Ray Tune must be installed: pip install "ray[tune]"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tunix_rt_backend.tuning import SweepConfig, SweepRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Run M34 optimization sweep."""
    parser = argparse.ArgumentParser(description="Run M34 hyperparameter tuning sweep")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Tunix RT API URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of trials to run (default: 5)",
    )
    parser.add_argument(
        "--dataset",
        default="dev-reasoning-v2",
        help="Dataset key (default: dev-reasoning-v2)",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help="Base model ID (default: google/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Timeout in seconds (0 = no timeout, default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for best params JSON (optional)",
    )

    args = parser.parse_args()

    # Create M34-specific config
    config = SweepConfig(
        name="M34 Optimization Sweep",
        dataset_key=args.dataset,
        base_model_id=args.model,
        metric_name="answer_correctness",
        metric_mode="max",
        num_samples=args.num_samples,
        max_concurrent_trials=1,
        timeout_seconds=args.timeout,
        # M34 search space per M34_answers.md
        search_space={
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
        },
    )

    logger.info("=" * 60)
    logger.info("M34 Optimization Sweep")
    logger.info("=" * 60)
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Dataset: {config.dataset_key}")
    logger.info(f"Model: {config.base_model_id}")
    logger.info(f"Num Samples: {config.num_samples}")
    logger.info(f"Search Space: {json.dumps(config.search_space, indent=2)}")
    logger.info("=" * 60)

    # Run sweep
    with SweepRunner(api_url=args.api_url) as runner:
        result = runner.run(config)

    # Report results
    if result.success:
        logger.info("=" * 60)
        logger.info("SWEEP COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Job ID: {result.job_id}")
        logger.info(f"Best Run ID: {result.best_run_id}")
        logger.info(f"Best Params:\n{json.dumps(result.best_params, indent=2)}")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_data = {
                "job_id": result.job_id,
                "best_run_id": result.best_run_id,
                "best_params": result.best_params,
                "config": config.to_api_payload(),
            }
            output_path.write_text(json.dumps(output_data, indent=2))
            logger.info(f"Results saved to: {output_path}")

        return 0
    else:
        logger.error("=" * 60)
        logger.error("SWEEP FAILED")
        logger.error("=" * 60)
        logger.error(f"Job ID: {result.job_id}")
        logger.error(f"Status: {result.status}")
        logger.error(f"Error: {result.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
