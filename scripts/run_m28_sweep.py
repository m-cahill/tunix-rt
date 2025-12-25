import httpx
import time
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000"

def check_dataset():
    """Ensure golden-v2 dataset exists."""
    logger.info("Checking for golden-v2 dataset...")
    try:
        # We can check via API if needed, or assume backend handles it if we seed it
        # Actually, let's use the API to list datasets
        # But for now, we'll assume the user ensures it or we run the seed tool
        pass
    except Exception as e:
        logger.warning(f"Failed to check dataset: {e}")

def run_sweep():
    logger.info("Starting M28 Tuning Sweep...")

    # 1. Create Job
    payload = {
        "name": "M28 Tuning Sweep",
        "dataset_key": "golden-v2",
        "base_model_id": "google/gemma-2b-it",
        "metric_name": "answer_correctness",
        "metric_mode": "max",
        "num_samples": 3,
        "max_concurrent_trials": 1,
        "search_space": {
            "learning_rate": {
                "type": "loguniform",
                "min": 1e-5,
                "max": 1e-4
            },
            "batch_size": {
                "type": "choice",
                "values": [2, 4]
            },
            "weight_decay": {
                "type": "uniform",
                "min": 0.0,
                "max": 0.1
            }
        }
    }

    try:
        logger.info(f"Creating job with payload: {json.dumps(payload, indent=2)}")
        resp = httpx.post(f"{API_URL}/api/tuning/jobs", json=payload, timeout=30.0)
        resp.raise_for_status()
        job = resp.json()
        job_id = job["id"]
        logger.info(f"Job created: {job_id}")

        # 2. Start Job
        logger.info(f"Starting job {job_id}...")
        resp = httpx.post(f"{API_URL}/api/tuning/jobs/{job_id}/start", timeout=60.0)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if resp.status_code == 501:
                logger.error("Ray Tune not installed or not available on backend.")
                sys.exit(1)
            raise e

        logger.info("Job started successfully.")

        # 3. Poll Status
        logger.info("Polling for completion...")
        while True:
            # We can't get job details by ID easily with current API?
            # TuningService has get_job but API list_jobs endpoint is what we have?
            # Wait, app.py doesn't seem to have GET /api/tuning/jobs/{id}.
            # It has GET /api/tuning/jobs (list).

            # Let's check list and filter
            resp = httpx.get(f"{API_URL}/api/tuning/jobs", timeout=30.0)
            resp.raise_for_status()
            jobs = resp.json()

            # Find our job
            current_job = next((j for j in jobs if j["id"] == job_id), None)

            if not current_job:
                logger.error("Job lost!")
                sys.exit(1)

            status = current_job["status"]
            logger.info(f"Job status: {status}")

            if status in ["completed", "failed"]:
                logger.info(f"Job finished with status: {status}")
                if status == "completed":
                    logger.info(f"Best Params: {json.dumps(current_job.get('best_params_json'), indent=2)}")
                    logger.info(f"Best Run ID: {current_job.get('best_run_id')}")
                else:
                    logger.error("Job failed.")
                break

            time.sleep(5)

    except httpx.ConnectError:
        logger.error("Could not connect to backend. Is it running?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_sweep()
