# M19: Hyperparameter Tuning with Ray Tune

Tunix RT integrates [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to enable hyperparameter optimization for your model training runs.

## Overview

The tuning system allows you to:
1.  Define a **Search Space** (e.g., learning rate range, batch size choices).
2.  Run multiple **Trials** in parallel (up to configured concurrency).
3.  Automatically select the **Best Run** based on a target metric (e.g., evaluation score).
4.  Persist all trial results and lineage in the database.

## Architecture

*   **Job**: Defines the experiment (search space, dataset, base model, metric).
*   **Trial**: A single execution of the training loop with a specific set of hyperparameters.
*   **Ray Tune**: Orchestrates the execution, sampling, and scheduling of trials.
*   **Tunix Execution**: Each trial invokes the standard Tunix execution pipeline (`TunixRun` + `Evaluation`).

## Usage

### 1. Create a Tuning Job

Submit a `POST` request to `/api/tuning/jobs`:

```json
{
  "name": "Gemma 2B LR Sweep",
  "dataset_key": "ungar_hcd-v1",
  "base_model_id": "google/gemma-2b-it",
  "metric_name": "score",
  "metric_mode": "max",
  "num_samples": 5,
  "max_concurrent_trials": 1,
  "search_space": {
    "learning_rate": {
      "type": "loguniform",
      "min": 1e-5,
      "max": 1e-3
    },
    "batch_size": {
      "type": "choice",
      "values": [4, 8, 16]
    }
  }
}
```

Supported parameter types:
*   `choice`: `{"type": "choice", "values": [...]}`
*   `uniform`: `{"type": "uniform", "min": ..., "max": ...}`
*   `loguniform`: `{"type": "loguniform", "min": ..., "max": ...}`
*   `randint`: `{"type": "randint", "min": ..., "max": ...}`

### 2. Start the Job

```bash
POST /api/tuning/jobs/{job_id}/start
```

This triggers the Ray Tune process. The job status will change to `running`.

### 3. Monitor Progress

```bash
GET /api/tuning/jobs/{job_id}
```

Response includes `trials` list with individual status and metrics.

## Requirements

*   **Ray Tune**: Must be installed (`pip install "ray[tune]>=2.9.0"`).
*   **Database**: Trials are logged to `tunix_tuning_trials` table.
*   **Artifacts**: Ray results are stored in `artifacts/tuning/{job_id}/`.

## Local Development

To run tuning locally:
1.  Ensure you have the `tuning` optional dependencies: `pip install -e ".[tuning]"`
2.  Start the backend.
3.  Ensure your machine has resources for running models (or use `dry_run` logic if mocking). 
    *   *Note: Currently, tuning jobs execute real training runs. Use small datasets or models for testing.*

## Future Work

*   Async execution via worker process (currently runs in background thread).
*   Cluster deployment (connecting to external Ray cluster).
*   Advanced schedulers (ASHA, PBT).
