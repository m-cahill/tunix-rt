# M30 Clarifying Questions

Before proceeding with the milestone, I have a few questions:

## Phase 1 — Unused mypy ignores

1. I found 21 `# type: ignore` comments across the codebase. Your `pyproject.toml` already has `warn_unused_ignores = true`. **Should I simply run mypy and remove only the ones it flags as unused**, or do you want me to review each one manually to narrow/justify them?

2. The plan mentions "integration/availability modules" specifically — I see ignores in:
   - `integrations/tunix/availability.py` (1)
   - `integrations/ungar/availability.py` (1)
   - `integrations/ungar/high_card_duel.py` (2)
   - `services/tuning_service.py` (2)
   - `services/tunix_execution.py` (7)
   - `services/evaluation.py` (1)
   - `routers/tunix_runs.py` (5)
   - `db/base.py` (1)

   **Are there specific ones you're aware of being flagged as unused, or should I rely on mypy's output?**

## Phase 2 — Dataset ingest E2E

1. The plan suggests `backend/tools/testdata/e2e_ingest.jsonl`. This directory doesn't exist yet. **Is this the correct location, or would you prefer it in `backend/tests/fixtures/` or elsewhere?**

2. The E2E test will call `POST /api/datasets/ingest` with a file path. The backend starts from `e2e/../backend` directory. **Should the fixture path be relative to the backend root (e.g., `tools/testdata/e2e_ingest.jsonl`)?**

3. **Should the E2E test also verify the ingested traces exist in the database** (e.g., by calling `/api/traces` and checking for them), or is asserting `ingested_count > 0` sufficient?

## Phase 3 — HTTP 422 deprecation

1. I found only **one** usage of `HTTP_422_UNPROCESSABLE_ENTITY` in production code (`datasets.py:58`). The rest are in test files checking response status codes.

2. **Should I update the test files to use `HTTP_422_UNPROCESSABLE_CONTENT` as well**, or leave tests referencing the deprecated constant since they're just checking numeric values?

## Phase 4 — Router docstrings

1. All 10 router modules already have single-line docstrings (e.g., `"""Dataset management endpoints."""`). The audit wants expanded docstrings covering domain, endpoints, and cross-cutting concerns.

2. **How detailed should these be?** For example, for `tunix_runs.py` (500+ lines, many endpoints), should I list every endpoint, or summarize the top-level concerns (e.g., "Run lifecycle management, log streaming, artifact access")?

## Phase 5 — Kaggle dry-run

1. The dry-run requires JAX/Flax/Optax dependencies from `[training]` extras. **Is this expected to be run locally on your machine only** (documenting the commands), or should I create a CI-compatible minimal test?

2. The plan says to use `dev-reasoning-v1` dataset (200 traces). This dataset exists as a flat JSONL file (`backend/datasets/dev-reasoning-v1.jsonl`), not in the versioned folder format that `train_jax.py` expects (`backend/datasets/{name}/dataset.jsonl`). **Should I create a proper versioned folder for it, or adjust the training script to handle flat JSONL files?**

## Phase 6 — Submission checklist

1. **Any specific video requirements or format** you know of for the competition (length, content focus, etc.), or should I leave that as a placeholder for M31?

2. The plan mentions "artifacts to export." **Is there a specific export format or naming convention** the competition expects, or should I document the current artifact structure?

---

Once you answer these, I'll finalize the task breakdown and begin implementation.
