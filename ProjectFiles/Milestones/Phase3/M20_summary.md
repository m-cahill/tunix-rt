# Milestone M20 Summary: Model Registry

**Status:** Complete ✅
**Date:** December 23, 2025

## Overview
Milestone M20 introduced a core MLOps capability: the **Model Registry**. This allows users to "promote" successful Tunix training runs into versioned, immutable model artifacts. This bridges the gap between experimentation (runs) and deployment (artifacts).

## Key Features Delivered

### 1. Data Model & Schema
*   **Model Artifacts:** Logical grouping for model families (e.g., "Gemma-2B-SFT").
*   **Model Versions:** Immutable snapshots of a model, linked to a specific Tunix Run.
*   **Schema:** Added `model_artifacts` and `model_versions` tables via Alembic migration (`8a9b0c1d2e3f`).

### 2. Artifact Storage Service
*   **Content-Addressed Storage:** Implemented SHA-256 based storage to ensure data integrity and deduplication.
*   **Local Backend:** Defaults to `backend/artifacts/model_registry`, configurable via `MODEL_REGISTRY_PATH`.
*   **Validation:** Enforces presence of critical files (e.g., `adapter_config.json` + weights OR `config.json` + weights) before promotion.

### 3. Model Registry Service & API
*   **Promotion Logic:** `POST /api/models/{id}/versions/promote` endpoint to copy artifacts, snapshot metadata/metrics, and create a version.
*   **Versioning:** Automatic `v1`, `v2` incrementing if no label is provided.
*   **Download:** `GET /api/models/versions/{id}/download` provides a ZIP of the model artifacts.

### 4. Frontend UI
*   **Registry Tab:** New navigation section for managing models.
*   **Workflows:**
    *   Create new Model Artifacts.
    *   View list of artifacts.
    *   View version details (SHA, size, source run).
    *   **Manual Promotion:** Form to promote a run ID to a specific model version.
    *   **Download:** Button to download artifacts.

## Quality & Reliability
*   **Tests:**
    *   Backend: Integration tests for creation, promotion (Adapter & Full tracks), and versioning logic.
    *   Frontend: Component tests for `ModelRegistry` ensuring UI renders and API calls are correct.
*   **Linting:** Fixed codebase-wide linting/formatting issues.
*   **Docs:** Created `docs/model_registry.md` with usage guide.

## Next Steps (M21 Candidates)
*   **Tuning Integration:** One-click promotion from "Best Trial" in Tuning UI.
*   **E2E Improvements:** Better error surfacing for dry-run failures.
*   **Dependency Maintenance:** Security updates for `setuptools`.

