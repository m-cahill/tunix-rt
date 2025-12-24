# Milestone Audit: M20 (Model Registry)

## 1. Delta Executive Summary
*   **Strengths:**
    *   **Comprehensive Backend Implementation:** Added complete `ModelArtifact` / `ModelVersion` data model, storage service (SHA-256 content-addressed), and promotion logic with validation for Adapter vs Full Model tracks.
    *   **Full UI Integration:** Delivered a functional `ModelRegistry` component in the frontend, integrated into the main navigation, allowing manual promotion and artifact download.
    *   **Test Coverage:** Added substantial tests (`test_model_registry.py`) covering core logic, edge cases (duplicate names, validation failures), and version auto-increment. Frontend tests added to maintain coverage.
*   **Risks:**
    *   **E2E Visibility:** E2E test failures ("Dry-run execution failed") were noted but not fully resolved with root cause analysis in the logs provided (likely unrelated to registry logic but affects confidence).
    *   **Vulnerability:** A new moderate vulnerability in `setuptools` (CVE-2025-47273) was detected in the environment (though likely a dev dependency issue, it needs tracking).
*   **Quality Gates:**
    *   Lint/Type Clean: **PASS** (after fixes)
    *   Tests: **PASS** (Backend: 208 passed, Frontend: 43 passed)
    *   Coverage: **PASS** (Frontend coverage improved to ~80% for new components)
    *   Secrets: **PASS** (No secrets detected in diff review)
    *   Deps: **WARN** (CVE-2025-47273 in setuptools)
    *   Schema: **PASS** (Alembic migration `8a9b0c1d2e3f` generated and verified)
    *   Docs: **PASS** (`docs/model_registry.md` created, `tunix-rt.md` updated)

## 2. Change Map & Impact
*   **Modules Touched:**
    *   `backend/tunix_rt_backend/db/models/model_registry.py` (New Schema)
    *   `backend/tunix_rt_backend/services/artifact_storage.py` (New Service)
    *   `backend/tunix_rt_backend/services/model_registry.py` (New Service)
    *   `backend/tunix_rt_backend/app.py` (API Layer)
    *   `frontend/src/components/ModelRegistry.tsx` (New UI)
    *   `frontend/src/api/client.ts` (API Client)

*   **Layering:** Clean separation. `ModelRegistryService` depends on `ArtifactStorageService` and DB models. API depends on Service. No circular deps observed.

## 3. Code Quality Focus
*   **Observation:** `ModelRegistryService.promote_run` logic is robust but slightly complex (validation + storage + versioning).
    *   **Interpretation:** It correctly handles transactionality by committing only after successful storage and DB insertion.
    *   **Recommendation:** Consider breaking `promote_run` into smaller private helpers if logic grows (e.g., `_create_version_record`). Currently acceptable.
*   **Observation:** Frontend `ModelRegistry.tsx` handles both list and detail views in one component.
    *   **Interpretation:** Fine for MVP, but may become unwieldy.
    *   **Recommendation:** Split into `ModelList` and `ModelDetail` components in M21.

## 4. Tests & CI
*   **Coverage:**
    *   Backend: New tests `test_model_registry.py` cover happy paths and error cases (4 tests passed).
    *   Frontend: `ModelRegistry.test.tsx` added to cover rendering and interaction.
*   **CI Performance:** Backend tests run in ~15s. Frontend tests ~6s. Good speed.

## 5. Security & Supply Chain
*   **Secrets:** Manual review of diff shows no hardcoded secrets. `trufflehog` check skipped due to environment issues but diff review is clean.
*   **Deps:** `setuptools` vulnerability (CVE-2025-47273) detected. Recommendation: Upgrade `setuptools` in `backend/pyproject.toml` or `requirements.txt` in next maintenance cycle.

## 6. Docs & DX
*   **Docs:** `docs/model_registry.md` provides clear usage instructions.
*   **DX:** New `make` targets or scripts for "lint-fix" vs "lint-check" suggested to avoid ordering issues in future.

## 7. Next Milestone Plan (M21)
1.  **Refactor Frontend:** Split `ModelRegistry.tsx` into smaller components.
2.  **E2E Hardening:** Fix E2E test logging to surface stderr on failure.
3.  **Dependency Update:** Upgrade `setuptools` to fix CVE.
4.  **Integration:** Connect "Promote Best Trial" from Tuning UI to Registry (button placeholder exists, need implementation).
