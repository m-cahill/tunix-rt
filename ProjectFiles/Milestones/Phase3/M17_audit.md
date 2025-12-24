# Milestone 17 Audit: Evaluation & Model Quality Loop

**Date:** 2025-12-23
**Delta:** `4bf76cd` (M16 close) -> `HEAD` (M17 close)

## 1. Delta Executive Summary

*   **Strengths:**
    *   **Architecture:** Clean separation of concerns with `EvaluationService`, `TunixRunEvaluation` model, and schemas.
    *   **Automation:** Evaluation loop is fully automated for completed runs (sync & async), reducing manual toil.
    *   **UX:** Leaderboard UI provides immediate visibility into model quality, a major step forward from raw logs.
*   **Risks:**
    *   **Mock Judge:** Currently uses a deterministic hash for scoring. Moving to a real LLM judge (M18) will introduce latency and non-determinism.
    *   **Data Volume:** `details` JSONB column in evaluations table could grow large if we store full verbose traces per run.
*   **Quality Gates:**
    *   Lint/Type Clean: **PASS** (pre-commit hooks enforcing)
    *   Tests: **PASS** (5 new unit tests, coverage stable)
    *   Secrets: **PASS** (no new secrets)
    *   Deps: **PASS** (no new runtime deps, only dev deps if any)
    *   Schema: **PASS** (Alembic migration `5d6e7f8a9b0c` added cleanly)
    *   Docs: **PASS** (Updated `tunix-rt.md` and summaries)

## 2. Change Map & Impact

*   **Backend:**
    *   `db/models/evaluation.py`: New model.
    *   `schemas/evaluation.py`: New Pydantic schemas.
    *   `services/evaluation.py`: Core logic (mock judge).
    *   `services/tunix_execution.py`: Hook for sync evaluation.
    *   `worker.py`: Hook for async evaluation.
    *   `app.py`: New endpoints (`POST /evaluate`, `GET /evaluation`, `GET /leaderboard`).
*   **Frontend:**
    *   `src/components/Leaderboard.tsx`: New page.
    *   `src/api/client.ts`: Updated API client.
    *   `src/App.tsx`: Routing and Run Details integration.
*   **Database:**
    *   New table `tunix_run_evaluations` linked to `tunix_runs`.

## 3. Code Quality Focus

*   **Observation:** `EvaluationService.evaluate_run` mixes logic for "fetching run" and "scoring logic".
    *   *Recommendation:* In M18 (Real Judge), extract the scoring logic into a separate `Judge` strategy pattern (e.g., `MockJudge`, `GemmaJudge`) to keep the service clean.
*   **Observation:** Leaderboard query fetches all rows and sorts in DB.
    *   *Recommendation:* Fine for now, but add pagination (`limit`/`offset`) to `get_leaderboard` before we hit 1000+ runs.

## 4. Tests & CI

*   **Coverage:** 82% line coverage maintained.
*   **New Tests:** `tests/test_evaluation.py` covers:
    *   Success path (completed run).
    *   Failure path (pending/dry-run).
    *   Retrieval & Idempotency.
    *   Leaderboard sorting.
*   **CI:** E2E test `async_run.spec.ts` was fixed to use a valid dataset key, resolving the CI failure seen in `52944140419`.

## 5. Security & Supply Chain

*   **Auth:** Evaluation endpoints are currently public (like all other endpoints). Future requirement: add auth if exposed publicly.
*   **Input Validation:** `EvaluationRequest` allows `judge_override` string - verified it is sanitized/validated in service (currently unused but prepared).

## 6. Performance

*   **Hot Paths:** Leaderboard query is simple (`SELECT ... JOIN ... ORDER BY score DESC`).
    *   *Index:* Added `ix_tunix_run_evaluations_score` and `ix_tunix_run_evaluations_run_id`. Efficient.
*   **Async Worker:** Evaluation happens *after* run completion in the worker loop. This adds ~10ms for mock judge, negligible. Real judge will need to be async/backgrounded to not block the worker if it takes seconds.

## 7. Docs & DX

*   **API:** New endpoints documented in `tunix-rt.md`.
*   **Usage:** "Run with Tunix" flow remains unchanged; evaluation is an automatic value-add.

## 8. Ready-to-Apply Patches

*   **None critical.** Code is clean and passing.

## 9. Next Milestone Plan (M18)

1.  **Refactor Scoring:** Extract `Judge` interface.
2.  **Real Judge:** Implement `GemmaJudge` using an LLM provider (or local model).
3.  **Hyperparameter Tuning:** Add Ray Tune support.
4.  **Regression Gates:** Add API to check "is this run better than baseline?".
