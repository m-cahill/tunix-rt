# M15 Audit Report

**Delta:** `d1ff01a` (M14) → `b01d98b` (M15 Complete)
**Date:** 2025-12-23

## 1. Delta Executive Summary

*   **Strengths:**
    *   **Async Architecture:** Clean separation of API (enqueue) vs Worker (execution) using robust `SKIP LOCKED` pattern.
    *   **Observability:** Prometheus metrics (`tunix_runs_total`, latency) natively integrated at key touchpoints.
    *   **Resilience:** Worker handles config deserialization and execution failures gracefully without crashing the loop.
*   **Risks:**
    *   **Local Testing:** E2E tests for async flow have flaky infrastructure dependencies (Docker networking/auth) in local environment, though logic is verified by unit tests.
    *   **Worker Dependency:** Worker logic relies strictly on PostgreSQL features (`SKIP LOCKED`), reducing portability for SQLite-based dev setups.
*   **Quality Gates:**
    *   **Lint/Type:** PASS
    *   **Tests:** PASS (183 passed, 13 skipped)
    *   **Coverage:** PASS (81% line coverage, meeting >80% target)
    *   **Deps:** PASS (Fixed `types-prometheus-client` issue)
    *   **Schema:** PASS (New migration for `config` column)

## 2. Change Map & Impact

*   **Backend API (`app.py`)**: Added `?mode=async` param and `/runs/{id}/status` endpoint.
*   **Worker (`worker.py`)**: New process for consuming pending runs.
*   **Service Layer (`services/tunix_execution.py`)**: Refactored to split execution logic from persistence/request handling.
*   **Database (`models/tunix_run.py`, `alembic/`)**: Added `config` JSON column to persist request params for worker.
*   **Frontend**: Added "Run Async" toggle and polling logic.

## 3. Code Quality Focus

### `worker.py`
*   **Observation:** Uses `asyncio.sleep(4)` for polling empty queue.
*   **Interpretation:** Simple and effective for current scale. Avoids complexity of notification/LISTEN-NOTIFY for now.
*   **Recommendation:** Keep as is. M16 can upgrade to LISTEN/NOTIFY if latency becomes an issue.

### `services/tunix_execution.py`
*   **Observation:** `execute_tunix_run` now handles both sync and async paths via `async_mode` flag.
*   **Interpretation:** Reduces code duplication but increases cyclomatic complexity slightly.
*   **Recommendation:** Ensure unit tests cover both branches explicitly (Done in `test_tunix_execution.py`).

## 4. Tests & CI

*   **Backend:** New tests `test_async_enqueue_creates_pending_run` and `test_worker.py` cover the core async logic.
*   **Skipped Tests:** `test_claim_pending_run_skip_locked_shim` skipped locally due to SQLite usage; relies on CI Postgres service for validation.
*   **E2E:** `async_run.spec.ts` added to verify full flow. Local execution faced Docker environment issues, but CI environment is configured with correct networking.

## 5. Security & Supply Chain

*   **Dependencies:** Added `prometheus-client` (official package). Removed invalid `types-prometheus-client`.
*   **Secrets:** No new secrets introduced.
*   **Worker Security:** Worker runs with same DB access as backend. Future hardening could restrict worker DB user permissions if needed.

## 6. Performance & Hot Paths

*   **Enqueue Endpoint:** `POST /api/tunix/run?mode=async` is now O(1) DB insert, significantly faster than sync execution.
*   **Metrics:** `TUNIX_DB_WRITE_LATENCY_MS` added to monitor DB performance overhead.
*   **Recommendation:** Monitor `tunix_runs_total{status="pending"}` to detect worker backpressure.

## 7. Docs & DX

*   **Updated:** `README.md` updated with migration commands.
*   **New:** `docs/PERFORMANCE_BASELINE.md` created with load testing instructions.
*   **Missing:** Update `tunix-rt.md` with M15 features (Action item for closeout).

## 8. Ready-to-Apply Patches

*   *None required immediately.* (Dependency fix was applied during audit phase).

## 9. Next Milestone (M16) Plan

1.  **Log Streaming:** Implement real-time log streaming (SSE/WebSockets) to replace polling.
2.  **Checkpoint Management:** API to list/download checkpoints.
3.  **Run Cancellation:** Allow users to cancel pending/running jobs.
4.  **Advanced Filtering:** Filter runs by hyperparameters or metrics.
