# Milestone 15: Async Execution & Run Registry (Complete)

**Status:** ✅ Complete
**Date:** 2025-12-23
**Commit:** `b01d98b`

## 🚀 Overview

M15 transforms Tunix RT from a synchronous, blocking execution tool into a robust, asynchronous training platform. Users can now launch training runs that process in the background, freeing up the UI and API for other tasks. This architecture lays the foundation for production-grade operations.

## ✨ Key Deliverables

### 1. Async Execution Engine
*   **Non-blocking API:** `POST /api/tunix/run?mode=async` immediately enqueues a job and returns a `pending` status.
*   **Status Polling:** `GET /api/tunix/runs/{id}/status` allows lightweight monitoring of run progress.
*   **Database-Backed Queue:** Uses PostgreSQL `SKIP LOCKED` for atomic, reliable job claiming without external queues (Redis/Celery not required).

### 2. Tunix Worker
*   **Dedicated Process:** New `worker.py` service that consumes pending runs independently of the API server.
*   **Robustness:** Handles execution failures gracefully, updating run status to `failed` with error details.
*   **Scalability:** Design supports running multiple worker instances in parallel (Postgres locking ensures safety).

### 3. Frontend Experience
*   **Run Async Toggle:** Users can choose between blocking (sync) and non-blocking (async) execution.
*   **Auto-Polling:** UI automatically polls for status updates on async runs and refreshes history upon completion.
*   **Status Indicators:** Visual feedback for `pending`, `running`, `completed`, and `failed` states.

### 4. Observability (Prometheus)
*   **Metrics Endpoint:** `/metrics` exposed for Prometheus scraping.
*   **Key Metrics:**
    *   `tunix_runs_total`: Counter by status and mode.
    *   `tunix_runs_duration_seconds`: Histogram of execution time.
    *   `tunix_db_write_latency_ms`: Monitor DB persistence overhead.

### 5. Code Quality & Security
*   **Refactored Service:** Extracted core execution logic to support both sync and async paths cleanly.
*   **Dependency Hygiene:** Cleaned up invalid types packages; strictly pinned `prometheus-client`.
*   **Coverage:** Maintained >80% line coverage (81.38%).

## 🛠️ Technical Details

*   **Schema Changes:** Added `config` JSON column to `tunix_runs` to persist full request parameters for deferred execution.
*   **Worker Logic:** Implemented a robust `while True` loop with `asyncio.sleep` backoff for empty queues.
*   **Docker:** Added `worker` service to `docker-compose.yml` for turnkey development environment.

## 🔜 Next Steps (M16)

*   **Real-time Logs:** Replace polling with SSE/WebSockets for live log streaming.
*   **Job Management:** Add ability to cancel running jobs.
*   **Checkpointing:** Expose intermediate model checkpoints via API.

