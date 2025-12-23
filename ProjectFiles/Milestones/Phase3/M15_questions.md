# M15 Clarifying Questions

1.  **Endpoint Strategy (Phase 1)**: The plan suggests `POST /api/tunix/run:async` or `POST /api/tunix/run?mode=async`. I plan to use `?mode=async` (query parameter) to keep the URL path clean and RESTful. Is this acceptable?

2.  **Worker Implementation (Phase 2)**:
    *   I will create a new entry point `backend/tunix_rt_backend/worker.py`.
    *   For the claim loop, the plan mentions "If you want multi-worker safety... optionally use SELECT … FOR UPDATE SKIP LOCKED". Since `asyncpg` / SQLAlchemy supports this, I plan to implement `SKIP LOCKED` for robustness from the start, falling back to simple updates for SQLite if we were supporting it (though currently we seem to be Postgres-only in docker-compose). Is strictly Postgres support sufficient for the worker?

3.  **Frontend Polling (Phase 3)**:
    *   Is a polling interval of **3-5 seconds** acceptable for the run status updates? (Health check is currently 30s).

4.  **Metrics (Phase 4)**:
    *   For the `/metrics` endpoint, I assume we should add `prometheus-client` to `pyproject.toml` and expose standard Prometheus text format. Please confirm.

5.  **Refactoring**: `execute_tunix_run` in `services/tunix_execution.py` currently mixes DB persistence and execution. For the worker, I'll likely need to extract the "execution" part (subprocess call) so the worker can reuse it without the synchronous DB logic (since the worker handles its own DB state transitions). Do you agree with this refactoring approach?
