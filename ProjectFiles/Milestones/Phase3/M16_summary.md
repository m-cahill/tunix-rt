M16 complete. All PRs implemented and verified.

1.  **Hardening**: Pinned backend deps and removed N+1 trace batch.
2.  **Live Logs**: 
    -   Schema: `tunix_run_log_chunks`.
    -   Backend: `LogManager` streams logs to DB.
    -   API: `GET /logs` (SSE).
    -   Frontend: `LiveLogs` component.
3.  **Cancellation**: 
    -   API: `POST /cancel`.
    -   Worker: Monitors status change and terminates process.
4.  **Artifacts**: 
    -   API: `GET /artifacts`, `GET /download`.
    -   Frontend: List and download links.

**Tests Passed**: 197 backend tests (including new `test_logs.py`).
**Documentation**: Updated `tunix-rt.md`.
