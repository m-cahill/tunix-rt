Here’s a **Cursor-handoff prompt for M16** that builds directly on M15’s audited next-steps: **real-time logs**, **job cancellation**, **checkpoint exposure**, plus two “low-effort / high-impact” hardening wins (**dependency pinning** + **N+1 batch fix**) called out in the full audit.   

---

## Cursor Prompt — Milestone M16: Operational UX (Live Logs + Cancel + Checkpoints) + Reproducibility Hardening

### Context (M15 baseline)

M15 shipped:

* Async run enqueue (`POST /api/tunix/run?mode=async`)
* Worker with Postgres `SKIP LOCKED`
* Status endpoint (`GET /api/tunix/runs/{id}/status`)
* Polling UI + Prometheus metrics
* Coverage gates still enforced and CI is green
   

M16 should improve **operator experience** and **enterprise-grade stability** without destabilizing CI:

1. **Real-time log streaming** (replace “wait + refresh” feel)
2. **Run cancellation** (pending/running)
3. **Checkpoint management** (list + download artifacts)
4. **Hardening**: pin floating backend deps + remove known N+1 batch write path (quick wins)
    

---

# Goals (M16)

## G1 — Live logs (SSE first)

Implement real-time log streaming using **Server-Sent Events (SSE)** (preferred over WebSockets for simplicity and reliability behind proxies).

## G2 — Cancel runs

Allow users to cancel **pending** and **running** runs with safe state transitions and subprocess termination where possible.

## G3 — Checkpoint registry + download

Expose checkpoints/artifacts for a run:

* list endpoint returns metadata
* download endpoint streams file safely (path traversal protected)

## G4 — Reproducibility & performance quick wins

* Pin backend runtime dependencies (remove floating `>=` where feasible)
* Switch trace batch creation to the existing optimized path and remove the unoptimized N+1 implementation


---

# Non-goals (explicitly out of scope)

* WebSockets log streaming
* Distributed cancellation across multiple hosts
* Full artifact retention policies / lifecycle management
* Multi-tenant authZ / ACLs beyond existing patterns

---

# Delivery strategy (keep CI green)

Do this as **4 small PRs**, each end-to-end verified:

## PR1 — Hardening quick wins (no new features)

### 1) Pin backend runtime deps

* In `backend/pyproject.toml`, change runtime deps from `>=` to `==` (or adopt a minimal lock approach).
* Do **not** pin dev tools in a way that breaks contributor workflow. Keep it pragmatic.

**Acceptance**

* `pip install -e ".[dev]"` works in CI
* All jobs green

### 2) Deprecate unoptimized trace batch

* In `services/traces_batch.py`, make the optimized implementation the default
* Remove/retire the unoptimized function if safe

**Acceptance**

* Existing tests pass unchanged
* No new regressions

---

## PR2 — Backend log streaming (SSE) + persistence model

### Design (keep it simple and durable)

Add a minimal “log chunk” persistence layer so the UI can stream and also replay logs later.

**DB**

* New table: `tunix_run_log_chunks`

  * `id` (pk)
  * `run_id` (fk)
  * `seq` (monotonic int)
  * `stream` (`stdout|stderr`)
  * `chunk` (text)
  * `created_at`
* Add index on `(run_id, seq)`

**Worker changes**

* When executing subprocess, read stdout/stderr incrementally
* Append chunks to DB (bounded):

  * enforce max total bytes per run (reuse existing truncation constant if present)
  * optionally keep only last N bytes in DB while still updating final `stdout/stderr` summary fields

**API**

* `GET /api/tunix/runs/{id}/logs` → SSE stream

  * supports `?since_seq=<int>` for resume
  * emits events: `log`, `status`, `heartbeat`
* `GET /api/tunix/runs/{id}/logs/snapshot` → returns last N lines for initial render (fast)

**Tests**

* Unit: worker writes chunks and increments seq
* Unit: SSE endpoint yields events for existing chunks (use TestClient + async generator)
* Contract: 404 on unknown run id

**Acceptance**

* Run produces streamed logs in dev (docker-compose)
* CI green

---

## PR3 — Frontend live logs + UX upgrade (replace polling feel)

### UI behavior

* In run detail panel:

  * show “Live Logs” tab
  * connect EventSource to `/logs` SSE endpoint
  * append log lines as they arrive
  * on reconnect, use `since_seq` to avoid duplicates

### Guardrails

* SSE connection closed automatically when status becomes terminal
* Add “Reconnect” button
* Keep existing polling for status as fallback (or have SSE also emit status events)

**E2E**

* Update/extend the async run Playwright test:

  * start async run
  * assert live log area receives at least 1 line
  * assert terminal status eventually shown

**Acceptance**

* No flaky selectors: use `data-testid` for log container and run status badge
* CI green

---

## PR4 — Cancellation + checkpoint management

### Cancellation model

Add run lifecycle semantics:

* New statuses: `cancel_requested`, `cancelled`
* API: `POST /api/tunix/runs/{id}/cancel`

  * If `pending`: mark `cancelled` immediately
  * If `running`: mark `cancel_requested` and worker attempts termination

**Worker cancel mechanics**

* Before starting a run: if `cancel_requested` or `cancelled`, do not execute
* During execution: periodically check cancel flag
* If subprocess running:

  * send SIGTERM, wait a short grace period, then SIGKILL if needed
* Persist final status: `cancelled`

**Tests**

* Unit: cancel pending run → cancelled
* Unit: cancel running run triggers termination path (mock subprocess)
* Ensure idempotency: repeated cancel calls don’t break state machine

### Checkpoint management

**Goal:** list & download checkpoint artifacts produced by Tunix runs.

**Storage convention**

* Standardize artifact root (configurable): `TUNIX_ARTIFACT_ROOT`
* Per-run directory: `{root}/{run_id}/`
* Worker writes known checkpoint files there, and/or records discovered artifacts

**DB (optional but recommended)**

* Table `tunix_run_artifacts`

  * `run_id`, `name`, `path`, `size_bytes`, `sha256`, `created_at`

**API**

* `GET /api/tunix/runs/{id}/artifacts` → list metadata
* `GET /api/tunix/runs/{id}/artifacts/{artifact_id}/download` → FileResponse streaming

  * Must protect against path traversal: only allow paths under artifact root

**Frontend**

* Add “Artifacts” tab
* List items with “Download” link

**Acceptance**

* Download works in dev docker-compose
* CI green

---

# Global guardrails (must implement during M16)

1. **State machine enforcement**

   * Centralize allowed transitions in one module (e.g., `run_state_machine.py`)
   * Tests for transition rules (pending→running→completed/failed/cancelled, etc.)

2. **Performance guardrail**

   * Log chunk insertion: batch commits (e.g., flush every N chunks or seconds) to avoid DB thrash

3. **Security guardrail**

   * Artifact downloads must be constrained to `TUNIX_ARTIFACT_ROOT`
   * No raw filesystem paths returned to frontend (return ids + display names only)

4. **CI always green discipline**

   * Each PR ends with:

     * `ruff format --check .`
     * `ruff check .`
     * `mypy`
     * `pytest`
     * `npm test`
     * `playwright test`

---

# Definition of Done (M16)

* Live logs stream in UI via SSE for async runs
* Cancel works for pending + running (worker terminates subprocess)
* Checkpoints/artifacts list + download works
* Backend deps pinned (or locked) without breaking installs
* N+1 trace batch path removed/retired
* CI green on all jobs and coverage gates still pass

---

# Suggested branch/PR names

* `m16-hardening-pins-and-batch`
* `m16-sse-log-streaming-backend`
* `m16-sse-log-streaming-frontend`
* `m16-cancel-and-artifacts`

---

If you want M16 even tighter, you can **merge PR4 into two** (cancel first, artifacts second). But the above sequencing is the “fastest path” while staying enterprise-grade and CI-safe.
