According to a document from **December 23, 2025**, M14 is **production-ready**, CI is green, and the next milestone is explicitly **M15 – Async Execution** (with “blocking training runs” called out as a top risk that async execution mitigates).

Below is an **M15 plan you can hand off to Cursor**, broken into **small, end-to-end verifiable phases** with “CI always green” guardrails.

(Attached refs:   )

---

## M15 North Star

Make Tunix runs **non-blocking** end-to-end:

* API can **enqueue** a run and return immediately.
* A worker executes the run out-of-band and updates `tunix_runs`.
* UI shows **Pending → Running → Completed/Failed** and auto-updates.
* Tests cover the queue/worker/status flow, and CI remains deterministic.

This directly targets the “blocking training runs” risk and aligns with the “M15 – Async Execution” direction already documented.

---

## Scope decisions (locked for M15)

### In-scope

1. **Async run execution path** (enqueue + worker + status)
2. **Polling-based UI** updates (no websockets yet)
3. **Webhook notification (optional)** on completion (lightweight “enterprise-y” integration point)
4. **Observability hooks**: metrics + DB write latency metric (ties into audit)
5. Apply the **ready-to-apply low-risk patches** first (to keep the codebase tidy before adding concurrency)

### Explicitly out-of-scope (push to M16+)

* Streaming logs (SSE/websocket)
* Cancel/retry semantics
* Multi-worker distributed locking beyond Postgres “good enough”
* Retention/archival policy automation

---

## Phase 0 — “Audit Patch Day” (keeps M15 clean)

These are already enumerated as ≤1-day follow-ups in the M14 audit; do them first as **PR#1**.

### P0.1 Apply DX patches (backend + docs)

* **Patch 1**: extract stdout/stderr truncation constant (`TUNIX_OUTPUT_MAX_BYTES = 10_240`)
* **Patch 2**: add DB write latency logging around commit (request path)
* **Patch 4**: README migration rollback snippet (alembic downgrade)

**Acceptance criteria**

* No API changes
* `pytest`, `ruff format --check .`, `ruff check .`, `mypy` all green

### P0.2 Fix frontend test warnings

* Wrap async state updates in `act()` per audit suggestion (Patch 3).

**Acceptance criteria**

* `npm test` passes **without** act warnings

### P0.3 CI speed guardrail

* Add Playwright browser cache (Patch 5).

**Acceptance criteria**

* E2E job reproducibly faster on second run (and no flakiness introduced)

---

## Phase 1 — Backend async “enqueue + status” (PR#2)

### P1.1 Add async enqueue endpoint (non-breaking strategy)

To preserve M14’s “zero breaking changes” posture while we migrate UI, introduce a new endpoint rather than changing the existing sync behavior immediately.

* Keep existing: `POST /api/tunix/run` (sync)
* Add new: `POST /api/tunix/run:async` **or** `POST /api/tunix/run?mode=async`

**Response shape**

* `{ run_id, status: "pending" }` (and minimal metadata)

### P1.2 Add status endpoint (small payload)

* `GET /api/tunix/runs/{run_id}/status`
* returns: `status`, `queued_at`, `started_at`, `completed_at`, `exit_code` (if present)

> Use the existing `tunix_runs` row as the single source of truth (the audit notes it’s forward-compatible for M15 async).

### P1.3 Schema + model updates (only if needed)

If the `tunix_runs` table already contains status/timestamps you need, **reuse** it. If anything is missing for queue semantics, add minimally:

* `queued_at` (or reuse `created_at`)
* `started_at`
* `completed_at`
* `status` enum-ish values: `pending | running | completed | failed`

**Guardrail**

* Provide an **idempotent** “enqueue” behavior: repeated enqueue calls with the same `client_request_id` (optional header) should not create duplicates. If too much, at least add a TODO + ADR note.

### Tests (backend)

* Unit test: enqueue creates run with `pending`
* Unit test: status endpoint returns `pending`
* Contract-ish test: invalid run_id → 404

---

## Phase 2 — Worker that claims + executes jobs (PR#3)

### P2.1 Worker implementation (Postgres-backed queue)

Prefer a DB-backed claim loop (no new infra), because it stays “always green” and avoids Redis/Celery setup friction while still being enterprise-respectable for a hackathon-scale system.

**Claim algorithm (safe + simple)**

* Atomically “claim” one pending job by updating status to `running` with a `WHERE status='pending'`
* Use `RETURNING` to get the claimed row

If you want multi-worker safety on Postgres, optionally use `SELECT … FOR UPDATE SKIP LOCKED`, but keep SQLite/unit tests via a fallback codepath.

### P2.2 Execution + persistence behavior

* Execute using existing Tunix execution service
* Persist:

  * `stdout/stderr` (respect `TUNIX_OUTPUT_MAX_BYTES`)
  * `exit_code`
  * `error` (if exception)
  * status → `completed` or `failed`
* Timeout guardrail: config `TUNIX_RUN_TIMEOUT_SECONDS` and mark `failed` on timeout

### P2.3 Worker entrypoint

Add a CLI module, e.g.

* `python -m tunix_rt_backend.worker`

Add `docker-compose` optional worker service for staging/dev parity.

### Tests (backend)

* Worker claims pending run and marks running/completed
* Worker handles exception → failed
* Worker truncates outputs correctly (via constant)

---

## Phase 3 — Frontend “pending UX + polling” (PR#4)

### P3.1 UI changes

* Update “Run Tunix” action to call async endpoint
* Show run immediately in list with `pending`
* Poll:

  * either `GET /runs` list refresh
  * or `GET /runs/{id}/status` for focused polling
* Stop polling when terminal state reached

### P3.2 E2E tests (Playwright)

Add one new spec path:

* Start async run (in dry-run mode)
* Assert status transitions visible (pending → completed)
* Assert details render without strict-mode collisions (continue using scoped/role/testid patterns)

**Guardrail**

* Use `data-testid` for run rows + status badge to avoid the historic “text selector collides with trace-json textarea” issue.

---

## Phase 4 — Observability + baseline performance (PR#5)

This is explicitly recommended as M15 work in the audit:

* Metrics endpoint
* DB write latency metric
* Load test baseline + thresholds

### P4.1 `/metrics` endpoint (Prometheus format)

Expose:

* `tunix_runs_total{status=…}`
* `tunix_runs_duration_seconds` (histogram if you want, or summary)
* `tunix_db_write_latency_ms` (summary/histogram)

### P4.2 Performance baseline doc

* Run `ab` or equivalent on `/api/tunix/runs?limit=20` at 100 r/s
* Record P50/P95/P99 and set alert thresholds (even if just documented)

---

## Cross-cutting guardrails (do these as you go)

1. **Pre-commit enforcement**: add (or reinforce) “run pre-commit before commit” in CONTRIBUTING and/or as a CI job step. The M14 summary explicitly calls this out as the prevention mechanism for format-gate failures.
2. **Deterministic CI**: keep worker tests fast; prefer dry-run execution in tests (as M14 did) to avoid Tunix dependency drift.
3. **No selector debt**: E2E must use role/scoped/testid selectors only (avoid raw `text=` collisions).
4. **Migration discipline**: any schema tweaks come with:

   * downgrade path
   * E2E applies migration
   * README rollback snippet stays current (Patch 4).

---

## “Cursor Handoff” checklist (what to implement, in order)

**PR#1 (P0):** apply patches 1–5 + act() cleanup + Playwright cache
**PR#2 (P1):** async enqueue endpoint + status endpoint + tests
**PR#3 (P2):** worker module + claim logic + tests + optional compose worker
**PR#4 (P3):** frontend polling UX + Playwright E2E coverage
**PR#5 (P4):** `/metrics` + baseline perf doc + thresholds

Each PR must end with:

* `ruff format --check . && ruff check . && mypy … && pytest`
* `npm test`
* `npx playwright test`
* CI green on both Python versions

---

If you want the *most aggressive* version of M15 (fully switching `POST /api/tunix/run` to async by default), do it as a **final step of PR#4** after the UI is migrated—so we don’t strand the frontend mid-flight.
