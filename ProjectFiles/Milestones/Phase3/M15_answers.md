Below are **direct, locked answers** to each M15 clarifying question. These are written so you can **proceed immediately with Cursor** without further back-and-forth.

(Reference: **M15 Clarifying Questions** )

---

## 1️⃣ Endpoint Strategy (Phase 1)

> **Plan:** `POST /api/tunix/run?mode=async`

**✅ Approved. This is the correct choice.**

**Why this is the right call**

* Keeps the resource canonical (`/tunix/run`)
* Avoids path proliferation (`:async` is RPC-ish and harder to evolve)
* Cleanly supports future modes:

  * `mode=sync` (default)
  * `mode=async`
  * `mode=dry_run`
* Preserves backward compatibility and makes migration trivial

**Lock this in.**
Document:

* Default = sync
* Async is opt-in via query param (for now)

---

## 2️⃣ Worker Implementation (Phase 2)

### Entry point

> `backend/tunix_rt_backend/worker.py`

**✅ Approved. Correct location and structure.**

This cleanly separates:

* API request lifecycle
* Background execution lifecycle

---

### Postgres + `SKIP LOCKED`

> Implement `SELECT … FOR UPDATE SKIP LOCKED`

**✅ Approved and recommended.**

### Database support decision (important)

> “Is strictly Postgres support sufficient?”

**✅ Yes — explicitly Postgres-only is acceptable and correct.**

**Rationale**

* CI + docker-compose already assume Postgres
* SQLite compatibility for async workers adds complexity with no payoff
* Hackathon + enterprise narrative favors correctness over lowest-common-denominator DBs
* `SKIP LOCKED` is the *right* primitive for multi-worker safety

**Guardrail to add**

* Add a module-level comment:

  > “Worker requires Postgres due to SKIP LOCKED semantics.”

This prevents future accidental regression.

---

## 3️⃣ Frontend Polling Interval (Phase 3)

> **Proposed:** 3–5 seconds

**✅ Approved. Use 4 seconds.**

**Why**

* Fast enough to feel responsive
* Slow enough to avoid unnecessary DB churn
* Aligns well with:

  * 30s health checks
  * Non-streaming UX expectations

**Guardrails**

* Stop polling on terminal states (`completed`, `failed`)
* Add exponential backoff if polling fails
* Keep polling logic isolated (easy future swap to SSE/WebSockets in M16)

---

## 4️⃣ Metrics (Phase 4)

> Add `prometheus-client` and expose text format

**✅ Confirmed. This is the correct approach.**

### Implementation notes

* Add `prometheus-client` to `pyproject.toml`
* Expose `/metrics` using standard text exposition
* Include at minimum:

  * Run count by status
  * Run duration
  * DB write latency

This directly supports:

* Audit recommendations
* Enterprise observability expectations
* Easy future Grafana integration

---

## 5️⃣ Refactoring `execute_tunix_run`

> Extract execution logic from DB persistence logic

**✅ Strongly agree. This refactor is not optional — it’s correct design.**

### Why this is the right move

* Current function violates single-responsibility
* Worker needs:

  * Execution logic
  * Without request-scoped DB semantics
* Sync path and async worker should share **execution**, not lifecycle

### Recommended structure

```text
services/
  tunix_execution.py
    - run_tunix_subprocess(...)      # pure execution
    - parse_execution_result(...)
    - execute_tunix_run_sync(...)    # API path
```

Worker flow:

* Claims job
* Calls `run_tunix_subprocess`
* Persists results + status transitions

API sync flow:

* Calls `execute_tunix_run_sync`
* Which internally uses the same execution primitive

**Net effect**

* Cleaner code
* Easier testing
* Future cancellation/retry logic becomes trivial

---

## ✅ Final Verdict

All proposed decisions are **approved and locked**.

You can proceed with M15 implementation immediately with:

* `?mode=async`
* Postgres-only worker using `SKIP LOCKED`
* 4s polling interval
* Prometheus metrics via `prometheus-client`
* Execution/persistence refactor

If you want, next I can:

* Generate a **Cursor-ready task breakdown per PR**
* Draft the **worker claim SQL** exactly
* Provide **test case skeletons** for each phase
