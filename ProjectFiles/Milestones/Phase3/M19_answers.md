Below are **locked, implementation-ready answers** to the M19 clarifying questions. These are **decisions**, not options, so Cursor can proceed without ambiguity. (Questions referenced from )

---

## 1️⃣ Ray Dependency Placement

**Decision:**
**Add Ray Tune as an optional dependency group** (e.g., `tuning`) in `backend/pyproject.toml`.

**Why:**

* Ray is **heavy** and not required for core runtime, CI, or E2E.
* Keeping it optional preserves **fast installs** and **CI stability**.
* M19 introduces tuning as a **capability**, not a baseline requirement.

**Implementation:**

```toml
[project.optional-dependencies]
tuning = [
  "ray[tune]==<pinned_version>"
]
```

**Guardrail:**

* CI does **not** install `.[tuning]` by default.
* Any code importing Ray must be behind a clear feature boundary (service/module level), not imported at app startup.

---

## 2️⃣ Trial Persistence Strategy

**Decision:**
**Implement `TunixTuningTrial` as a real SQLAlchemy model/table now.**

**Why:**

* You explicitly need a **clean FK** to `TunixRun`.
* JSON blobs on the job don’t scale to:

  * querying
  * debugging
  * lineage tracing
* This table becomes the backbone for:

  * trial inspection
  * future UI
  * auditability

**Minimal schema (locked):**

* `id`
* `tuning_job_id` (FK)
* `run_id` (FK → `tunix_runs.run_id`)
* `params_json`
* `metric_value`
* `status`
* timestamps

You can keep it **lean** now and extend later.

---

## 3️⃣ Ray Trainable & Database Initialization

**Decision:**
**Yes — the Ray trainable must initialize its own DB session**, and **yes — reuse the worker pattern**.

**How (pattern to reuse):**

* Mirror the approach in `worker.py`:

  * create engine
  * create session factory
  * manage lifecycle inside the process

**Rules:**

* Do **not** pass sessions across process boundaries.
* Treat each Ray trial as a **short-lived worker**.
* Explicitly close sessions at the end of each trial.

**Outcome:**

* Clean isolation
* No cross-process leaks
* Matches your existing async/worker architecture

---

## 4️⃣ API Prefix

**Decision:**
**Yes — use `/api/tuning/...`**.

**Endpoints (locked):**

* `POST /api/tuning/jobs`
* `POST /api/tuning/jobs/{id}/start`
* `GET  /api/tuning/jobs`
* `GET  /api/tuning/jobs/{id}`

**Why:**

* Keeps tuning **logically separated** from runs/evaluations
* Prevents route clutter
* Leaves room for future sub-resources (trials, artifacts, logs)

---

## 🔒 Summary for Cursor

* **Ray** → optional dependency group `tuning`
* **Trials** → real table (`TunixTuningTrial`) with FK to `TunixRun`
* **Ray trainables** → initialize DB like `worker.py`
* **API** → `/api/tuning/*`

These choices optimize for:

* CI stability
* auditability
* future UI/analysis
* clean separation of concerns

If you want next, I can:

* Draft the **exact `pyproject.toml` diff**
* Provide a **Ray trainable skeleton** wired to your services
* Sketch the **TuningJob + Trial ORM models** verbatim
