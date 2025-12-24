Below is a **copy/paste prompt for Cursor** to execute **M19**. It assumes **M18 delivered** the pluggable `Judge` system (incl. `GemmaJudge`), regression baseline plumbing, and the paginated evaluation/leaderboard work. 

---

## Cursor Prompt — M19: Hyperparameter Tuning (Ray Tune) + Best-Run Selection

### Goal

Implement **M19 Hyperparameter Tuning** so we can run structured sweeps (locally first) and automatically pick/store the **best configuration + best run**, using our existing **TunixRun + evaluation + Judge** pipeline.

Use **Ray Tune**’s modern **`Tuner`** API, report metrics from each trial, and support concurrency limits + resumable output storage. ([Ray][1])

---

### Scope (keep tight)

1. **Backend-first** (required):

   * New tuning “experiment” concept (DB model + service + API).
   * Ray Tune runner that executes trials by invoking our existing run/eval pipeline.
   * Persist best result + link it to a `TunixRun` id.
2. **Frontend** (minimal but included):

   * Simple list/detail UI for tuning experiments + best result.
   * Link to the winning run/evaluation pages we already have.

---

### Key Design Decisions (implement these)

* Use Ray Tune **`tune.Tuner(..., param_space=..., tune_config=..., run_config=...)`**. ([Ray][1])
* Trials must report at least one scalar metric via **`tune.report({metric: value, ...})`**. ([Ray][1])
* Persist outputs in a deterministic directory via Tune storage config (so experiments are inspectable/resumable). ([Ray][2])
* Limit parallelism with **`max_concurrent_trials`** (default small for CI/local). ([Ray][1])

---

### Data Model

Add ORM models (names are suggestions; keep consistent with existing patterns):

**`TunixTuningJob`**

* `id` (uuid)
* `name`
* `status` (`created|running|completed|failed|canceled`)
* `dataset_key`
* `base_model_id` (or `model_id`)
* `mode` (`local` for now; leave room for async later)
* `metric_name` (e.g., `score_mean`)
* `metric_mode` (`max|min`)
* `num_samples`
* `max_concurrent_trials`
* `search_space_json` (validated; JSON)
* `best_run_id` (FK to `tunix_runs.run_id`, nullable until done)
* `best_params_json`
* `ray_storage_path` (string)
* timestamps

**`TunixTuningTrial`** (optional but recommended)

* `id`
* `tuning_job_id`
* `run_id` (FK to `tunix_runs.run_id`)
* `params_json`
* `metric_value`
* `status`
* timestamps

> If you want to keep schema minimal: skip `TunixTuningTrial` and store just “best”, but still store *at least* a structured list of trial summaries somewhere (DB or Ray output parsing).

---

### Backend Implementation Tasks

1. **Dependencies**

   * Add Ray Tune dependency (pin reasonably; keep CI stable).
   * Ensure backend install remains deterministic.

2. **Validation layer**

   * Define a schema for `search_space_json`:

     * allow `{ "param": {"type":"choice","values":[...] } }`
     * allow `{ "param": {"type":"uniform","min":..., "max":...} }`
     * (optional) loguniform/int/uniform
   * Convert schema → Ray Tune objects (`tune.choice`, `tune.uniform`, etc.).

3. **Tuning runner**

   * Create `backend/services/tuning_service.py`:

     * `create_job(...)`
     * `start_job(job_id)` (runs Ray Tune)
     * `get_job(job_id)` / `list_jobs(...)`
   * Implement Ray **trainable** function used by `Tuner`:

     * It receives trial params.
     * It triggers a real `TunixRun` (reuse existing service methods).
     * After run completes, it triggers evaluation (Judge) and produces a scalar metric.
     * Calls `tune.report({metric_name: metric_value, "run_id": run_id})`. ([Ray][1])
   * After tuner completes:

     * Find best result (Ray result grid).
     * Write `best_run_id` + `best_params_json` to `TunixTuningJob`.
     * Optionally store all trials in `TunixTuningTrial`.

4. **Storage + reproducibility**

   * Use a per-job storage directory, e.g. `artifacts/tuning/<job_id>/`.
   * Configure Ray Tune output storage accordingly (don’t rely on random `~/ray_results`). ([Ray][2])
   * Store a copy of:

     * request payload
     * resolved param_space
     * final best result summary

5. **API routes**

   * `POST /api/tuning/jobs` create
   * `POST /api/tuning/jobs/:id/start` start
   * `GET /api/tuning/jobs` list (paginated)
   * `GET /api/tuning/jobs/:id` detail (include best + trial summaries)

6. **Testing**

   * Unit tests for:

     * search-space validation/conversion
     * create/list/get job
   * Integration-style test (fast):

     * run a tiny tuning job with 1–2 samples using a **fake trainable** or a **mocked run+eval** path, and assert `best_run_id` is populated.

7. **Docs**

   * `docs/tuning.md`:

     * how to create/start a job
     * how metrics are computed (ties to Judge evaluation)
     * where artifacts are stored
     * how to reproduce a tuning run

---

### Frontend (minimal)

* Add a “Tuning” page:

  * List tuning jobs (status, dataset_key, metric, best score, created_at)
  * Detail view: params space, current status, best params, link to winning run
* Keep styling consistent; no deep UX polish in M19.

---

### Definition of Done

* ✅ New tuning job can be created and run locally.
* ✅ Produces multiple trials, each creates a real `TunixRun`, then evaluates it.
* ✅ Best run is persisted (`best_run_id` + best params) and visible in UI.
* ✅ CI green (format/lint/tests).
* ✅ Docs added.

---

### Guardrails

* Default `max_concurrent_trials=1` unless explicitly set (avoid melting laptops/CI). ([Ray][1])
* If a trial fails, mark it failed and continue unless too many failures (simple threshold).
* Keep Ray usage encapsulated so we can later swap from “local Tune” to “cluster/async”.

---

If you want, I can also generate a **ready-to-paste API request example** (`curl`) and a **sample search_space_json** you can use to smoke-test M19 immediately.

[1]: https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html "Hyperparameter Tuning with Ray Tune — Ray 2.53.0"
[2]: https://docs.ray.io/en/latest/_sources/tune/tutorials/tune-storage.rst.txt "docs.ray.io"
