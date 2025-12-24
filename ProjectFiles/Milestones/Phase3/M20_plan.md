Below is a **Cursor handoff prompt** for **M20**, aligned with the M19 audit’s recommended direction (“Model Registry”) and the fact that M19 is now complete/pushed.  

---

## Cursor Handoff Prompt — M20: Model Registry + Artifact Promotion (Enterprise-Grade, E2E Verified)

### Context

* **M19 is complete and pushed**: Ray Tune integration + DB job/trial lineage + Tuning UI. 
* The M19 audit recommends **M20 = Model Registry**: create model/version tables, implement promotion (Run Output → Registry), add API + UI for versioning and serving. 

### M20 Goal (North Star)

Create a **first-class Model Registry** that can:

1. **Promote** a completed `TunixRun` output to a **versioned** model artifact,
2. Provide **immutable, reproducible** metadata (provenance + config + metrics),
3. Support basic **listing, tagging, viewing, and downloading** in UI,
4. Integrate cleanly with tuning (e.g., “promote best trial”).

**Hard requirement:** stay **end-to-end verified**: migrations + API + UI + tests + docs, CI green.

---

# Phase 0 — Baseline Gate (no new features yet)

**Objective:** ensure we don’t build on a shaky base.

✅ Do:

* Pull latest `main`.
* Run: backend format/lint/tests, frontend build/tests, and e2e (if available).
* If anything fails, fix it first in a tiny PR.

**Exit criteria:** CI is green on main before feature work starts.

---

# Phase 1 — Data Model + Migration (minimal but future-proof)

## 1.1 Add DB Models

Create new models (module: `backend/tunix_rt_backend/db/models/model_registry.py` or similar):

### `ModelArtifact`

Represents a logical model “family” (e.g., “Gemma3-1B Tunix SFT”).

* `id` (uuid / int)
* `name` (unique-ish, indexed)
* `description`
* `task_type` (optional string)
* `created_at`, `updated_at`

### `ModelVersion`

Represents a specific promoted artifact build.

* `id`
* `artifact_id` (FK → ModelArtifact)
* `version` (string or int; enforce uniqueness with `(artifact_id, version)`)
* `source_run_id` (FK → TunixRun, nullable but expected)
* `status` (e.g., `created`, `ready`, `failed`)
* `metrics_json` (JSONB) — store eval results snapshot
* `config_json` (JSONB) — store training config snapshot
* `provenance_json` (JSONB) — git sha, dataset manifest id, base model id, etc.
* `storage_uri` (string) — where files live (local path initially)
* `sha256` (string, indexed) — content hash of promoted bundle
* `size_bytes`
* `created_at`

**Guardrails:**

* ModelVersion is **immutable** once `status=ready` (enforce at service layer; optional DB constraint).
* Promotion should be **idempotent** by `(source_run_id, sha256)` (prevent duplicates).

## 1.2 Alembic Migration

Create migration to add tables + indexes + constraints.

**Exit criteria:**

* `alembic upgrade head` succeeds.
* Models import cleanly.

---

# Phase 2 — Artifact Storage + Promotion Service

## 2.1 Storage Abstraction (local-first)

Implement a small storage interface (e.g., `backend/tunix_rt_backend/services/artifact_storage.py`):

* `put_directory(src_dir) -> (storage_uri, sha256, size_bytes)`
* `get(storage_uri) -> filesystem path OR streaming handle`
* Start with **local filesystem** under something like:

  * `backend/artifacts/model_registry/<sha256>/...`
* Use **content-addressed storage** (`sha256`) to guarantee immutability.

## 2.2 Promotion Service

Implement `ModelRegistryService`:

* `create_artifact(name, description, …)`
* `create_version_from_run(artifact_id, source_run_id, version_label?, tags?)`

Promotion logic:

1. Validate `TunixRun` exists and is in a promotable state (e.g., completed/has outputs).
2. Locate run outputs (whatever artifact directory / files currently produced).
3. Bundle/copy outputs into registry storage (directory copy).
4. Compute sha256 + size.
5. Snapshot **metrics** and **config** at time of promotion into JSON fields.
6. Write `ModelVersion` row; set `status=ready`.

**Guardrails:**

* Refuse promotion if run has no artifacts.
* Refuse promotion if run is failed (unless explicitly `force=true`).
* If the exact same sha256 already exists for the same run, return the existing ModelVersion (idempotent).

**Exit criteria:**

* Unit tests cover: happy path, missing artifacts, failed run, idempotency.

---

# Phase 3 — API Layer

Add endpoints (paths can be adjusted to match project conventions):

### Artifacts

* `POST /api/models` → create `ModelArtifact`
* `GET /api/models` → list artifacts
* `GET /api/models/{artifact_id}` → artifact details + versions

### Versions

* `POST /api/models/{artifact_id}/versions/promote`

  * body: `{ "source_run_id": "...", "version": "v1" }`
* `GET /api/models/versions/{version_id}`
* `GET /api/models/versions/{version_id}/download`

  * returns a zip/tarball stream OR a signed/local link approach (start simple: stream zip)

**Security baseline:**

* If the app already has auth, ensure endpoints are protected consistently.
* If no auth exists, at least add a **single guard**: disable download in production unless explicitly enabled via env var.

**Exit criteria:**

* API tests for creation/listing/promotion/download.

---

# Phase 4 — Frontend UI (Model Registry Tab)

Add a new navigation tab: **Model Registry**.

### MVP Screens

1. **Model list** (cards/table): name, description, latest version, created_at
2. **Model detail**: versions table with:

   * version label
   * metrics summary (key metric + value)
   * sha256
   * source run link
   * download button

### Integrations

* On **Tuning job detail**, add:

  * “Promote Best Trial” button → calls promote endpoint using job.best_run_id or best trial run id (if stored)

**Exit criteria:**

* `npm run build` passes
* UI loads without console errors
* Manual smoke: create model → promote run → see version → download works

---

# Phase 5 — Tests + E2E + Docs

## 5.1 Backend tests

* Unit tests: storage hashing, idempotency, promotion validation
* Integration tests: create run fixture with artifacts, promote, fetch, download

## 5.2 E2E (if the suite supports it)

Add/extend a Playwright flow:

* create model artifact
* promote from a known completed run fixture
* verify version appears
* click download (at least confirms endpoint returns 200)

## 5.3 Docs

Add `docs/model_registry.md`:

* concepts: artifact vs version
* promote run → version
* download usage
* provenance fields

Update main system doc to include registry.

**Exit criteria:**

* CI green
* Docs updated
* Migration + API + UI + tests merged

---

# Definition of Done (M20)

* [ ] DB tables + migration exist and validated
* [ ] Promotion copies artifacts into content-addressed registry storage
* [ ] ModelVersion stores: sha256, size, config snapshot, metrics snapshot, provenance snapshot, source_run_id
* [ ] API supports create/list/detail/promotion/download
* [ ] Frontend shows Model Registry and version details
* [ ] Tuning UI can “Promote Best Trial”
* [ ] Tests added (unit + integration); E2E updated if applicable
* [ ] CI green, reproducible local run instructions documented

---

# Extra Guardrails (Add if cheap, otherwise backlog)

* Add a small “registry retention” policy (keep last N versions) — optional
* Add a “promote requires eval complete” check — recommended
* Add SBOM-ish metadata capture for model version (dependency lock hash) — optional

---

## Implementation Notes (to reduce churn)

* Keep M20 **local-first** (filesystem storage). Don’t introduce S3/GCS yet.
* Prefer **minimal surface area**: one service + one storage abstraction.
* Keep the versioning scheme simple: either integer auto-increment per artifact or user-provided string validated.

---

If you want, paste the **current TunixRun artifact output structure** (where files land) and I’ll tailor the promotion/storage code paths to match it exactly (so Cursor doesn’t have to guess).
