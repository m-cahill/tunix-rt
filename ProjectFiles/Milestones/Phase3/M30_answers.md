Here are my calls on each item so you can finalize M30 and start immediately.

---

## Phase 1 — Unused mypy ignores

1. **Remove only what mypy flags as unused**, then do a **light pass on the remaining ignores**:

* Step A: run mypy, delete anything flagged by `unused-ignore`
* Step B: for the ignores that remain, quickly ensure they are:

  * **narrow** (use error-code-specific ignores if applicable), and
  * have a **one-line rationale comment** if it’s not obvious

This keeps scope tight while still preventing “ignore barnacles.”

2. **Rely on mypy output**—I’m not pre-anchoring any specific file as “known unused.” Given Starlette/JAX/torch optionality differences between machines/CI, the authoritative signal is the CI mypy run.

---

## Phase 2 — Dataset ingest E2E

1. Location: prefer **`backend/tests/fixtures/`** (it’s clearly test-owned and won’t be confused with production tooling).

* Suggested path: `backend/tests/fixtures/e2e/e2e_ingest.jsonl`

2. Path in the E2E call: **relative to backend root**.

* Use: `tests/fixtures/e2e/e2e_ingest.jsonl`

3. Verification strength:

* **Minimum required:** `ingested_count > 0`
* **Preferred (still stable):** after ingest, call `POST /api/datasets/build` with a filter that selects those ingested traces (e.g., by `source_name` / provenance tag you set in the fixture), and assert `trace_count > 0`.

That proves “ingest → DB persistence → dataset build” end-to-end without brittle assumptions about trace listing order.

---

## Phase 3 — HTTP 422 deprecation

Starlette now deprecates `HTTP_422_UNPROCESSABLE_ENTITY` in favor of `HTTP_422_UNPROCESSABLE_CONTENT` (RFC 9110 naming). ([GitHub][1])

1. Yes: update the **one production usage** to the new constant.

2. Tests:

* If tests **import the deprecated constant**, update them too (or they’ll eventually emit warnings).
* Best practice for tests is often to assert on the **integer** `422` to avoid constant churn across Starlette versions (same semantics either way).

---

## Phase 4 — Router docstrings

Keep them **summary-level**, not exhaustive.

For big routers like `tunix_runs.py`, do **3–8 lines** covering:

* domain responsibility (“run lifecycle, logs, artifacts, metrics”)
* key endpoint families (not every route)
* cross-cutting concerns (e.g., pagination, artifact storage, cancellation semantics, error mapping)

No need to list every endpoint.

---

## Phase 5 — Kaggle dry-run

1. **Local-only** for M30. Don’t add CI for it yet (CI intentionally doesn’t install training extras; forcing it tends to destabilize pipelines). If you later want it, do a manual `workflow_dispatch` “training smoke” job as a separate milestone.

2. **Make `dev-reasoning-v1` conform to the versioned folder format** rather than teaching `train_jax.py` a second data layout.

* Create: `backend/datasets/dev-reasoning-v1/dataset.jsonl` + `backend/datasets/dev-reasoning-v1/manifest.json`
* If you want to keep the flat file around, keep it as a generated artifact (or move it + update docs/scripts).

This reduces surface area and aligns with everything else you’ve built (builder/export/provenance).

---

## Phase 6 — Submission checklist

1. Video requirements (not a placeholder): the competition page specifies:

* **3 minutes or less**
* **published to YouTube**
* **attached to Kaggle Media Gallery** ([Kaggle][2])

So bake those into the checklist now.

2. Artifact export format: no special naming convention seems mandated in the rules page itself (it points you back to the competition “Requirements”). ([Kaggle][3])
   So: document **your current artifact structure** + a recommended “final bundle” naming (run id, dataset key, checkpoint dir, report json). Keep it repo-native and reproducible.

---

If you want a single “decision summary” line for Cursor: **Use mypy as the source of truth, put the ingest fixture under `backend/tests/fixtures/e2e/`, update tests to avoid deprecated 422 constants, expand router docstrings at summary granularity, standardize dev dataset into folder format, and codify the 3-minute YouTube Media Gallery video requirement.**

[1]: https://github.com/encode/starlette/blob/master/starlette/status.py?utm_source=chatgpt.com "starlette/starlette/status.py at main"
[2]: https://www.kaggle.com/competitions/google-tunix-hackathon?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
[3]: https://www.kaggle.com/competitions/google-tunix-hackathon/rules?utm_source=chatgpt.com "Google Tunix Hack - Train a model to show its work"
