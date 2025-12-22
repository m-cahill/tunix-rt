According to a document from **2025-12-21 to 2025-12-22**, the clean next step is **M11 = Option A (Complete App Extraction)** *plus* knocking out the remaining “fix-first / guardrail” items from the full audit (SBOM, pinned actions, pre-commit, production training docs, SLOs, optional CI tiering, plus a frontend security/coverage sweep).

Below is a **Cursor handoff prompt** for **M11**.

---

## Cursor Prompt — M11: Stabilize + Complete Service Extraction + Training Script Smoke Tests

### Context (do not skip)

* M10 introduced a service layer and thinned `app.py`, but **Phase 5 training script dry-run tests were explicitly deferred to M11**.
* Audit recommends M11 complete the “thin controller” pattern: **all business logic in `services/`, endpoints <20 lines each, and target `app.py <600 lines`**.
* Audit also flags “fix-first” items (SBOM job, pin Actions to SHAs, pre-commit), plus docs (`TRAINING_PRODUCTION.md`, `PERFORMANCE_SLOs.md`) and some frontend work (coverage + vite/vitest to clear npm audit).

### M11 Goal

**Finish architectural hardening** so future work (eval expansion / real Tunix integration) happens on a disciplined base:

1. **Complete extraction** of remaining heavy endpoints into `services/` (UNGAR + dataset build),
2. add **training-script dry-run smoke tests** (fast, CI-safe),
3. implement/verify **security + DX guardrails** from the full audit,
4. optional: **frontend coverage + dependency upgrades** if time/risk permits.

### Non-goals

* No new “big features” (no new eval DB schema, no real Tunix API integration yet).
* No changes to core trace schema unless required for tests/docs.

---

## Phase 0 — Baseline Gate (mandatory)

**Deliverable:** `docs/M11_BASELINE.md`

1. Branch: `m11-stabilize`.
2. Run local CI parity (or repo’s documented CI commands) and record:

   * backend tests + coverage
   * mypy
   * frontend tests
   * e2e tests
3. Create `docs/M11_BASELINE.md` with:

   * commit SHA
   * pass/fail status per job
   * key metrics (coverage, test counts)

Acceptance:

* Baseline doc exists.
* No behavior changes yet.

---

## Phase 1 — Fix-First & Stabilize (security + DX)

### 1A) Re-enable/verify SBOM generation in CI

Audit wants SBOM generation re-enabled with a corrected CycloneDX invocation and uploaded artifact.

Tasks:

* Verify existing SBOM job. If missing/broken:

  * Add/repair CycloneDX SBOM generation (JSON)
  * Upload as CI artifact
  * Ensure JSON validates (best-effort)

Notes:

* CycloneDX tooling/format guidance: ([GitHub][1])

Acceptance:

* CI uploads an SBOM artifact successfully.

### 1B) Pin GitHub Actions to immutable SHAs

Tasks:

* Replace `uses: owner/action@vX` with `uses: owner/action@<full_sha>`.
* Add/update Dependabot config to keep those SHAs fresh.

Rationale:

* GitHub recommends pinning to SHAs for supply-chain integrity. ([GitHub Docs][2])

Acceptance:

* All workflows use SHA-pinned actions.

### 1C) Add pre-commit hooks (ruff + mypy)

Audit explicitly calls for pre-commit hooks to catch issues locally.

Tasks:

* Add `.pre-commit-config.yaml` with:

  * ruff (lint)
  * ruff-format
  * mypy (backend)
* Update README/dev docs with install + run instructions.

Acceptance:

* `pre-commit run --all-files` works locally.

---

## Phase 2 — Docs + Architecture Locks (guardrails)

### 2A) ADR-006: Tunix API Abstraction Pattern

Audit: ADR-006 is “CRITICAL” to future-proof Tunix integration via an adapter pattern.

Tasks:

* Create `docs/adr/ADR-006-tunix-api-abstraction.md`:

  * decision: protocol/adapter for Tunix client
  * consequences + testing strategy (mock client)
  * how training scripts should depend on the protocol, not Tunix directly

Acceptance:

* ADR exists and is consistent with current code direction.

### 2B) Production training docs

Tasks:

* Create `docs/TRAINING_PRODUCTION.md`:

  * “local mode” vs “production mode”
  * required env vars / secrets handling
  * how to run “dry-run” validation
  * how to run real training later (placeholders ok, but clear)

Acceptance:

* Doc is actionable and matches current CLI/scripting.

### 2C) Performance SLOs doc

Tasks:

* Create `docs/PERFORMANCE_SLOs.md` with P95 targets per endpoint and how to measure.

Acceptance:

* SLOs listed for all public endpoints.

---

## Phase 3 — Complete App Extraction (core of M11)

**Objective:** move remaining “fat controller” logic into services and keep `app.py` thin.

### 3A) Extract UNGAR endpoints into service

Audit target: `services/ungar_generator.py` with new tests and `app.py` reduced.

Tasks:

* Create `backend/tunix_rt_backend/services/ungar_generator.py`:

  * `get_status()`
  * `generate_high_card_duel_traces(...)`
  * `export_high_card_duel_jsonl(...)`
* `app.py` endpoints become thin wrappers (<20 lines):

  * parse params
  * call service
  * translate known errors into HTTP responses
* Keep optional-dependency behavior intact (works without UNGAR installed).

Testing:

* Add service-level unit tests for:

  * “UNGAR not installed” path (expected 501/Not Implemented behavior)
  * pure conversion/export behavior that doesn’t require UNGAR import (use fixtures)

Acceptance:

* `app.py` shrinks; endpoints are thin.
* New tests pass in default CI (no UNGAR required).

### 3B) Extract dataset build into service

Audit target: `services/datasets_builder.py` + tests; helps reach `app.py <600` goal.

Tasks:

* Create `backend/tunix_rt_backend/services/datasets_builder.py`:

  * build manifest
  * file/materialization logic (if any)
  * return value structure unchanged
* Controller calls into service.

Testing:

* Add 4+ service tests (per audit) focusing on meaningful branches.

Acceptance:

* `app.py < 600 lines` (target) and each modified endpoint <20 lines.

Implementation constraint:

* **Do not use the same SQLAlchemy `AsyncSession` concurrently**; no `asyncio.gather()` on one session. ([SQLAlchemy][3])

---

## Phase 4 — Training Script Dry-Run Smoke Tests (mandatory)

M10 explicitly deferred this to M11. The original plan describes the exact approach: add `--dry-run` and test via subprocess. 

Tasks:

1. Ensure the training script(s) support `--dry-run`:

   * loads YAML config
   * validates required fields
   * computes output dirs
   * validates manifest schema (if applicable)
   * exits 0 **without** running training
2. Add `backend/tests/test_training_scripts_smoke.py`:

   * call scripts via `subprocess.run([...,"--dry-run"])`
   * assert exit code 0
   * assert a couple stable stdout markers

Acceptance:

* Tests are fast and deterministic.
* No extra deps required.
* CI remains green.

---

## Phase 5 (Optional) — Frontend coverage to 70%

Audit calls out frontend coverage (~60%) and asks to raise to 70% with ~5 component tests.

Tasks:

* Add 5 component tests (RTL/Vitest) for:

  * dataset export UI
  * UNGAR panel basic render + error state
  * trace viewer/export selectors
* Ensure coverage artifact shows ≥70% lines.

Acceptance:

* Frontend tests pass; coverage ≥70%.

---

## Phase 6 (Optional / High Risk) — Upgrade Vite/Vitest to clear npm audit

Audit flags moderate dev-only vulnerabilities and recommends upgrading to Vite 7 / Vitest 4 with thorough testing.

Tasks:

* Upgrade in a separate commit.
* Run unit tests + build + E2E.
* Confirm `npm audit` has 0 moderate+.

Acceptance:

* 0 moderate+ audit findings
* build + tests green

---

## Suggested commit sequence (keep diffs reviewable)

1. `chore(m11): baseline doc`
2. `ci(m11): sbom + pin actions`
3. `chore(m11): add pre-commit`
4. `docs(m11): training production + performance slos + ADR-006`
5. `refactor(m11): extract ungar endpoints to service`
6. `refactor(m11): extract dataset build to service`
7. `test(m11): training script dry-run smoke tests`
8. `test(m11): raise frontend coverage` (optional)
9. `chore(m11): upgrade vite/vitest` (optional, separate PR if preferred)
10. `docs(m11): M11 summary + update tunix-rt.md`

---

## Definition of Done (M11)

* CI green on all jobs.
* `app.py` thin-controller style; target **<600 lines** and endpoints **<20 lines** each (for modified endpoints).
* UNGAR + dataset build logic moved to `services/` with service-level tests.
* Training scripts have `--dry-run` and smoke tests exist. 
* ADR-006 + production training docs + SLO doc shipped.
* (Optional) frontend coverage ≥70%, npm audit cleaned.

---

### Reference docs attached

 

[1]: https://github.com/CycloneDX/cyclonedx-python?utm_source=chatgpt.com "CycloneDX SBOM Generator for Python Projects"
[2]: https://docs.github.com/en/actions/reference/security/secure-use?utm_source=chatgpt.com "Secure use reference - GitHub Docs"
[3]: https://docs.sqlalchemy.org/en/latest/orm/extensions/asyncio.html?utm_source=chatgpt.com "Asynchronous I/O (asyncio) — SQLAlchemy 2.0 ..."
