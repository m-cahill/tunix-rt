Here’s how I’d run **M21** to get **quick audit-score wins** while staying small/CI-verifiable. This plan is driven by (a) the M20 milestone audit/summary and (b) the full codebase audit’s “biggest opportunities” list.   

---

## M21 Direction

**Theme:** “Security + CI/DX hardening + Tuning→Registry integration”
**Goal:** raise audit score by knocking out the **highest leverage, lowest risk** gaps:

* Frontend supply-chain vulns + dependency drift risk
* E2E failure visibility (make CI failures self-explaining)
* Remove repo noise / stale-doc risk from `ProjectFiles/Workflows`
* Close the M20 follow-ups: “Promote Best Trial” + UI refactor (optional)

M20 explicitly calls out these as the next best candidates.  

---

# Cursor Handoff Prompt — M21 (Quick Wins + Audit Score Boost)

## Context

* M20 Model Registry is shipped and green. 
* M20 audit flags: E2E visibility gap + moderate setuptools CVE + follow-up candidates. 
* Full audit flags: frontend vulnerabilities (Vite/esbuild), dependency drift due to `^`, and `ProjectFiles/Workflows` bloat. 

---

## Phase 0 — Baseline Gate (mandatory)

1. Pull `main`, run full CI-equivalent locally:

   * Backend: `ruff format --check . && ruff check . && pytest -q`
   * Frontend: `npm ci && npm run build && npm test`
   * E2E: run locally if feasible
2. If CI is not green on main, fix that first in a tiny PR.

**Exit:** green baseline.

---

## Phase 1 — Security quick wins (highest ROI)

### 1.1 Patch setuptools CVE-2025-47273 (backend)

* Upgrade `setuptools` to **>= 78.1.1** (patched version). ([NVD][1])
* Ensure `pip-audit` is clean afterward.

**Acceptance:**

* `pip-audit` no longer reports CVE-2025-47273
* CI passes

### 1.2 Fix frontend npm audit (Vite/esbuild) without breaking tooling

* Run `npm audit` (or use existing audit output) and patch:

  * Prefer upgrading Vite to the safe range used by the ecosystem
  * If the issue is the known esbuild advisory, use `package.json` `"overrides"` to force a safe esbuild (commonly `>= 0.25.0`), *without* `npm audit fix --force` (which can downgrade Vite). ([GitHub][2])

**Acceptance:**

* `npm audit` returns **0** moderate+ vulnerabilities (or documented exception if upstream states false-positive, but prefer a real fix)
* `npm run build` passes
* Frontend tests pass

---

## Phase 2 — Supply chain / drift hardening (fast audit-score bump)

### 2.1 Pin frontend deps (remove `^`)

Full audit explicitly calls this out as a risk. 

* Replace `^x.y.z` ranges with **exact versions** in `frontend/package.json`.
* Add repo-level guardrail:

  * `.npmrc`: `save-exact=true` (so future adds stay pinned) ([GitHub][3])

**Acceptance:**

* No caret ranges remain in `frontend/package.json`
* `npm ci && npm run build && npm test` pass

---

## Phase 3 — Repo hygiene (quick DX + docs-score bump)

### 3.1 Tame `ProjectFiles/Workflows` bloat

Full audit recommends gitignore or moving logs out of git. 

* Add `ProjectFiles/Workflows/` to `.gitignore` **if these are generated artifacts**.
* If already committed and must remain, move to `ProjectFiles/_archive/` and document “not source of truth”.

**Acceptance:**

* CI unchanged
* Repo noise reduced + doc note added (short)

---

## Phase 4 — E2E hardening (stop “Dry-run execution failed” dead ends)

M20 audit called out E2E visibility as a risk. 

### 4.1 Attach stderr/details on failure in Playwright

* In the async-run polling helper:

  * If terminal state is `failed`, fetch run detail (`GET /api/runs/{id}` or equivalent)
  * Attach `status_message` + `stderr` excerpt (+ traceback if present) to the test output via Playwright attachments. ([Playwright][4])

**Acceptance:**

* Next failure shows the *real* backend cause in CI logs/artifacts
* No behavior changes when passing

### 4.2 CI artifact upload “always()”

* Ensure Playwright blob/trace report uploads even on failure using `if: always()` in GitHub Actions. ([GitHub][5])

---

## Phase 5 — Product integration quick win (small feature, high perceived maturity)

### 5.1 “Promote Best Trial” from Tuning UI → Registry

M20 next-steps explicitly includes this. 

* Add backend endpoint or frontend wiring that:

  * resolves the best trial/run for a tuning job
  * calls `POST /api/models/{artifact_id}/versions/promote` with that run id
* Keep it minimal: a single button + toast + refresh

**Acceptance:**

* Works end-to-end in local dev
* Frontend tests updated
* CI green

---

## Optional Phase 6 — UI maintainability (only if it’s cheap)

M20 audit suggests splitting `ModelRegistry.tsx`. 

* Extract `ModelList` + `ModelDetail` components (no design changes)
* Keep tests intact

---

# Definition of Done (M21)

* [ ] setuptools CVE fixed (>=78.1.1) ([NVD][1])
* [ ] frontend npm audit clean (vite/esbuild addressed) ([GitHub][2])
* [ ] frontend deps pinned + `save-exact` guardrail ([GitHub][3])
* [ ] `ProjectFiles/Workflows` bloat addressed 
* [ ] E2E failure shows stderr/details via Playwright attachments ([Playwright][4])
* [ ] CI artifacts uploaded even on failures ([GitHub][5])
* [ ] (If included) “Promote Best Trial” works end-to-end 

---

## Why these are “audit score” quick wins

* **Security** improves immediately (frontend audit + setuptools CVE). ([NVD][1])
* **Supply chain** improves (exact versions + save-exact). ([GitHub][3])
* **DX/docs** improve (reduce log noise in repo). 
* **Reliability** improves (E2E failures become actionable). ([Playwright][4])

If you paste the **current `npm audit` output** (or the CI job snippet that lists the 4 moderate vulns), I can tighten Phase 1.2 into an exact “change these versions / add these overrides” patch list so Cursor doesn’t have to iterate.

[1]: https://nvd.nist.gov/vuln/detail/CVE-2025-47273?utm_source=chatgpt.com "CVE-2025-47273 Detail - NVD"
[2]: https://github.com/vitejs/vite/issues/19412?utm_source=chatgpt.com "esbuild security issue from npm audit #19412 - vitejs/vite"
[3]: https://github.com/bodadotsh/npm-security-best-practices?utm_source=chatgpt.com "bodadotsh/npm-security-best-practices"
[4]: https://playwright.dev/docs/api/class-testinfo?utm_source=chatgpt.com "TestInfo"
[5]: https://github.com/nrwl/ci/issues/97?utm_source=chatgpt.com "How can we record artifacts on failure too? · Issue #97"
