Below is an **M1 plan you can handoff to Cursor** that builds directly on your M0 audit: **branch coverage = 0%, missing env validation, missing dependency/secret scanning + SBOM, and light DX gaps**.  

M1 keeps scope tight: **hardening + guardrails + “integration boundary correctness”** (still no RediAI feature creep).

---

# M1 North Star

Make the M0 foundation **enterprise-grade** without expanding product scope:

1. **Raise test rigor** (branch coverage + better branch-path tests)
2. **Add security/supply-chain baseline** (dependency scan, secret scan, SBOM, Dependabot)
3. **Harden configuration** (validated env/settings)
4. **Stabilize integration boundary** (RediAI client robustness + optional local contract checks)
5. Keep CI fast + deterministic

Branch coverage + branch gating is supported via coverage/pytest-cov (`--cov-branch`). ([pytest-cov][1])

---

# M1 Scope

## In-scope (M1)

* **Backend**: branch coverage ≥ 70% + tests for error/if-else paths; validated settings; RediAI client robustness; optional TTL caching.
* **Frontend**: small refactor to typed API client + optional polling (30s) + tests.
* **CI/Security**: pip-audit, npm audit, gitleaks, SBOM generation, Dependabot config. ([GitHub][2])
* **Docs**: ADRs for M0 decisions; short CI + coverage strategy note.

## Explicitly out-of-scope (defer to M2)

* DB migrations / Alembic (audit says “future M1 features will need this,” but it’s not required to harden the base) 
* Auth, deployments, metrics/OTel, trace storage

---

# M1 Deliverables & Acceptance Criteria

## A) Testing & Coverage (highest priority)

### A1. Turn on branch coverage + add real branch tests

* Enable branch measurement (`--cov-branch`) and set gates:

  * **Line coverage**: keep ≥ 80% (already ~82%)
  * **Branch coverage**: **≥ 70%** (currently 0%) 
* Add tests to cover branches in:

  * `app.py` (mock vs real mode paths, “down/error” responses)
  * `redi_client.py` (non-2xx response path, timeout path)
  * `settings.py` (validation failures + defaults)

**Why:** Your audit calls out branch coverage as the #1 gap and notes 4 branches with 0 covered. 

**Tooling note:** pytest-cov supports branch via `--cov-branch`. ([pytest-cov][1])

### A2. Optional: add lightweight contract/property tests (non-blocking)

* Add **Schemathesis** contract tests against your FastAPI OpenAPI schema, **as a non-blocking job** (nightly or main-only) so PRs stay fast. ([Schemathesis][3])

---

## B) Configuration Hardening (env validation)

### B1. Validate settings on startup

* Use `pydantic-settings` for env loading and apply `@field_validator` rules:

  * `REDIAI_BASE_URL` must be a valid URL
  * Ports in range 1–65535
  * `REDIAI_MODE` must be in `{mock, real}`
* Fail fast with clear error messages if invalid.

Pydantic Settings is explicitly designed for env-based settings management. ([Pydantic][4])

---

## C) Security & Supply Chain Baseline

### C1. Dependency scanning

* **pip-audit** (warn-only first, then enforce later): scans Python env for known vulnerabilities. ([GitHub][2])
* **npm audit** (warn-only first): reports known vulnerabilities and exits non-zero if findings exist. ([npm Docs][5])

### C2. Secret scanning

* Add **gitleaks** action to CI. ([GitHub][6])

### C3. SBOM generation

* Generate CycloneDX SBOM for backend (as CI artifact). ([CycloneDX BOM Tool][7])

### C4. Dependabot

* Add `.github/dependabot.yml` for:

  * `pip` (backend) weekly
  * `npm` (frontend + e2e) weekly
    Dependabot config is driven by `dependabot.yml`. ([GitHub Docs][8])

---

## D) Integration Boundary Improvements (still “small”)

### D1. Improve RediAI health behavior (better diagnostics)

* In `redi_client.health()`:

  * Treat non-2xx as down with explicit `HTTP <code>` message
  * Differentiate connection vs timeout vs HTTP error in error string (helps debugging)
* Add tests for each branch (this helps branch coverage and reliability simultaneously). 

### D2. Optional: simple TTL cache for `/api/redi/health`

* Cache RediAI health response for ~30s to avoid hammering RediAI during UI polling.
* Unit test cache hit/miss behavior.

(Keep this minimal; no external cache system.)

---

## E) Frontend Maintainability (tiny)

### E1. Add a minimal typed API client

* Create `frontend/src/api/client.ts` with:

  * `getApiHealth()`
  * `getRediHealth()`
* Use it from `App.tsx` (reduces coupling and makes future endpoints cleaner). 

### E2. Optional: 30s polling

* Add polling for both health endpoints
* Ensure cleanup on unmount
* Add unit tests with fake timers

---

## F) DX + Docs

### F1. Makefile (top DX win)

* `make install`, `make test`, `make lint`, `make e2e`

### F2. ADRs (lightweight but important)

Add `docs/ADR_00X_*.md`:

* ADR-001: Mock/Real RediAI integration pattern
* ADR-002: CI strategy (paths-filter conditional jobs)
* ADR-003: Coverage strategy (line + branch thresholds)

Audit explicitly recommends ADRs. 

---

# M1 Phases (small PRs, each independently mergeable)

## Phase 1 — Coverage + reliability (PRs 1–3)

1. **PR1:** Enable branch coverage + add branch tests for `app.py`
2. **PR2:** Add branch tests for `redi_client.py` + improved error handling
3. **PR3:** Settings validation (pydantic-settings + validators) + tests

**Phase DoD:** CI enforces **branch ≥ 70%** + line ≥ 80%.

## Phase 2 — Security baseline (PRs 4–6)

4. **PR4:** pip-audit + npm audit jobs (warn-only, upload reports) ([GitHub][2])
5. **PR5:** gitleaks CI job ([GitHub][6])
6. **PR6:** CycloneDX SBOM artifact + Dependabot config ([CycloneDX BOM Tool][7])

## Phase 3 — Maintainability polish (PRs 7–9)

7. **PR7:** Frontend typed API client + tests
8. **PR8 (optional):** 30s polling + TTL cache + tests
9. **PR9:** ADRs + Makefile (and update docs)

## Phase 4 — Quality tier (optional, main-only / nightly)

10. **PR10 (optional):** Schemathesis contract tests as nightly/main-only (non-blocking initially) ([Schemathesis][3])

---

# Definition of Done (M1)

* ✅ Backend: **Line ≥ 80%** AND **Branch ≥ 70%**, enforced in CI ([pytest-cov][1])
* ✅ Settings: invalid env config fails fast with clear errors ([Pydantic][4])
* ✅ Security baseline in CI:

  * pip-audit + npm audit (warn-only OK for M1)
  * gitleaks scan
  * SBOM uploaded
  * Dependabot enabled ([GitHub][2])
* ✅ Frontend: still green, tests stable, build passes
* ✅ Docs: ADRs exist for key decisions
* ✅ CI remains fast with conditional jobs

---

# Single Cursor Handoff Prompt (copy/paste)

```text
Implement Milestone M1 for m-cahill/tunix-rt. M0 is complete and CI is green. M1 is a hardening milestone: raise test rigor (branch coverage), add configuration validation, and establish a security/supply-chain baseline—without expanding product scope.

Reference current audit priorities:
- Branch coverage is 0% (must fix).
- Missing environment variable validation.
- No dependency scanning, secret scanning, or SBOM.
- ADRs missing.
(See attached M00_audit.md and M00_summary.md.)

Constraints:
- Keep changes small and mergeable (many small PRs).
- CI must remain deterministic; RediAI must never be required in CI.
- Maintain existing conditional CI design (paths-filter), keep runtime fast.
- Conventional Commits required.

PHASE 1: Coverage + Reliability
1) Enable branch coverage measurement and enforcement:
   - Update backend pytest/coverage config to run branch coverage (pytest-cov --cov-branch).
   - Gates: line coverage >= 80% and branch coverage >= 70% (backend).
2) Add tests to cover conditional branches:
   - app.py: REDIAI_MODE mock vs real, down/error return paths.
   - redi_client.py: non-2xx HTTP path, timeout/connection error path.
3) Improve RediClient.health() diagnostics:
   - non-2xx -> {"status":"down","error":"HTTP <code>"}
   - exception -> {"status":"down","error":"<class>: <message>"}
   - Add tests for each branch.
4) Add strict environment variable validation:
   - Use pydantic-settings (if not already) and add @field_validator checks:
     - REDIAI_MODE in {mock, real}
     - REDIAI_BASE_URL valid URL
     - ports in 1..65535
   - Add tests verifying invalid config fails fast (clear error messages).

PHASE 2: Security + Supply Chain Baseline (warn-only initially)
5) Add pip-audit job (warn-only) for backend and upload report artifact.
6) Add npm audit job (warn-only) for frontend and upload report artifact.
7) Add gitleaks secret scanning job.
8) Add SBOM generation:
   - CycloneDX SBOM for backend (artifact).
9) Add Dependabot:
   - .github/dependabot.yml for pip (backend) and npm (frontend + e2e), weekly schedule.

PHASE 3: Maintainability + DX
10) Frontend: add minimal typed API client module (frontend/src/api/client.ts) and refactor App.tsx to use it; keep tests updated.
11) Optional: add 30s polling in UI (clean interval teardown) + backend TTL cache for /api/redi/health; test both.
12) Add Makefile with install/test/lint/e2e targets.
13) Add ADRs in docs/:
   - ADR-001 mock/real RediAI integration
   - ADR-002 CI conditional jobs strategy
   - ADR-003 coverage strategy (line + branch)

Optional Phase 4 (main-only or nightly, non-blocking initially):
14) Add Schemathesis contract test job against FastAPI OpenAPI schema.

Verification requirements:
- All unit tests pass (backend/frontend), E2E still passes.
- CI enforces backend line>=80 and branch>=70.
- Security jobs run and report (warn-only ok).
- SBOM artifact uploaded.
- Dependabot config added.
- Documentation updated and accurate.

Deliver each numbered item as a small PR-sized commit (or separate PRs if preferred), using Conventional Commits.
```

---

If you want M1 to include **one additional “real integration” check** without risking CI, the safest pattern is: add a **separate `integration` pytest marker** that runs only when `REDIAI_MODE=real` locally (skipped in CI). That preserves your “integration early” preference while keeping CI fully deterministic.

[1]: https://pytest-cov.readthedocs.io/en/latest/config.html?utm_source=chatgpt.com "Configuration - pytest-cov 7.0.0 documentation"
[2]: https://github.com/pypa/pip-audit?utm_source=chatgpt.com "pypa/pip-audit"
[3]: https://schemathesis.readthedocs.io/?utm_source=chatgpt.com "Schemathesis"
[4]: https://docs.pydantic.dev/latest/concepts/pydantic_settings/?utm_source=chatgpt.com "Settings Management - Pydantic Validation"
[5]: https://docs.npmjs.com/cli/v9/commands/npm-audit?utm_source=chatgpt.com "npm-audit"
[6]: https://github.com/gitleaks/gitleaks-action?utm_source=chatgpt.com "Protect your secrets using Gitleaks-Action"
[7]: https://cyclonedx-bom-tool.readthedocs.io/?utm_source=chatgpt.com "CycloneDX SBOM Generation Tool for Python - Read the Docs"
[8]: https://docs.github.com/en/code-security/dependabot/working-with-dependabot/dependabot-options-reference?utm_source=chatgpt.com "Dependabot options reference"
