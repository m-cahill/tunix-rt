# Codebase Audit: Tunix RT (M23 Snapshot)

**Auditor:** CodeAuditorGPT  
**Date:** December 24, 2025  
**Snapshot:** `95a45d417f88a20614986c215f97853716627762`  
**Focus:** Reliability, Security, CI/CD, Architecture

---

## 1. Executive Summary

**Score: 3.8/5.0 (High Maturity)**

Tunix RT exhibits a high degree of maturity for a research/hackathon project, with enterprise-grade CI/CD patterns (coverage gates, security scanning, artifact management) and clear documentation. The architecture is well-structured into a monorepo with distinct separation of concerns.

**Strengths:**
*   **Robust CI/CD Pipeline:** Multi-stage workflows with path filtering, strict coverage gates (80% line/68% branch), and automated security scanning (SAST/SCA).
*   **Documentation Culture:** Comprehensive documentation (SLOs, Milestones, ADRs) ensures extensive context retention.
*   **Observability First:** Performance SLOs are explicitly defined (M11) and monitoring hooks (Py-Spy, Locust) are documented.

**Opportunities:**
*   **Dependency Determinism (Backend):** While frontend uses `package-lock.json`, the backend lacks a lockfile (`poetry.lock` or `uv.lock`), relying on `pyproject.toml` constraints which may drift.
*   **Test Tier Separation:** CI runs all tests indiscriminately. Separating "Smoke" (critical path) from "Regression" would improve feedback loop speed as the suite grows.
*   **Explicit "Smoke" Coverage:** The current coverage gate is monolithic. A dedicated low-threshold smoke check is missing.

**Heatmap:**
| Category | Score | Trend |
| :--- | :--- | :--- |
| **Architecture** | 4/5 | ➡️ Stable |
| **Modularity** | 4/5 | ↗️ Improving (M23 Refactor) |
| **Code Health** | 4/5 | ➡️ Stable |
| **Tests & CI** | 4.5/5 | ↗️ Strong |
| **Security** | 4/5 | ➡️ Stable |
| **Performance** | 3/5 | ➡️ Baseline set |
| **DX** | 3.5/5 | ➡️ Good |
| **Docs** | 5/5 | ⭐️ Exemplary |

---

## 2. Codebase Map

```mermaid
graph TD
    Root[Repo Root] --> Backend[backend/]
    Root --> Frontend[frontend/]
    Root --> E2E[e2e/]
    Root --> Docs[docs/]
    
    Backend --> App[app.py (Entrypoint)]
    Backend --> Svc[services/]
    Backend --> DB[db/ (SQLAlchemy)]
    Backend --> Schemas[schemas/ (Pydantic)]
    Backend --> Tests_Bk[tests/]
    
    Frontend --> Src[src/]
    Frontend --> Tests_Fe[tests/]
    
    E2E --> Playwright[tests/]
```

**Drift Analysis:**
*   **Intended:** Clean separation of API, Business Logic (Services), and Data Access.
*   **Actual:** Adherence is strong. M10/M23 refactors successfully extracted logic from `app.py` into `services/`.
*   **Citation:** `backend/tunix_rt_backend/services/judges.py` (M23) demonstrates clean service abstraction.

---

## 3. Modularity & Coupling

**Score: 4/5**

The backend effectively uses the Service Repository pattern (implied) via `services/` and `db/`. Pydantic schemas decouple API contracts from database models.

*   **Tight Coupling (Observation):** `tunix_rt_backend/services/tunix_execution.py` likely has hard dependencies on specific training logic/libraries (`tunix`, `jax`) that are optional.
*   **Decoupling Strategy (Recommendation):** Continue using "Optional Dependency" patterns (Adapter pattern) for heavy ML libraries to keep the core CRUD lightweight.
*   **Evidence:** `backend/pyproject.toml` defines `training` and `ungar` as optional extras, confirming modular intent.

---

## 4. Code Quality & Health

**Score: 4/5**

Codebase uses `ruff` (lint/format) and `mypy` (types) in CI. M23 fixed `act()` warnings in frontend tests.

*   **Anti-pattern:** "Catch-all" exception handling in some service layers (inferred from common patterns, verifiable via `grep`).
*   **Fix Example (Hypothetical):**
    *   *Before:* `except Exception as e: return 500`
    *   *After:* `except (ValueError, KeyError) as e: raise HTTPException(400, detail=str(e))`

---

## 5. Docs & Knowledge

**Score: 5/5**

Documentation is a standout feature.
*   **Onboarding:** `README.md` and `tunix-rt.md` provide clear entry points.
*   **Decisions:** `docs/adr/` captures architectural context.
*   **Gap:** A "New Developer's First PR" guide is implied but could be explicit in `CONTRIBUTING.md` (currently `docs/training_readiness.md` serves similar purpose).

---

## 6. Tests & CI/CD Hygiene

**Score: 4.5/5**

**Architecture:**
*   **Tier 1 (Smoke):** `training-smoke.yml` exists but is "Non-Blocking" (Tier 3 behavior). True "PR Smoke" is mixed into the main `backend` job.
*   **Tier 2 (Quality):** `backend` job enforces strict lint/type/test/coverage.
*   **Tier 3 (Nightly):** `training-smoke` scheduled daily.

**Assessment:**
*   **Coverage:** Strict gates (`coverage_gate.py`) are excellent.
*   **Hygiene:** `paths-filter` prevents unnecessary runs. Artifacts (coverage, reports) are preserved.
*   **Gap:** Backend tests run as a monolith. As the suite grows (currently fast), splitting "Fast Unit" vs "Slow Integration" will be necessary.

---

## 7. Security & Supply Chain

**Score: 4/5**

*   **Tools:** `pip-audit`, `npm audit`, `gitleaks` (via action), SBOM (`cyclonedx`).
*   **Risk:** **No backend lockfile.** `pyproject.toml` specifies dependencies, but without a `poetry.lock` or `uv.lock`, builds are not bit-for-bit reproducible and vulnerable to supply chain drift.
*   **Recommendation:** Adopt `uv` or `poetry` to generate a lockfile, or compile `requirements.txt` with hashes.

---

## 8. Performance & Scalability

**Score: 3/5**

*   **SLOs:** Defined in `docs/PERFORMANCE_SLOs.md` (e.g., P95 < 150ms for trace creation).
*   **Monitoring:** Profiling plan exists (`py-spy`, `locust`).
*   **Reality:** Metrics are "Baseline set" but not continuously asserted in CI yet.
*   **Plan:** Add a basic "Load Smoke" test in CI (e.g., `locust` headless for 10s) to catch gross regressions.

---

## 9. Developer Experience (DX)

**Score: 3.5/5**

*   **Setup:** `make` or `script` helpers exist (`scripts/dev.ps1`).
*   **Feedback:** CI is fast (~2-5 mins implied).
*   **Friction:** Windows-first environment (PowerShell scripts) might be friction for Linux/Mac contributors, though standard tools (`pytest`, `npm`) work everywhere.

---

## 10. Refactor Strategy

**Selected: Option A (Iterative)**

*   **Rationale:** The codebase is stable. Improvements should be incremental to avoid disrupting the "Training Readiness" (M22/M23).
*   **Goals:**
    1.  Lock backend dependencies.
    2.  Formalize Test Tiers (Smoke vs Full).
    3.  Automate SLO assertions.

---

## 11. Future-Proofing & Risk Register

| Risk | Impact | Likelihood | Mitigation |
| :--- | :--- | :--- | :--- |
| **Dependency Drift** | High | High | Add Backend Lockfile immediately. |
| **Slow CI** | Med | Med | Split Test Tiers (Smoke/Unit vs Integration). |
| **Model/Judge Drift** | High | Low | Frozen Evaluation Semantics (M22). |

---

## 12. Phased Plan & Small Milestones

### Phase 0 — Fix-First & Stabilize (0–1 day)
*Focus: Secure the supply chain and quick wins.*

| ID | Milestone | Category | Acceptance Criteria | Risk | Rollback | Est | Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0-01** | **Backend Lockfile** | Security | `uv.lock` or `requirements.txt` (pinned) committed; CI uses it. | Low | Revert to `pip install .` | 1h | Infra |
| **P0-02** | **Smoke Marker** | CI | `pytest -m smoke` runs < 10s; CI job added. | Low | Remove marker | 1h | Backend |

### Phase 1 — Document & Guardrail (1–3 days)
*Focus: Enforce architectural standards.*

| ID | Milestone | Category | Acceptance Criteria | Risk | Rollback | Est | Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1-01** | **Strict Smoke Gate** | CI | New CI job fails if Smoke suite coverage < 5% (ensure minimal coverage). | Low | Disable job | 2h | Infra |
| **P1-02** | **Deprecation Policy** | Docs | ADR added for API versioning/deprecation. | None | N/A | 1h | Lead |

### Phase 2 — Harden & Enforce (3–7 days)
*Focus: Performance and deeper validation.*

| ID | Milestone | Category | Acceptance Criteria | Risk | Rollback | Est | Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P2-01** | **Load Smoke** | Perf | CI runs `locust` (10 users, 30s); fails if P95 > SLO. | Med | Disable job | 4h | Backend |

### Phase 3 — Improve & Scale (Weekly)
*Focus: Feature expansion.*

| ID | Milestone | Category | Acceptance Criteria | Risk | Rollback | Est | Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P3-01** | **OpenTelemetry** | Ops | Traces emitted for key paths; connected to localized Jaeger/Console. | Med | Disable OTel | 1d | Backend |

---

## 13. Machine-Readable Appendix (JSON)

```json
{
  "issues": [
    {
      "id": "SEC-001",
      "title": "Missing Backend Lockfile",
      "category": "security",
      "path": "backend/pyproject.toml",
      "severity": "high",
      "priority": "high",
      "effort": "low",
      "impact": 5,
      "confidence": 1.0,
      "evidence": "No poetry.lock, uv.lock, or requirements.txt found in backend root.",
      "fix_hint": "Run 'pip freeze > requirements.lock' or adopt 'uv pip compile'."
    }
  ],
  "scores": {
    "architecture": 4,
    "modularity": 4,
    "code_health": 4,
    "tests_ci": 4.5,
    "security": 4,
    "performance": 3,
    "dx": 3.5,
    "docs": 5,
    "overall_weighted": 3.8
  },
  "phases": [
    {
      "name": "Phase 0 — Fix-First & Stabilize",
      "milestones": [
        {
          "id": "P0-01",
          "milestone": "Add Backend Lockfile",
          "acceptance": ["Lockfile committed", "CI uses lockfile"],
          "risk": "low",
          "rollback": "Delete lockfile",
          "est_hours": 1
        }
      ]
    }
  ],
  "metadata": {
    "repo": "tunix-rt",
    "commit": "95a45d417f88a20614986c215f97853716627762",
    "languages": ["python", "typescript"]
  }
}
```
