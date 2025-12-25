# M21 Milestone Summary

**Status:** ✅ **COMPLETE**
**Date:** December 24, 2025
**Milestone:** M21 - Security Hardening & E2E Reliability
**Branch:** `main` (integrated)

---

## 🚀 Executive Summary

Milestone M21 was a critical hardening sprint. We eliminated all known security vulnerabilities (backend & frontend), stabilized the CI/CD pipeline by fixing SBOM generation and E2E test flakiness, and delivered the "Promote Best Trial" feature to connect the Tuning and Registry workflows.

**Key Wins:**
*   🔒 **Security:** 0 Vulnerabilities (Clean `npm audit` & `pip-audit`).
*   🧪 **Reliability:** E2E tests are now hermetic (self-seeding), eliminating environmental flakes.
*   📦 **Stability:** Frontend dependencies are strictly pinned (`save-exact=true`).

---

## 🛠️ Deliverables

### 1. Security Hardening
*   **Backend:** Patched `setuptools` (CVE-2025-47273).
*   **Frontend:** Upgraded `vite`/`vitest` and overrode `esbuild` to clear 4 moderate vulnerabilities.
*   **Supply Chain:** Enforced exact versioning in `frontend/package.json` and `.npmrc`.

### 2. E2E Reliability
*   **Refactor:** `async_run.spec.ts` now dynamically creates a trace and builds a dataset via API before testing.
*   **Observability:** CI attaches backend stderr and trace zips on failure.
*   **Guardrails:** Added explicit checks for dataset existence in tests.

### 3. Feature: Promote Best Trial
*   **UI:** Added "Promote Best" button to Tuning Job view.
*   **Logic:** Promotes the best run (based on metric) to a Model Registry artifact.

### 4. Repo Hygiene
*   **Cleanup:** Ignored `ProjectFiles/Workflows` logs.
*   **CI:** Fixed `cyclonedx-py` invocation to use environment inspection.

---

## 📊 Metrics

| Metric | Status | Notes |
| :--- | :--- | :--- |
| **Backend Tests** | 208 Passing | 13 Skipped (Optional deps) |
| **Backend Coverage** | ~69% | Slight dip (accepted for M21, fix in M22) |
| **Frontend Vulns** | 0 | Down from 4 |
| **E2E Pass Rate** | 100% | Flakiness resolved |

---

## ⏭️ Next Steps (M22)

**Theme:** Evaluation & Quality Loop

1.  **Coverage Recovery:** Add unit tests for new M20/M21 service logic.
2.  **Evaluation:** Implement "Answer Correctness" scoring (Ground Truth).
3.  **Frontend:** Dataset Browser & Leaderboard visualization.
