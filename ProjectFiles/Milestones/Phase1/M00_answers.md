Here are my answers / decisions for each M0 clarifying question, aligned with “**stay small and tested**” while still integrating with your locally running RediAI. 

---

## 1) Scope & enhancement integration

**Answer:** **M0 = minimal foundation** (as in M00_plan.md).
All the heavier enhancements (OTel, SBOM, Scorecard, SLSA provenance, mutation testing, 85%+ branch coverage, etc.) should be **M1+** and layered incrementally.

**Reason:** Full-stack + RediAI probe + unit tests + E2E + CI is already a real M0. Adding the whole enhancement suite now increases failure surface and slows iteration.

---

## 2) Python version

**Answer:** Use **Python 3.11** as the local/dev target.
Run CI on **3.11 and 3.12** (keep it tight for M0).

---

## 3) Coverage targets for M0

**Answer:** **70% line coverage** gate in M0 (backend).
Raise to **85% (line+branch)** in M1 when the surface area stabilizes.

---

## 4) RediAI integration (critical clarification)

**Answer:** Confirmed: **Option B — Integration Project**.
`tunix-rt` is a new project that integrates with your locally running RediAI via `/api/redi/health`, with **mock mode in CI** and **real mode locally**.

---

## 5) Deployment targets

**Answer:** **No deployment in M0.**
M0 is local + CI only. Add Netlify/Render (or equivalents) in **M1 or M2**.

---

## 6) Repository structure

**Answer:** **Yes — the proposed structure is correct.**
One small tweak I recommend (optional): keep `e2e/` but **reuse frontend Playwright deps** if you want fewer lockfiles. If you prefer maximum isolation, keep `e2e/package.json` as proposed.

---

## 7) Technology stack confirmation

**Answer:** **Confirmed as written:**

* Backend: FastAPI + httpx + pytest
* Frontend: Vite + React + TS + Vitest + RTL
* E2E: Playwright
* Package manager: **npm**
* CI: GitHub Actions + `dorny/paths-filter` for conditional jobs

No changes needed for M0.

---

## 8) Conventional Commits

**Answer:** **Yes — strict Conventional Commits** for all commits.

Example:

* `feat(backend): add /api/health and RediAI health probe`
* `test(frontend): add App status rendering tests`
* `chore(ci): add conditional workflow with paths-filter`

---

## 9) Git workflow

**Answer:** Use a **feature branch + PR** even if you’re solo:

* Branch: `feat/m0-foundation`
* Merge via PR into `main`

**Why:** It guarantees CI runs in the same way it will later, and it keeps a clean audit trail from day 1.

---

## 10) Pre-existing files

**Answer:** Proceed with:

* ✅ Keep `VISION.md`, `ProjectFiles/`, `.cursorrules` **as-is**
* ✅ **Overwrite** `README.md` with the M0 quickstart + dev workflows
* ✅ **Update** `tunix-rt.md` at the end of M0 with the M0 contract + runbook

---

## 11) CI strategy for empty sections

**Answer:** **Yes — matches intent.**

* Always-run `changes` job using `dorny/paths-filter`
* Conditional jobs (`backend`, `frontend`, `e2e`) based on outputs
* This avoids merge-blocking weirdness while keeping CI fast.

---

## 12) Documentation in `tunix-rt.md`

**Answer:** Yes, that’s sufficient for M0:

* Endpoints: `/api/health`, `/api/redi/health`
* RediAI mode: `REDIAI_MODE=mock|real`
* Env vars: `REDIAI_BASE_URL`, `REDIAI_HEALTH_PATH`
* Local run instructions (backend, frontend, e2e, compose)

No DB schema needed in M0 (compose DB is just wiring).

---

## 13) License & copyright

**Answer:** Use:

**`Copyright 2025 Michael Cahill`**

(If you later want this under an org name, we can update it in M1 with no disruption.)

---

### Consolidated M0 decisions (final)

* **M0 minimal**; enhancements deferred to **M1+**
* Python **3.11 dev**, CI **3.11–3.12**
* Coverage gate **70%** in M0
* RediAI integration included as **health probe**, **mock in CI**
* **No deployment** in M0
* **Feature branch + PR**
* Overwrite README; update `tunix-rt.md` after M0
* CI uses **paths-filter** with conditional jobs

If you want, I can now re-emit the **single Cursor prompt** with these decisions “locked in” (branch name, python matrix, copyright line, etc.).
