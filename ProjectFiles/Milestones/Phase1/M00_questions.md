# M0 Implementation - Clarifying Questions

## 1. **Scope & Enhancement Integration**

**Question:** Should M0 be **minimal** (as described in M00_plan.md) or should it include **all enhancements** from the three enhancement prompts?

- **M00_plan.md** describes a bare-minimum stack: health endpoints, basic UI, 70% coverage gate
- **Enhancement prompts** require: 85% coverage, OpenTelemetry, SBOM, Scorecard, SLSA provenance, mutation testing, etc.

**Proposed approach:** 
- M0 = Minimal foundation (M00_plan.md as written)
- M1+ = Layer in enhancements incrementally (AuditEnhancementsV2, EnhancementsV2, TestingEnhancementsV2)

**Do you agree, or should M0 be feature-complete with all enhancements?**

---

## 2. **Python Version**

**Question:** Which Python version should we target?

- M00_plan.md: Not specified
- AuditEnhancementsV2.md: Python 3.10-3.12 matrix
- EnhancementsV2.md: Python 3.11+
- TestingEnhancementsV2.md: Python 3.11+

**Proposed:** Use **Python 3.11+** as primary, test against **3.11-3.12** in CI (simpler than 3.10-3.12 range).

---

## 3. **Coverage Targets for M0**

**Question:** What coverage gate for M0?

- M00_plan.md Phase M0.2: 70% for M0
- All enhancement prompts: 85% line + branch

**Proposed:** 
- M0: **70%** (get infrastructure working)
- M1: Raise to **85%** with comprehensive test suite

---

## 4. **RediAI Integration - CRITICAL CLARIFICATION NEEDED**

**Context:** You've shared docs for two existing projects:
1. **RediAI** (`rediai.md`): Comprehensive AI framework with game theory, RL, XAI, workflow registry, etc.
2. **RediWrapper** (`RediWrapper-README.md`): Game wrapper API (Malmo, Chess, Poker)

**Question:** How does the NEW `tunix-rt` project relate to these existing projects?

**Possible interpretations:**

**Option A - Fresh Project (Most Likely):**
- `tunix-rt` is a NEW, standalone project for the Tunix Hackathon
- The "RediAI" mentioned in M00_plan.md is actually a **typo/naming confusion**
- We should build a simple trace storage/reasoning service instead
- M0 health endpoints should be renamed to something like `/api/trace/health` or `/api/tunix/health`

**Option B - Integration Project:**
- `tunix-rt` integrates WITH the existing RediAI framework
- M0's `/api/redi/health` actually probes your running RediAI instance
- The Tunix Hackathon submission will use RediAI's workflow registry for trace management
- We're building a thin layer on top of RediAI

**Option C - Fork/Variant:**
- `tunix-rt` is a lightweight fork of RediAI specifically for the hackathon
- We use some RediAI concepts but build a simpler, focused system

**ANSWER (from user):**
- ✅ **Option B - Integration Project**
- `tunix-rt` is a NEW project that integrates with RediAI
- RediAI is currently running locally
- We need to validate end-to-end integration
- M00_plan.md is correct as written

**Implementation approach:**
- Build tunix-rt as a standalone full-stack app
- Include RediAI client for health probes in M0
- Support mock mode (CI) and real mode (local dev)
- Later milestones will add deeper RediAI integration (workflow registry, trace management)

---

## 5. **Deployment Targets**

**Question:** When should we set up deployment?

- M00_plan.md: No deployment mentioned
- EnhancementsV2.md + TestingEnhancementsV2.md: Deploy to Netlify (frontend) + Render (backend) with preview deployments

**Proposed:**
- M0: No deployment (local + CI only)
- M1 or M2: Add Netlify/Render deployment with preview flows

---

## 6. **Repository Structure**

**Question:** Confirm final directory layout for M0:

```
tunix-rt/
  LICENSE (Apache-2.0)
  README.md
  .gitignore
  .editorconfig
  .env.example
  
  backend/
    pyproject.toml
    ruff.toml (or in pyproject.toml)
    mypy.ini (or in pyproject.toml)
    tunix_rt_backend/
      __init__.py
      app.py
      redi_client.py
      settings.py
    tests/
      test_health.py
      test_redi_health.py
  
  frontend/
    package.json
    package-lock.json
    vite.config.ts
    tsconfig.json
    src/
      main.tsx
      App.tsx
    tests/
      App.test.tsx
  
  e2e/
    package.json
    playwright.config.ts
    tests/
      smoke.spec.ts
  
  docker-compose.yml
  
  .github/workflows/
    ci.yml
  
  ProjectFiles/  (documentation, prompts, logs - keep as is)
  VISION.md
  tunix-rt.md
```

**Is this structure correct?**

---

## 7. **Technology Stack Confirmation**

**Confirmed from M00_plan.md:**
- Backend: FastAPI, PostgreSQL (via docker-compose), httpx (for RediAI client)
- Frontend: Vite, React, TypeScript, Vitest + React Testing Library
- E2E: Playwright
- Package manager: **npm** (not yarn/pnpm)
- CI: GitHub Actions with dorny/paths-filter for conditional jobs

**Any changes needed?**

---

## 8. **Conventional Commits**

**Question:** Should we use strict Conventional Commits format for all commits?

M00_plan.md mentions "Conventional Commits per phase."

**Proposed format:**
```
feat(backend): add health endpoints with dependency injection
test(backend): add deterministic tests for RediAI health probe
chore(ci): add GitHub Actions workflow with path filtering
docs: update README with quickstart instructions
```

**Confirmed?**

---

## 9. **Git Workflow**

**Question:** How should we commit and push?

**Proposed:**
- Work on `main` branch directly (since repo is empty and you're sole dev)
- OR: Create feature branch `feat/m0-foundation` and merge via PR
- Small, atomic commits per M00_plan.md phase (M0.1, M0.2, etc.)

**What's your preference?**

---

## 10. **Pre-existing Files**

**Question:** What should we do with existing files?

Current repo has:
- `README.md` (minimal)
- `tunix-rt.md` (minimal)
- `VISION.md` (keep)
- `ProjectFiles/` (keep)
- `.cursorrules` (keep)

**Proposed:**
- Overwrite `README.md` with comprehensive M0 quickstart
- Update `tunix-rt.md` as we add features (per .cursorrules)
- Leave `VISION.md` and `ProjectFiles/` untouched

**Confirmed?**

---

## 11. **CI Strategy for Empty Sections**

**Question:** M00_plan.md mentions using `dorny/paths-filter` to avoid the "required checks + path filters" merge-block issue.

**Proposed CI flow:**
1. Job `changes` runs always (uses dorny/paths-filter)
2. Conditional jobs: `backend`, `frontend`, `e2e` (run based on changes output)
3. When touching README only, only `changes` job runs (workflow still completes cleanly)

**This matches your intent?**

---

## 12. **Documentation in tunix-rt.md**

**Question:** Per .cursorrules: "After adding a major feature or completing a milestone, update tunix-rt.md."

**Proposed M0 update to tunix-rt.md:**
- Document database schema (if any for M0 - probably none, just health endpoints)
- Document RediAI integration (mock vs real modes)
- Document API endpoints (/api/health, /api/redi/health)
- Document local development setup

**Is this sufficient, or do you want more detail?**

---

## 13. **License & Copyright**

**Question:** License confirmed as Apache-2.0. Should we include a copyright holder?

**Proposed LICENSE header:**
```
Copyright [year] [your name or organization]

Licensed under the Apache License, Version 2.0...
```

**What should the copyright line be?**

---

## Summary & Decisions

Based on your clarification and the project context, here are the **finalized decisions for M0**:

### Confirmed Decisions

1. **Scope**: Minimal M0 per M00_plan.md, enhancements deferred to M1+
2. **Python Version**: 3.11+ (test against 3.11-3.12 in CI)
3. **Coverage Target**: 70% for M0 (raise to 85% in M1)
4. **RediAI Integration**: 
   - Real: `tunix-rt` calls locally-running RediAI
   - Mock: CI uses mock client (no RediAI dependency)
   - Endpoint: Probe RediAI's health endpoint (likely `/health` or `/api/health`)
5. **Deployment**: Not in M0 (add in M1/M2)
6. **Package Manager**: npm (as specified in M00_plan.md)
7. **Conventional Commits**: Yes, use strict format
8. **Git Workflow**: Work on `main` branch directly (solo dev, empty repo)
9. **Existing Files**: Overwrite README.md, update tunix-rt.md after M0
10. **License**: Apache-2.0, copyright holder TBD (can add later)

### Repository Structure (Confirmed)

```
tunix-rt/
  LICENSE (Apache-2.0)
  README.md
  .gitignore
  .editorconfig
  .env.example
  
  backend/
    pyproject.toml
    tunix_rt_backend/
      __init__.py
      app.py
      redi_client.py
      settings.py
    tests/
      test_health.py
      test_redi_health.py
  
  frontend/
    package.json
    package-lock.json
    vite.config.ts
    tsconfig.json
    src/
      main.tsx
      App.tsx
    tests/
      App.test.tsx
  
  e2e/
    package.json
    playwright.config.ts
    tests/
      smoke.spec.ts
  
  docker-compose.yml
  
  .github/workflows/
    ci.yml
  
  ProjectFiles/  (keep as is)
  VISION.md      (keep as is)
  tunix-rt.md    (update after M0)
```

### Next Steps

**I will now:**
1. ✅ Create detailed M0 todo list (6 phases from M00_plan.md)
2. ✅ Begin systematic implementation
3. ✅ Update tunix-rt.md after completing M0

**Ready to begin implementation!** 🚀
