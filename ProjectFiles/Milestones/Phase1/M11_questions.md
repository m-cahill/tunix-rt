# M11 Clarifying Questions

## Project Analysis Summary

I've analyzed your codebase and the M11 plan. Here's what I found:

**Current State:**
- ✅ Backend: 90% line coverage, 88% branch, 132 tests passing
- ✅ Service layer exists with 2 services: `traces_batch.py`, `datasets_export.py`
- ✅ `app.py` currently 741 lines (target: <600 lines)
- ❌ No `.pre-commit-config.yaml` exists
- ❌ SBOM generation disabled (lines 193-207 in ci.yml)
- ❌ GitHub Actions use tag pinning (@v4) not SHA pinning
- ❌ Training scripts have no tests (train_sft_tunix.py, eval_*.py)
- ⚠️  Frontend coverage ~60% (target: 70%)
- ⚠️  4 moderate npm audit findings (dev-only, Vite/Vitest related)

**M11 Scope (from plan):**
- **Phase 0**: Baseline doc
- **Phase 1**: Fix-first (SBOM, pin Actions, pre-commit)
- **Phase 2**: Docs (ADR-006, TRAINING_PRODUCTION.md, PERFORMANCE_SLOs.md)
- **Phase 3**: Complete app extraction (UNGAR + dataset build to services)
- **Phase 4**: Training script dry-run smoke tests
- **Phase 5-6**: Optional (frontend coverage + Vite/Vitest upgrade)

---

## Questions

### 1. **Risk Appetite for Optional Phases (5-6)**

The plan marks Phase 5 (frontend coverage to 70%) and Phase 6 (Vite/Vitest upgrade) as **optional**.

**Question:** Should I:
- **A)** Complete all 6 phases including optionals (higher risk on Vite upgrade)
- **B)** Complete mandatory phases (0-4), then attempt Phase 5, skip Phase 6
- **C)** Complete mandatory phases only (0-4), defer frontend work to M12

**Context:** 
- Vite 7 / Vitest 4 upgrade is marked "High Risk" in the audit
- Frontend npm audit findings are dev-only (CVSS 5.3)
- Phase 5 is lower risk (just adding tests)

**My recommendation:** Option B (complete mandatory + Phase 5, defer Vite upgrade to separate focused effort)

---

### 2. **Training Script Test Strategy**

The M11 plan says: "add `--dry-run` and test via subprocess" (line 223).

**Question:** Should the training script tests be:
- **A)** Subprocess-based smoke tests (as stated in plan): Call scripts via `subprocess.run([..., "--dry-run"])` and check exit codes
- **B)** Unit tests with mocks: Import training functions and mock Tunix API calls
- **C)** Both: Smoke tests for CLI interface + unit tests for core functions

**Context:**
- Subprocess tests are faster to implement but provide less granular coverage
- Unit tests require refactoring scripts to extract testable functions
- Current training scripts are in `training/` directory (outside backend package)

**My recommendation:** Option A initially (subprocess smoke tests as planned), with refactoring for unit tests deferred to future milestone.

---

### 3. **Priority Order (If Time Constrained)**

If we run into time constraints, what's your preferred priority order for completing phases?

**Proposed priority:**
1. **Phase 0** (baseline) - mandatory
2. **Phase 1** (security fixes) - highest value for production readiness
3. **Phase 3** (app extraction) - core architectural goal
4. **Phase 4** (training tests) - explicitly deferred from M10
5. **Phase 2** (docs) - important but not blocking
6. **Phase 5** (frontend coverage) - nice-to-have
7. **Phase 6** (Vite upgrade) - can be separate PR

**Question:** Does this priority align with your goals, or would you reorder?

---

### 4. **Branch Strategy**

The M11 plan mentions creating a `m11-stabilize` branch (line 35).

**Question:**
- **A)** Create `m11-stabilize` branch and work there (requires PR at end)
- **B)** Work directly on `main` with atomic commits (your typical workflow based on M10)
- **C)** Create branch but merge frequently (small PRs per phase)

**My recommendation:** Option B (atomic commits to main) to match your M10 workflow, unless you need a formal PR review process.

---

### 5. **Documentation Update Timing**

The plan has "update tunix-rt.md" as the final commit (line 291).

**Question:**
- **A)** Update `tunix-rt.md` only at the end (per plan)
- **B)** Update after each major phase for incremental tracking
- **C)** Update twice: once after Phase 3 (architecture changes), once at end (summary)

**My recommendation:** Option A (end only) to keep diffs clean and avoid merge conflicts during active work.

---

### 6. **SBOM Tool Preference**

The audit recommends using `cyclonedx-py` CLI (not the Python module invocation that failed in M4).

**Question:** For re-enabling SBOM generation, should I:
- **A)** Use `cyclonedx-py` as recommended in audit
- **B)** Investigate and fix the original `cyclonedx-bom` module invocation
- **C)** Use a different SBOM tool (e.g., `syft`, `trivy`)

**My recommendation:** Option A (cyclonedx-py CLI) for fastest resolution.

---

### 7. **GitHub Actions SHA Pinning Tool**

The audit mentions using `pin-github-action` tool or Dependabot.

**Question:** Should I:
- **A)** Manually pin to SHAs and set up Dependabot config for auto-updates
- **B)** Use a tool like `pin-github-action` to automate the initial pinning
- **C)** Pin manually without Dependabot (manual updates going forward)

**My recommendation:** Option A (manual pin + Dependabot) for best security/maintenance balance.

---

### 8. **Training Script `--dry-run` Implementation**

**Question:** Should the `--dry-run` flag:
- **A)** Validate config, compute paths, then exit 0 (no actual training calls)
- **B)** Validate config, initialize models/datasets (but not train), then exit 0
- **C)** Do everything except the actual training loop (more comprehensive validation)

**Context:** More validation = slower dry-run but catches more issues.

**My recommendation:** Option A (lightweight validation only) to keep tests fast and deterministic.

---

### 9. **Coverage Gate Adjustment**

The current backend coverage is 90% line / 88% branch. After refactoring, coverage might dip slightly.

**Question:** Should I:
- **A)** Keep gates at current levels (80% line, 68% branch)
- **B)** Raise gates to match current coverage (90% line, 88% branch)
- **C)** Keep gates but add margin as audit suggests (82% line gate with 90% current)

**My recommendation:** Option A (keep existing gates) to avoid making future work harder.

---

### 10. **Commit Granularity**

The plan suggests 10 commits (line 280-291). 

**Question:** Should I:
- **A)** Follow the exact commit sequence in the plan (10 commits)
- **B)** Use more atomic commits (one per task, ~15-20 commits)
- **C)** Consolidate into fewer commits per phase (~5-6 commits)

**My recommendation:** Option B (atomic commits) for better rollback granularity and easier debugging.

---

## Ready to Proceed

Once you answer these questions, I'll:
1. Create a detailed TODO list with your preferences
2. Start with Phase 0 (baseline documentation)
3. Work through phases systematically
4. Keep you updated on progress

**Estimated total time:** 2-3 days of focused work (per audit: 19.5 hours)

---

**Please respond with your preferences for each question (can be just "1A, 2A, 3-yes, 4B..." format).**
