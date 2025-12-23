# M1 Clarifying Questions

Before implementing M1, I need clarification on the following points:

## 1. Branch Coverage Implementation Priority

The M1 plan targets **70% branch coverage** (currently 0%). Looking at the codebase:
- `app.py`: 1 branch (mock vs real mode selection in `get_redi_client`)
- `redi_client.py`: 2 branches (error handling paths, response status checks)
- `settings.py`: 1 branch (property method)

**Question 1a:** Should I focus on all modules equally, or prioritize critical paths (e.g., `redi_client.py` error handling)?

**Question 1b:** The plan mentions "add tests for error/if-else paths" but current tests already cover some error paths. Should I add more granular tests (e.g., test HTTP 404 vs 500 vs timeout separately)?

## 2. Environment Variable Validation Scope

The plan calls for `pydantic-settings` field validators. Current `settings.py` already uses `pydantic-settings`.

**Question 2a:** What level of validation is expected?
- **Option A (Minimal):** Just type validation + URL format check for `REDIAI_BASE_URL`
- **Option B (Moderate):** Add validators for port ranges (1-65535), enum validation for `REDIAI_MODE` (only "mock" or "real")
- **Option C (Strict):** Add URL reachability checks, DNS resolution, etc.

**Question 2b:** Should invalid configuration:
- Raise exception immediately on import (fail-fast)?
- Log warning and use defaults?
- Return detailed validation error messages?

## 3. Security Scanning Strategy

The plan includes pip-audit, npm audit, gitleaks, and SBOM generation, all initially **warn-only**.

**Question 3a:** When should these transition from warn-only to **blocking**? 
- After M1 completion?
- After a grace period (e.g., 1 sprint)?
- Never block, just report?

**Question 3b:** For npm audit, the project has ~4 moderate vulnerabilities (dev dependencies). Should I:
- Run `npm audit fix` immediately?
- Document them and defer to M2?
- Add to CI as-is and track separately?

## 4. Frontend API Client Architecture

The plan suggests creating `frontend/src/api/client.ts` with typed API functions.

**Question 4a:** Should this client:
- Be a simple wrapper around `fetch` (like current code)?
- Include error handling and retry logic?
- Use a library like `axios` or `ky`?

**Question 4b:** Should I also add TypeScript interfaces for API responses (e.g., `HealthStatus`, `RediHealthResponse`)?

## 5. TTL Caching Implementation

The plan mentions optional TTL caching for `/api/redi/health` (30s cache).

**Question 5a:** Should I implement this as:
- **Option A:** Simple in-memory dict with timestamp (as shown in audit)
- **Option B:** Use `functools.lru_cache` with time-aware wrapper
- **Option C:** Use a library like `cachetools`
- **Option D:** Skip for M1 (marked as optional)

**Question 5b:** Should the cache TTL be configurable via environment variable or hardcoded?

## 6. CI/CD Workflow Organization

The plan includes 14 numbered items across 4 phases. 

**Question 6a:** Do you want:
- **One PR per phase** (e.g., PR1 = Phase 1 with items 1-4)?
- **One PR per numbered item** (14 separate PRs)?
- **Logical groupings** (e.g., PR1 = coverage tests, PR2 = security baseline, PR3 = frontend + docs)?

**Question 6b:** Should I create a feature branch `feat/m1-hardening` and merge sub-PRs into it, then merge to main? Or work directly against main?

## 7. ADR (Architecture Decision Records) Format

The plan calls for 3 ADRs: mock/real integration, CI strategy, coverage strategy.

**Question 7a:** Do you have a preferred ADR template (e.g., MADR, Nygard format)? Or should I use the example from the audit document?

**Question 7b:** Where should ADRs be stored?
- `docs/adr/` directory?
- `ProjectFiles/ADRs/`?
- Root `docs/` directory?

## 8. Makefile vs Cross-Platform Scripts

The plan includes a Makefile for DX (developer experience).

**Question 8a:** Since you're on Windows (PowerShell), should I also provide PowerShell equivalents (`.ps1` scripts) alongside the Makefile, or is Makefile sufficient (via WSL/Git Bash)?

**Question 8b:** Should the Makefile include Docker Compose commands (e.g., `make docker-up`, `make docker-down`)?

## 9. Dependabot Configuration Scope

The plan mentions weekly Dependabot updates for pip and npm.

**Question 9a:** Should Dependabot:
- Auto-merge patch updates (if tests pass)?
- Only create PRs for manual review?
- Group updates by ecosystem?

**Question 9b:** Should I exclude any dependencies from Dependabot (e.g., major version updates)?

## 10. Optional Features Priority

The M1 plan marks several items as "optional":
- Schemathesis contract tests (Phase 4)
- 30s polling in frontend (Phase 3, item 11)
- TTL caching (Phase 3, item 11)

**Question 10:** Should I:
- Implement all optional features if time permits?
- Skip all optional features and focus on core M1 requirements?
- Implement only specific optional features (which ones)?

## 11. Test Coverage Margin Strategy

The audit recommends a 68% gate (with 70% target) to prevent flaky CI.

**Question 11:** Should I:
- Keep the gate at 70% (current value)?
- Lower to 68% as recommended by audit?
- Set different thresholds for line vs branch coverage?

## 12. Branch Coverage Measurement

The current `pyproject.toml` has `branch = true` in `[tool.coverage.run]`, but the report shows 0% branch coverage.

**Question 12a:** Should I verify the branch coverage is actually being measured? The existing config looks correct but may need validation.

**Question 12b:** Do you want branch coverage enforced with a separate threshold (e.g., `--cov-branch-min=70`) or just reported?

## Summary of Decisions Needed

To proceed efficiently, please provide guidance on:

1. **Priority items** (questions 1, 2, 6, 10, 12) - These affect the implementation approach
2. **Nice-to-have items** (questions 3, 5, 7, 8, 9, 11) - I can use reasonable defaults if you prefer

**My Recommendations** (will wait for your confirmation):
- **Q1:** Focus on `redi_client.py` and `app.py` branches first, add granular error tests
- **Q2:** Option B (moderate validation), fail-fast on startup
- **Q3:** Keep warn-only through M1, revisit for M2
- **Q4:** Simple wrapper with TypeScript interfaces
- **Q5:** Option A (simple in-memory), hardcoded 30s TTL
- **Q6:** 3-4 logical PRs, work on feature branch
- **Q7:** Use audit example format, store in `docs/adr/`
- **Q8:** Provide both Makefile and PowerShell scripts
- **Q9:** PRs only, manual review
- **Q10:** Implement TTL caching + polling, skip Schemathesis for M1
- **Q11:** Use 70% for both line and branch
- **Q12:** Verify measurement, enforce with `fail_under` in config

Please confirm or correct these recommendations, and I'll proceed with implementation.
