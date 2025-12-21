# ADR-002: CI Conditional Jobs Strategy with Path Filtering

**Status:** Accepted

**Date:** 2025-12-20 (M1 Milestone)

**Context:**

tunix-rt is a monorepo with multiple components:
- Backend (Python/FastAPI)
- Frontend (TypeScript/React/Vite)
- E2E tests (Playwright)
- Documentation (Markdown)

**Challenges:**

1. **Wasted CI Time**: Running backend tests when only README changed
2. **Merge Blocking**: GitHub required status checks + path filters can block PRs when all jobs skip
3. **Developer Experience**: Slow CI feedback wastes developer time
4. **Cost**: Unnecessary job runs increase CI minutes usage

**Decision:**

Implement **conditional jobs with path filtering** using `dorny/paths-filter@v2`:

1. **Always-Run Filter Job**:
   ```yaml
   jobs:
     changes:
       runs-on: ubuntu-latest
       outputs:
         backend: ${{ steps.filter.outputs.backend }}
         frontend: ${{ steps.filter.outputs.frontend }}
         # ... other outputs
   ```

2. **Conditional Downstream Jobs**:
   ```yaml
   backend:
     needs: changes
     if: needs.changes.outputs.backend == 'true' || needs.changes.outputs.workflow == 'true'
     # ... job steps
   ```

3. **Path Filters Configuration**:
   ```yaml
   filters: |
     backend:
       - 'backend/**'
     frontend:
       - 'frontend/**'
     workflow:
       - '.github/workflows/**'
   ```

4. **E2E Runs on Any Code Change**:
   - Triggers if backend, frontend, e2e, or workflow files change
   - Ensures integration testing coverage

**Consequences:**

### Positive

- ✅ **Fast CI**: Backend tests skip when only frontend changes (and vice versa)
- ✅ **No Merge Blocking**: `changes` job always runs, provides stable required check
- ✅ **Cost Savings**: Estimated 40-60% reduction in unnecessary job runs
- ✅ **Clear Intent**: Path filters document component boundaries
- ✅ **Flexible**: Easy to add new components or adjust paths

### Negative

- ⚠️ **Complexity**: More complex than "run everything" approach
  - **Assessment**: Worth it - path filters are well-understood pattern
- ⚠️ **Maintenance**: Need to update filters when adding new components
  - **Mitigation**: Document in this ADR and in ci.yml comments

### Behavior Matrix

| Change Type | Backend Job | Frontend Job | E2E Job | Changes Job |
|------------|-------------|--------------|---------|-------------|
| `backend/**` only | ✅ Runs | ⏭️ Skips | ✅ Runs | ✅ Runs |
| `frontend/**` only | ⏭️ Skips | ✅ Runs | ✅ Runs | ✅ Runs |
| `README.md` only | ⏭️ Skips | ⏭️ Skips | ⏭️ Skips | ✅ Runs |
| `backend/**` + `frontend/**` | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |
| `.github/workflows/**` | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |

**Why dorny/paths-filter?**

- Industry standard (16k+ stars, widely used)
- Solves the "required check + skip = blocked PR" problem
- Supports glob patterns and multiple outputs
- Well-maintained and documented

**Alternatives Considered:**

1. **GitHub paths filter (built-in)**:
   - Rejected: Causes PR blocking when jobs skip but are required
   - GitHub issue: https://github.com/actions/runner/issues/491

2. **Run everything always**:
   - Rejected: Wastes CI time and money
   - Rejected: Slow feedback for developers

3. **Separate workflows per component**:
   - Rejected: Hard to coordinate required checks
   - Rejected: Duplicate setup steps across workflows

4. **Nx/Turborepo-style caching**:
   - Rejected: Overkill for current project size
   - Reconsidered if: Project grows to 10+ components

**Implementation Details:**

### Security Jobs Exception

Security jobs (pip-audit, npm audit, gitleaks) follow different rules:
- **gitleaks**: Always runs (scans entire repo, no path filtering)
- **security-backend**: Runs if `backend/**` or workflow changes
- **security-frontend**: Runs if `frontend/**` or workflow changes

Rationale: Security scans are fast (<30s) and catching vulnerabilities early is critical.

### Caching Strategy

All jobs use GitHub Actions caching:
- **pip**: Caches based on `backend/pyproject.toml`
- **npm**: Caches based on `**/package-lock.json`

Cache invalidation is automatic when lockfiles change.

**Review:**

This strategy should be reviewed if:
- Adding 5+ new components (consider Nx/Turborepo)
- CI time becomes problematic again (profile and optimize)
- GitHub fixes the required-check-skip-blocking issue

**References:**

- dorny/paths-filter: https://github.com/dorny/paths-filter
- GitHub Actions conditional jobs: https://docs.github.com/en/actions/using-jobs/using-conditions-to-control-job-execution
- GitHub issue on path filtering: https://github.com/actions/runner/issues/491

