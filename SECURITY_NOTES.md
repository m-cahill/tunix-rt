# Security Notes

## Current Security Posture

This document tracks known security findings and remediation plans for the tunix-rt project.

**Last Updated:** M1 Milestone (2025-12-20)

---

## Known Vulnerabilities

### Frontend (npm audit)

**Status:** 4 moderate severity vulnerabilities (dev dependencies only)

**Summary:**
- **esbuild** (≤0.24.2): GHSA-67mh-4wv8-2f99 - Development server can respond to any website requests
  - CVSS Score: 5.3 (Moderate)
  - Impact: Development-only issue, does not affect production builds
  - Fix Available: Requires major version update to vite 7.x

- **vite** (0.11.0 - 6.1.6): Indirect dependency via esbuild
  - Impact: Development-only dependency
  - Fix Available: Requires major version update to vite 7.x

- **vite-node** (≤2.2.0-beta.2): Indirect dependency via vite
  - Impact: Development-only dependency (vitest)
  - Fix Available: Requires major version update to vitest 4.x

- **vitest** (multiple ranges): Direct dev dependency
  - Impact: Test framework, not used in production
  - Fix Available: Requires major version update to vitest 4.x

**Assessment:**
- **Risk Level:** LOW
- **Rationale:** All vulnerabilities are in development dependencies only
- **Production Impact:** NONE - These packages are not bundled in production builds
- **Development Impact:** Minor - Affects local dev server security only

**Remediation Plan:**
- **M1:** Document findings (this file), add npm audit to CI (warn-only)
- **M2:** Evaluate major version updates for vite/vitest
- **M2:** Run `npm audit fix --force` if safe, or manually update packages
- **M2:** Test thoroughly after updates to ensure compatibility

---

### Backend (pip-audit)

**Status:** No known vulnerabilities

**Last Scan:** M1 Milestone
**Result:** All dependencies clean

---

## Security Scanning

### Automated Scanning (CI)

M1 introduces automated security scanning in the CI pipeline:

1. **pip-audit** (Backend)
   - Frequency: Every PR/push to main
   - Mode: Warn-only (non-blocking)
   - Artifacts: JSON reports uploaded for 30 days
   - Transition to blocking: M2 (High/Critical only)

2. **npm audit** (Frontend)
   - Frequency: Every PR/push to main
   - Mode: Warn-only (non-blocking)
   - Artifacts: JSON reports uploaded for 30 days
   - Transition to blocking: M2 (High/Critical only)

3. **Gitleaks** (Secret Scanning)
   - Frequency: Every commit
   - Mode: **BLOCKING** (prevents merge if secrets detected)
   - Scope: Full git history
   - Purpose: Prevent credential leaks

4. **SBOM Generation** (Backend)
   - Frequency: Every PR/push to main
   - Format: CycloneDX JSON
   - Artifacts: Stored for 90 days
   - Purpose: Supply chain transparency

### Dependabot

Configuration: `.github/dependabot.yml`

- **Backend (pip):** Weekly updates, Monday
- **Frontend (npm):** Weekly updates, Monday
- **E2E (npm):** Weekly updates, Monday
- **GitHub Actions:** Weekly updates, Monday

**Policy:**
- Major version updates: Ignored (to reduce churn)
- Minor/patch updates: Auto-PR created
- Security updates: Always included (override major version policy)
- Auto-merge: Disabled (manual review required)

---

## Security Best Practices

### Development

1. **Never commit secrets** - Use environment variables
2. **Run `npm audit`** before committing frontend changes
3. **Run `pip-audit`** before committing backend changes
4. **Review Dependabot PRs** promptly
5. **Test after dependency updates** to ensure compatibility

### Production

1. **Environment Variables** - All secrets via env vars (never hardcoded)
2. **CORS Configuration** - Restricted to known origins (currently localhost:5173)
3. **Timeout Settings** - All external requests have timeouts (5s for RediAI)
4. **Input Validation** - Pydantic validators enforce configuration correctness
5. **Fail-Fast** - Invalid configuration stops application startup

---

## Remediation History

### M1 (2025-12-20)

- ✅ Added pip-audit scanning (warn-only)
- ✅ Added npm audit scanning (warn-only)
- ✅ Added gitleaks secret scanning (blocking)
- ✅ Added SBOM generation (backend)
- ✅ Configured Dependabot (weekly updates)
- ✅ Documented current npm audit findings
- ✅ Added Pydantic validators for settings

### M0 (Initial)

- ✅ No hardcoded secrets
- ✅ Dependency lockfiles (package-lock.json, pyproject.toml)
- ✅ CORS configuration
- ✅ Request timeouts

---

## Contact

For security concerns or to report vulnerabilities, please:

1. **Do not** create public GitHub issues
2. Contact the maintainer directly
3. Allow reasonable time for remediation before disclosure

---

## References

- [pip-audit documentation](https://github.com/pypa/pip-audit)
- [npm audit documentation](https://docs.npmjs.com/cli/v9/commands/npm-audit)
- [Gitleaks documentation](https://github.com/gitleaks/gitleaks)
- [CycloneDX SBOM specification](https://cyclonedx.org/)
- [Dependabot documentation](https://docs.github.com/en/code-security/dependabot)

