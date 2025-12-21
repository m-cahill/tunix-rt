Here are the M1 decisions for each question in **M01_questions.md**. 

---

## 1) Branch coverage implementation priority

### 1a) Prioritize or equal focus?

**Prioritize critical paths first**, but we will cover all branches by the end of Phase 1:

1. `redi_client.py` (network/HTTP failure handling is the real integration boundary risk)
2. `app.py` (`REDIAI_MODE` switching + error mapping)
3. `settings.py` (validation branches)

### 1b) How granular should tests be?

**Granular, but not exhaustive.** Add **3 focused tests** that each represents a distinct failure mode:

* **Non-2xx response** (pick one representative, e.g. 404 or 500)
* **Timeout**
* **Connection error**

That’s enough to cover meaningful branches without turning M1 into a taxonomy project.

---

## 2) Environment variable validation scope

### 2a) Validation level

**Option B (Moderate)**:

* URL format validation for `REDIAI_BASE_URL`
* Enum validation for `REDIAI_MODE ∈ {mock, real}`
* Port range checks (1–65535) where relevant
* No DNS/reachability checks in settings validation (those belong to runtime health). Pydantic Settings is exactly for loading/validating env-driven configuration. ([Pydantic][1])

### 2b) Behavior on invalid config

**Fail-fast on startup with clear validation errors** (do not “warn and default”).

* This is an integration project; invalid env should stop boot immediately so failures are loud and early.

---

## 3) Security scanning strategy

### 3a) When to go from warn-only → blocking?

* **M1:** keep **pip-audit** and **npm audit** as **reporting / warn-only**
* **M2:** switch to **blocking for High/Critical** (not Moderate) once you’ve had one sprint to remediate/triage findings.
* **Gitleaks:** **blocking immediately** (secrets are not “warn-only”). Gitleaks is purpose-built for detecting secrets in repos and CI. ([GitHub][2])
* pip-audit is explicitly a vuln scanner for Python deps. ([GitHub][3])

### 3b) npm audit moderate vulns in dev deps

**Do not run `npm audit fix` blindly in M1.**
Instead:

* Add CI as-is (warn-only)
* Create a short `SECURITY_NOTES.md` section listing the current findings + whether they’re dev-only
* Plan remediation in M2, where you can bump packages intentionally and keep diffs reviewable

npm audit behavior (including that it reports vulnerabilities and can apply fixes) is documented by npm. ([npm Docs][4])

---

## 4) Frontend API client architecture

### 4a) Wrapper vs retry vs axios/ky

Use a **simple wrapper around `fetch`** (no axios/ky in M1).
Include:

* consistent JSON parsing
* a small `ApiError` type that captures status + message

Keep it tiny and dependency-free.

### 4b) Add TypeScript interfaces?

**Yes.** Add `HealthResponse` and `RediHealthResponse` interfaces and use them in the client functions.

---

## 5) TTL caching for `/api/redi/health`

### 5a) Implementation choice

Implement **Option A (simple in-memory TTL)** for M1:

* `last_value`, `last_checked_at`, `ttl_seconds`
* If “fresh”, return cached response
* Otherwise call RediAI and refresh cache

This keeps dependency count low. (We can move to `cachetools` later if needed; it provides TTL caches, but it’s extra surface area for M1.) ([cachetools.readthedocs.io][5])

### 5b) TTL configurable?

**Configurable via env var**, default 30 seconds:

* `REDIAI_HEALTH_CACHE_TTL_SECONDS=30`

---

## 6) CI/CD workflow organization

### 6a) PR strategy

Do **3 logical PRs** (fast review, minimal conflicts):

1. **PR1:** Branch coverage measurement + new branch tests + Redi client diagnostics + settings validation
2. **PR2:** Security baseline (pip-audit, npm audit, gitleaks, SBOM, Dependabot)
3. **PR3:** Frontend typed API client + DX (Makefile/scripts) + ADRs (+ optional TTL/polling)

### 6b) Branch strategy

Use **feature branches per PR off `main`**, merge PR-by-PR into `main`.
No long-lived staging branch unless you expect heavy parallel work.

---

## 7) ADR format & location

### 7a) Template

Use **Nygard-style ADRs** (Status / Context / Decision / Consequences).
It’s lightweight and fits M1’s “small but serious” goal.

### 7b) Location

Store ADRs in: **`docs/adr/`** (standard, discoverable).

---

## 8) Makefile vs cross-platform scripts

### 8a) Windows / cross-platform

Provide **both**:

* `Makefile` (nice DX for Mac/Linux/WSL)
* `scripts/*.ps1` PowerShell equivalents (so you’re not forcing WSL/Git Bash)

### 8b) Include docker commands?

**Yes.** Add:

* `make docker-up`, `make docker-down`, `make docker-logs`
  (and the same in PS scripts)

---

## 9) Dependabot configuration

### 9a) Auto-merge?

**No auto-merge in M1.** PRs only, manual review.
Dependabot is configured via `dependabot.yml`, and you can later add grouping rules. ([GitHub Docs][6])

### 9b) Exclusions?

**Ignore major version updates for now** (reduce churn), but allow minor/patch and security updates.
We can loosen this later once the repo stabilizes.

---

## 10) Optional features priority

For M1:

* ✅ Implement **TTL caching** (backend) + **30s polling** (frontend) *if it stays small and well-tested*
* ❌ Skip **Schemathesis** in M1 (add it in M2 as a non-blocking nightly)

(You already have solid E2E; Schemathesis is extra surface area.)

---

## 11) Coverage margin strategy

Set **targets** and **gates** separately:

* **Target branch coverage:** 70%
* **Gate branch coverage:** **68%** (buffer against tiny diffs / rounding)
* **Line coverage gate:** keep **≥ 80%** (you’re already there and it’s stable)

Because pytest-cov’s standard `--cov-fail-under` is a single number, we should enforce **branch% separately** via a small script that reads `coverage json`. Branch coverage measurement itself is enabled via `--cov-branch`. ([pytest-cov][7])

---

## 12) Branch coverage measurement

### 12a) Verify branch coverage is actually measured

**Yes—verify and fix the invocation.**
Even if coverage config has `branch=true`, the simplest reliable move is to ensure CI runs pytest with **`--cov-branch`**. ([pytest-cov][7])

### 12b) Enforce separate thresholds?

**Yes.** Do it with:

* `pytest --cov --cov-branch --cov-report=json:coverage.json ...`
* a tiny `tools/coverage_gate.py` that fails if:

  * `line < 80` or `branch < 68`

This gives you real “enterprise-grade” gates without needing multiple pytest runs.

---

## One extra CI guardrail (re: conditional jobs)

Keep the existing approach: **`dorny/paths-filter` + conditional jobs** so required checks don’t get stuck when paths don’t change. This is a known monorepo pain point and `paths-filter` is designed for conditional job execution. ([GitHub][8])

And for E2E stability: continue using Playwright’s `webServer` to boot local servers in CI. ([Playwright][9])

---

If you want, I can now rewrite the **M1 Cursor handoff prompt** to reflect these locked decisions (PR grouping, coverage gates, strict settings, warn-only security scans, TTL/polling included).

[1]: https://docs.pydantic.dev/latest/concepts/pydantic_settings/?utm_source=chatgpt.com "Settings Management - Pydantic Validation"
[2]: https://github.com/gitleaks/gitleaks?utm_source=chatgpt.com "Find secrets with Gitleaks"
[3]: https://github.com/pypa/pip-audit?utm_source=chatgpt.com "pypa/pip-audit"
[4]: https://docs.npmjs.com/cli/v9/commands/npm-audit/?utm_source=chatgpt.com "npm-audit"
[5]: https://cachetools.readthedocs.io/?utm_source=chatgpt.com "cachetools — Extensible memoizing collections and ..."
[6]: https://docs.github.com/en/code-security/dependabot/working-with-dependabot/dependabot-options-reference?utm_source=chatgpt.com "Dependabot options reference"
[7]: https://pytest-cov.readthedocs.io/en/latest/config.html?utm_source=chatgpt.com "Configuration - pytest-cov 7.0.0 documentation"
[8]: https://github.com/dorny/paths-filter?utm_source=chatgpt.com "Conditionally run actions: files modified by PR, branch, commits"
[9]: https://playwright.dev/docs/test-webserver?utm_source=chatgpt.com "Web server"
