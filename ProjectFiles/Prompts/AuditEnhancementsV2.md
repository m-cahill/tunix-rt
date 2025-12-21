You are an implementation agent. Apply these enhancements to the current GitHub repo in four small phases with end-to-end verification at each phase. Use GitHub Actions (ubuntu-latest), Python 3.10–3.12 matrix, and create minimal, auditable artifacts. Follow all acceptance criteria exactly.

===============================================================================
SCOPE & STANDARDS (authoritative baselines)
===============================================================================

- Security frameworks & supply chain:
  
  - Map repo practices to NIST SSDF SP 800-218 (include a short mapping table). 
  - If ML/AI code exists, link to SSDF 800‑218A profile note. 
  - Use OWASP ASVS v5.0 L2 as the web/API security reference when applicable. 
  - Add OpenSSF Scorecard workflow (warn-first; see Phase 3), SLSA provenance attestations for built artifacts, and GitHub Dependency Review on PRs.

- Core quality tools (Python):
  
  - Lint/format via **Ruff** (single tool for lint + formatter + import sorting).
  - Static typing via **mypy** (strict on repo; fail build on type errors).
  - Docstrings via **pydocstyle**.
  - Complexity via **radon** (fail at CC grade > C).
  - Tests via **pytest** + **coverage.py** (fail-under: 85% lines, 80% branches).
  - Property-based testing via **Hypothesis** (library available; not a gate).
  - Security: **Bandit** (fail on >= HIGH), **pip-audit** (strict), **Gitleaks** (fail if secrets found).
  - SBOM: **CycloneDX** for Python.
  - Containers (if a Dockerfile exists): **Trivy** image scan (fail on HIGH/CRITICAL), **Cosign** keyless sign images.

- CI evidence & dashboards:
  
  - GitHub Actions **job summary** for a one‑page audit report each run.
  - Publish Sphinx docs + “QA & Evidence” page to GitHub Pages.
  - Upload machine‑readable artifacts (coverage.xml, radon, bandit, pip‑audit, gitleaks, SBOM, Scorecard JSON) on each run.

===============================================================================
PHASE 0: PRE-FLIGHT (idempotent detection)
===============================================================================

- Detect repo layout (package root, tests/, presence of Dockerfile).
- If no Python packaging is found, do not alter build backends; only add tool config files.
- Add .gitignore entries for common artifacts if missing.

===============================================================================
PHASE 1: LINT, TYPES, DOCSTRINGS, PRE-COMMIT (end-to-end)
===============================================================================
Create/update the following files (merge without clobbering existing config):

1) `pyproject.toml` (add only [tool.*] blocks if file exists):
   [tool.ruff]
   line-length = 100
   target-version = "py310"
   fix = true
   extend-select = ["I"]                 # import sorting
   [tool.ruff.lint]
   
   # keep defaults; can add ignores later if needed
   
   [tool.mypy]
   python_version = "3.10"
   check_untyped_defs = true
   disallow_incomplete_defs = true
   disallow_untyped_defs = true
   warn_unused_ignores = true
   no_implicit_optional = true
   strict_equality = true

2) `.pydocstyle` (pydocstyle uses ini-like files):
   [pydocstyle]
   convention = google
   add-ignore = D105,D107

3) `.pre-commit-config.yaml` (pin hook revs by resolving latest stable tags at run time, then write them):
   repos:
   
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: <pin-latest>          # agent resolves and pins exact tag
     hooks:
     - id: ruff               # lint
       args: ["--fix"]
     - id: ruff-format        # formatter
   - repo: https://github.com/pre-commit/mirrors-mypy
     rev: <pin-latest>
     hooks:
     - id: mypy
   - repo: local                # Gitleaks as a local hook
     hooks:
     - id: gitleaks
       name: gitleaks
       entry: gitleaks protect --staged --redact
       language: system
       pass_filenames: false
   - repo: https://github.com/PyCQA/pydocstyle
     rev: <pin-latest>
     hooks:
     - id: pydocstyle

4) `requirements-dev.in` (top-level tools; we will compile a pinned lock):
   ruff
   mypy
   pytest
   pytest-cov
   hypothesis
   coverage
   bandit
   pip-audit
   pip-licenses
   radon
   pydocstyle
   sphinx
   furo
   cyclonedx-bom
   gitleaks
   pip-tools

5) Compile pinned lockfile:
   
   - Run: `python -m pip install -U pip pip-tools`
   - Run: `pip-compile --generate-hashes -o requirements-dev.txt requirements-dev.in`
   - Commit both `.in` and pinned `requirements-dev.txt`.

6) Add `CODEOWNERS` (if absent):
   
   * @your-org/maintainers
     docs/** @your-org/docs
     .github/** @your-org/infra

7) Run locally in CI to verify:
   
   - `pre-commit install`
   - `pre-commit run --all-files`

===============================================================================
PHASE 2: TESTS, COVERAGE & COMPLEXITY (end-to-end)
===============================================================================
Files:

1) `.coveragerc`
   [run]
   branch = True
   source = .
   [report]
   fail_under = 85
   omit =
     */tests/*
     **/__init__.py

2) Add a safety net test if the repo has zero tests:
   
   - Create `tests/test_sanity.py` with a trivial test so CI proves the wiring.
   - Do not overfit; existing tests remain source of truth.

3) GH Actions workflow `.github/workflows/ci.yml`
   name: CI
   on:
     pull_request:
     push: { branches: [ main ] }
   permissions:
     contents: read
     security-events: write
     attestations: write
     id-token: write
     pages: write
   jobs:
     build-test:
   
       runs-on: ubuntu-latest
       strategy:
         matrix: { python: [ "3.10", "3.11", "3.12" ] }
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: ${{ matrix.python }}, cache: "pip" }
         - run: python -m pip install -r requirements-dev.txt
         - name: Ruff (lint + format check)
           run: |
             ruff check .
             ruff format --check .
         - name: mypy
           run: mypy .
         - name: pydocstyle
           run: pydocstyle .
         - name: Radon complexity (fail on > C)
           run: |
             radon cc -s -n C .
             # radon exits 0 even if issues printed; enforce failure:
             if [ -n "$(radon cc -s -n C . | tail -n +1)" ]; then
               echo "Complexity > C found"; exit 1; fi
         - name: Pytest + coverage
           run: |
             pytest -q --maxfail=1 --disable-warnings --cov=. --cov-report=xml --cov-report=term-missing
             coverage report --fail-under=85
         - name: Upload coverage.xml
           uses: actions/upload-artifact@v4
           with: { name: coverage-xml, path: coverage.xml }
         - name: CI summary
           run: |
             echo "## Quality gates" >> $GITHUB_STEP_SUMMARY
             echo "- Coverage:" >> $GITHUB_STEP_SUMMARY
             coverage report >> $GITHUB_STEP_SUMMARY

===============================================================================
PHASE 3: SECURITY & SUPPLY CHAIN (end-to-end)
===============================================================================
Augment `ci.yml` with additional jobs:

- `security` job:
  steps:
  
  - checkout + setup-python (as above)
  - install dev reqs
  - name: Bandit
    run: bandit -r . -lll -q -f json -o bandit.json
  - name: pip-audit (strict)
    run: pip-audit -r requirements-dev.txt -f json -o pip_audit.json --strict
  - name: Gitleaks scan (repo working tree)
    run: gitleaks detect -v --redact --report-format json --report-path gitleaks.json
  - name: Upload security artifacts
    uses: actions/upload-artifact@v4
    with: { name: security-reports, path: "*.json" }
  - name: Security summary
    if: always()
    run: |
      echo "## Security" >> $GITHUB_STEP_SUMMARY
      echo "- Bandit, pip-audit, Gitleaks run completed." >> $GITHUB_STEP_SUMMARY

- `dependency-review` job:
  uses: actions/dependency-review-action@v4
  (minimum config to block PRs that introduce vulnerable deps)

- `scorecard` job (warn-first; non-blocking):
  uses: ossf/scorecard-action@v2
  with: { results_file: results.sarif, publish_results: true }
  continue-on-error: true
  Upload SARIF as artifact.

- `sbom` job:
  
  - run: `cyclonedx-bom -o sbom.cdx.json -e`
  - upload artifact.

- `provenance` job (SLSA attestation for build artifacts):
  
  - build a wheel/sdist if the repo is packageable (`python -m build`).
  - `uses: actions/attest-build-provenance@v1` with subject pointing to uploaded artifacts.

- Optional container path (only if `Dockerfile` exists):
  
  - Build image: `docker build -t ${{ github.repository }}:${{ github.sha }} .`
  - Trivy: `trivy image --exit-code 1 --severity HIGH,CRITICAL $IMAGE`
  - Cosign keyless sign: `cosign sign --yes $IMAGE`

===============================================================================
PHASE 4: DOCS & EVIDENCE PUBLISH (end-to-end)
===============================================================================

- Sphinx: if no docs/, bootstrap minimal:
  docs/conf.py + docs/index.rst (or .md)
- Build docs: `sphinx-build -b html docs docs/_build/html`
- Publish docs + a single "QA & Evidence" page (links to artifacts) to GitHub Pages:
  - Upload with actions/upload-pages-artifact@v4
  - Deploy with actions/deploy-pages@v4

===============================================================================
SUPPORTING FILES TO ADD
===============================================================================

- `.github/workflows/dependency-review.yml` (uses actions/dependency-review-action@v4)
- `.github/workflows/scorecard.yml` (ossf/scorecard-action@v2; non-blocking)
- `SECURITY.md` (brief SSDF mapping table + how to report vulnerabilities)
- `docs/qa.md` (collects links to artifacts for auditors)

===============================================================================
BRANCH PROTECTIONS (if admin permissions)
===============================================================================

- Enforce required status checks: CI, dependency-review.
- Require 1–2 approving reviews; enforce CODEOWNERS.
- If not permitted to change settings, open an issue with exact GH CLI commands.

===============================================================================
ACCEPTANCE CRITERIA (hard gates)
===============================================================================

- `ruff`, `mypy`, `pydocstyle` all pass.
- `radon` shows no CC grade > C.
- Test coverage >= 85% lines, >= 80% branches (fail build otherwise).
- `bandit` no HIGH; `pip-audit` passes strict; `gitleaks` passes.
- SBOM (`sbom.cdx.json`) uploaded each run.
- If packageable: build artifacts + SLSA provenance attestation uploaded.
- If Dockerfile exists: Trivy passes; Cosign signature created.
- Sphinx docs + “QA & Evidence” page published to Pages.
- Job summary includes: coverage totals + links to artifacts.

Commit all changes in a single PR titled: 
  "chore(ci): add audit-ready quality & security gates (SSDF/ASVS/SLSA/Scorecard)"
Use Conventional Commits in all messages.

END.
