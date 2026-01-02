# Codebase Audit Report (M39)

**Date:** January 1, 2026
**Auditor:** CodeAuditorGPT (via Cursor)
**Target:** Tunix RT (M39 Snapshot)

---

## 1. Executive Summary

**Score: 4.2 / 5.0 (High Health)**

Tunix RT exhibits **mature engineering practices** rarely seen in rapid-iteration repos. The project features a strict 3-tier CI pipeline, enforced coverage gates (>70%), automated security scanning (SBOM/audit), and a clean monorepo structure. The recent M39 pivot to local execution demonstrates adaptability to hardware constraints.

**Strengths:**
1.  **CI/CD Discipline:** GitHub Actions workflow (`ci.yml`) is exemplary, using `paths-filter` for efficiency, explicit dependency locking (`uv.lock`/`package-lock.json`), and coverage enforcement (`coverage_gate.py`).
2.  **Documentation as Truth:** Milestone-driven development (`ProjectFiles/Milestones/`) provides exceptional traceability. Hardware constraints (TPU vs GPU) are explicitly documented (`TRAINING_MODES.md`).
3.  **Modern Stack:** Leveraging `uv` for Python, `Vite` for React, and `Playwright` for E2E ensures a fast, reproducible DX.

**Biggest Opportunities:**
1.  **Hardware Abstraction:** The training logic is split by framework (`train_jax.py` vs `training_pt/train.py`) rather than unified behind an abstraction layer, leading to logic duplication (config parsing, dataset loading).
2.  **Frontend Coverage:** Backend coverage is robust (>75%), but frontend testing remains ~45%, leaving UI logic vulnerable.
3.  **Config Management:** Training configurations are duplicated across `training/configs/` for different hardware targets (TPU/GPU/CPU).

---

## 2. Codebase Map

```mermaid
graph TD
    Root[Repo Root] --> Backend[backend/]
    Root --> Frontend[frontend/]
    Root --> E2E[e2e/]
    Root --> Docs[docs/ & ProjectFiles/]
    
    Backend --> App[tunix_rt_backend/]
    Backend --> Training[training/]
    Backend --> Tests[tests/]
    
    App --> Routers[routers/]
    App --> Services[services/]
    App --> DB[db/ (Models)]
    
    Training --> JAX[train_jax.py]
    Training --> Torch[training_pt/train.py]
    
    Frontend --> Src[src/]
    Src --> Components[components/]
    Src --> API[api/]
```

**Observation:** The structure is clean. The split between `tunix_rt_backend` (app) and `training` (scripts) is logical. M39 introduced `training_pt/` at the root level (parallel to `backend/training` logic?), which is a slight inconsistency in the map.

---

## 3. Modularity & Coupling

**Score: 4/5**

*   **Observation:** Backend uses a clear Router-Service-DB pattern (`backend/tunix_rt_backend/`).
*   **Observation:** Training logic is split by framework (`train_jax.py` vs `training_pt/train.py`).
*   **Interpretation:** While modular, maintaining two parallel training stacks (JAX/PyTorch) risks feature drift. M39 duplicated logic (dataset loading, config parsing) into `training_pt/train.py` to move fast.
*   **Recommendation:** Extract common training utilities (logging, dataset loading, config parsing) into a shared `tunix_rt_backend.training.utils` module usable by both scripts.

---

## 4. Code Quality & Health

**Score: 4.5/5**

*   **Observation:** `pyproject.toml` enforces `ruff` (linting/formatting) and `mypy` (strict mode).
*   **Observation:** CI runs `ruff format --check` and `mypy` on every push (`ci.yml:85-92`).
*   **Evidence:** `.github/workflows/ci.yml` lines 85-92.
*   **Recommendation:** Continue strict enforcement. The pre-commit hooks in `scripts/dev.ps1` are a good DX touch.

---

## 5. Docs & Knowledge

**Score: 5/5**

*   **Observation:** `tunix-rt.md` serves as a comprehensive "living README". `ProjectFiles/Milestones/` logs every major decision.
*   **Observation:** `TRAINING_MODES.md` and `M39_summary.md` explicitly document hardware constraints (TPU/GPU blockers).
*   **Strengths:** The documentation explains *why* (architectural decisions), not just *how*.

---

## 6. Tests & CI/CD Hygiene

**Score: 4.5/5**

*   **Observation:** 3-Tier CI architecture is visible:
    *   **Smoke:** `pytest` / `npm test` on every push.
    *   **Quality:** Coverage gates (`coverage_gate.py`) enforce >70% backend.
    *   **E2E:** `playwright` with Postgres service container (`ci.yml:258`).
*   **Observation:** Dependency caching (`setup-uv`, `setup-node`) is used correctly.
*   **Observation:** `paths-filter` prevents unnecessary runs.
*   **Recommendation:** Ensure frontend coverage gates are as strict as backend (currently ~45% vs 70%).

---

## 7. Security & Supply Chain

**Score: 4/5**

*   **Observation:** `pip-audit` and `npm audit` run in CI (`ci.yml:154`, `ci.yml:209`).
*   **Observation:** `gitleaks` runs on push (`ci.yml:243`).
*   **Observation:** SBOM generation via `cyclonedx-py` (`ci.yml:197`).
*   **Observation:** `uv.lock` and `package-lock.json` ensure reproducible builds.
*   **Recommendation:** Make security audits blocking (currently `continue-on-error: true`).

---

## 8. Performance & Scalability

**Score: 3.5/5**

*   **Observation:** Async SQLAlchemy + FastAPI handles concurrency well.
*   **Observation:** Training performance is bound by hardware support (RTX 5090 blocker).
*   **Recommendation:** Once PyTorch supports sm_120, implement gradient checkpointing and compiled models (`torch.compile`) to maximize RTX 5090 throughput.

---

## 9. Developer Experience (DX)

**Score: 4/5**

*   **Observation:** `Makefile` and `scripts/dev.ps1` provide cross-platform convenience.
*   **Observation:** Local dev requires Docker + Python + Node.
*   **Pain Point:** Hardware fragmentation (TPU vs GPU vs CPU) makes "it works on my machine" harder for training.
*   **Recommendation:** Create explicit `requirements-gpu.txt` vs `requirements-cpu.txt` to streamline the PyTorch/JAX setup conflicts documented in M39.

---

## 10. Phased Plan (M40 Preparation)

### Phase 0: Stabilize & Unify (Immediate)
| ID | Milestone | Criteria | Risk | Est | Owner |
|---|---|---|---|---|---|
| M40-01 | Shared Training Utils | Extract common config/logging from `train_jax.py` & `training_pt/train.py` | Low | 2h | Backend |
| M40-02 | Frontend Coverage Uplift | Add tests for `LiveLogs.tsx` & `Leaderboard.tsx` to reach 60% | Low | 4h | Frontend |

### Phase 1: Hardware Readiness (When Ecosystem Catches Up)
| ID | Milestone | Criteria | Risk | Est | Owner |
|---|---|---|---|---|---|
| M40-03 | PyTorch 2.7/Nightly Upgrade | Update `pyproject.toml` when sm_120 wheels land | Med | 1h | Ops |
| M40-04 | RTX 5090 Benchmark | Run full fine-tuning on local GPU | Med | 4h | Ops |

### Phase 2: Production Polish
| ID | Milestone | Criteria | Risk | Est | Owner |
|---|---|---|---|---|---|
| M40-05 | Strict Security | Make `pip-audit` blocking in CI | Low | 1h | Sec |
| M40-06 | Final E2E | Full trace-to-training-to-eval flow test | Med | 4h | QA |

---

## 11. Machine-Readable Appendix

```json
{
  "scores": {
    "architecture": 4,
    "modularity": 4,
    "code_health": 4.5,
    "tests_ci": 4.5,
    "security": 4,
    "performance": 3.5,
    "dx": 4,
    "docs": 5,
    "overall_weighted": 4.2
  },
  "issues": [
    {
      "id": "DUP-001",
      "title": "Duplicate Training Logic",
      "category": "modularity",
      "path": "training_pt/train.py",
      "severity": "medium",
      "priority": "medium",
      "evidence": "Dataset loading and config parsing duplicated between JAX and PT scripts",
      "fix_hint": "Extract to backend/tunix_rt_backend/training/utils.py"
    },
    {
      "id": "HW-001",
      "title": "RTX 5090 Incompatibility",
      "category": "compatibility",
      "path": "backend/pyproject.toml",
      "severity": "high",
      "priority": "high",
      "evidence": "PyTorch Nightly lacks sm_120 support",
      "fix_hint": "Wait for upstream wheels or build from source"
    }
  ]
}
```
