# M45 Milestone Audit — Curriculum Reasoning Training

**Audit Date:** 2026-01-09  
**Auditor:** CodeAuditorGPT  
**Delta:** M44 (4629ae2) → M45 (pending commit)  
**Phase:** 5 (Exploratory Research)

---

## 1. Delta Executive Summary (≤7 bullets)

### Strengths
- ✅ **Clean research isolation**: All M45 artifacts in `research/m45_curriculum_reasoning/` — zero contamination of M42 submission
- ✅ **Comprehensive provenance**: `provenance.json` captures all hashes, configs, and reproducibility metadata
- ✅ **Evidence-backed hypothesis**: Qualitative trace analysis with concrete examples, not speculation

### Risks/Opportunities
- ⚠️ **Large binary artifacts**: Model checkpoints (~5GB each) in `research/` may bloat repo
- ⚠️ **No `.gitignore` update**: Checkpoint binaries should likely be excluded from git
- 🔧 **Opportunity**: Add git-lfs or exclude checkpoints from version control

### Quality Gates

| Gate | Status | Note |
|------|--------|------|
| Lint/Type Clean | ✅ PASS | New Python files are standalone scripts |
| Tests | ✅ PASS | No new test failures; M45 is research-only |
| Coverage Non-Decreasing | ✅ PASS | No production code changed |
| Secrets Scan | ✅ PASS | No secrets in new files |
| Deps CVE Non-New High | ✅ PASS | No new dependencies added |
| Schema/Infra Migration Ready | ✅ N/A | No schema changes |
| Docs/DX Updated | ✅ PASS | tunix-rt.md updated with M45 summary |

---

## 2. Change Map & Impact

```mermaid
graph LR
    subgraph "M45 Research (New)"
        A[split_dataset.py] --> B[data/*.jsonl]
        C[run_curriculum.py] --> D[checkpoints/]
        E[eval_all_checkpoints.py] --> F[eval/*.jsonl]
        G[provenance.json] --> H[analysis.md]
    end
    
    subgraph "Existing (Modified)"
        I[tunix-rt.md] --> J["M45 Enhancements section"]
    end
    
    subgraph "Unchanged"
        K[M42 Submission]
        L[Backend/Frontend]
        M[Training Infrastructure]
    end
```

**Dependency Direction:** Clean. Research scripts import from standard libraries only (torch, transformers, yaml, json). No new dependencies on backend modules.

**Layering Violations:** None. Research directory is fully isolated.

---

## 3. Code Quality Focus (Changed Files Only)

### 3.1 `research/m45_curriculum_reasoning/split_dataset.py`

**Observation:** 290 lines, well-structured with clear sections (configuration, generators, main).

**Interpretation:** Good separation of concerns. Each trace generator is self-contained.

**Recommendation:** None — code is clean and well-documented.

### 3.2 `research/m45_curriculum_reasoning/run_curriculum.py`

**Observation:** 340 lines, handles sequential training with checkpoint chaining.

```python:232:242:research/m45_curriculum_reasoning/run_curriculum.py
# Good: Clear stage configuration
STAGES = {
    "A": {
        "config": CONFIGS_DIR / "stage_a.yaml",
        "data": DATA_DIR / "stage_a.jsonl",
        "output": CHECKPOINTS_DIR / "stage_a",
        "init_from": None,  # Base model
        "epochs": 2,
    },
    ...
}
```

**Interpretation:** Configuration is centralized and explicit. Good for reproducibility.

**Recommendation:** Consider extracting `SFTDataset` class to a shared module if reused elsewhere.

### 3.3 `research/m45_curriculum_reasoning/eval_all_checkpoints.py`

**Observation:** 299 lines, evaluates 4 checkpoints in sequence with memory cleanup.

```python:186:190:research/m45_curriculum_reasoning/eval_all_checkpoints.py
# Good: Explicit memory management
del model
del tokenizer
torch.cuda.empty_cache()
```

**Interpretation:** Proper GPU memory cleanup between checkpoints prevents OOM.

**Recommendation:** None — code handles resource management correctly.

---

## 4. Tests & CI (Delta)

### Coverage Delta
- **Changed Production Code:** 0 lines
- **New Test Code:** 0 (M45 is research, not testable production code)
- **Coverage Impact:** None — no production changes

### CI Status
- **M45 is research-only** — not part of CI pipeline
- **Existing CI:** Unaffected (backend 384 tests, frontend 75 tests)

### Flakiness
- No flaky tests introduced
- Research scripts are deterministic (seed=42)

---

## 5. Security & Supply Chain (Delta)

### Secrets Check
- ✅ No secrets in new files
- ✅ No API keys, tokens, or credentials
- ✅ HuggingFace auth uses existing CLI token (not hardcoded)

### Dangerous Patterns
- ✅ No `eval()` or `exec()` usage
- ✅ No subprocess with shell=True on user input
- ✅ No unsafe file operations

### New Dependencies
- **None added** — scripts use existing venv packages

### Third-Party Risks
- ✅ No new dependencies = no new supply chain risk

---

## 6. Performance & Hot Paths

### M45 Performance
- Training: ~8.3 samples/sec on RTX 5090 (acceptable for research)
- Evaluation: ~1.5 samples/sec (batch size 1, could optimize)

### No Production Impact
- M45 is research-only; no production paths affected

---

## 7. Docs & DX (Changed Surface)

### Updated Documentation
- ✅ `tunix-rt.md`: M45 Enhancements section added
- ✅ `research/m45_curriculum_reasoning/analysis.md`: Qualitative findings
- ✅ `research/m45_curriculum_reasoning/provenance.json`: Full reproducibility manifest

### What a New Dev Must Know
1. M45 is in `research/` — not production code
2. Checkpoints are large (~5GB each) — may need git-lfs
3. Run from project root with `.venv-gpu` activated

### Docs PR Suggestion
Add a `research/README.md` explaining:
- Purpose of research directory
- How to reproduce M45
- What to exclude from git

---

## 8. Ready-to-Apply Patches (≤5)

### Patch 1: Add .gitignore for Research Checkpoints

**Title:** Exclude model checkpoints from git tracking

**Why:** Model checkpoints are ~5GB each. Tracking them bloats the repo.

**Patch Hint:**
```diff
# .gitignore
+ # Research model checkpoints (large binaries)
+ research/*/checkpoints/
+ research/*/checkpoints/**/*.safetensors
+ research/*/checkpoints/**/*.pt
+ research/*/checkpoints/**/*.pth
+ research/*/checkpoints/**/*.bin
```

**Risk:** Low | **Rollback:** Remove lines from .gitignore

---

### Patch 2: Add research/README.md

**Title:** Add README for research directory

**Why:** Explain purpose and usage for future developers.

**Patch Hint:**
```markdown
# Research Directory

This directory contains exploratory research experiments.
These are NOT production code and NOT part of the submission.

## M45: Curriculum Reasoning Training

See `m45_curriculum_reasoning/analysis.md` for findings.

### Reproduce
```bash
.\.venv-gpu\Scripts\Activate.ps1
python research/m45_curriculum_reasoning/run_curriculum.py
```
```

**Risk:** Low | **Rollback:** Delete file

---

### Patch 3: Fix Windows stdout encoding in split_dataset.py

**Title:** Consistent UTF-8 handling across all scripts

**Why:** Windows encoding issues already fixed in other scripts.

**Observation:** Already fixed. No action needed.

---

## 9. Next Milestone Plan (fits in <1 day)

| Task | Time | Acceptance Criteria |
|------|------|---------------------|
| Add .gitignore for checkpoints | 10 min | Checkpoints excluded from git status |
| Add research/README.md | 15 min | README explains M45 reproduction |
| Commit and push M45 | 10 min | Clean commit on GitHub |
| Create M46 plan | 45 min | M46_plan.md with structured self-correction scope |
| Review M46 scope with user | 30 min | Locked decisions documented |

---

## 10. Machine-Readable Appendix (JSON)

```json
{
  "delta": {
    "base": "4629ae2",
    "head": "pending_m45_commit"
  },
  "quality_gates": {
    "lint_type_clean": "pass",
    "tests": "pass",
    "coverage_non_decreasing": "pass",
    "secrets_scan": "pass",
    "deps_cve_nonew_high": "pass",
    "schema_infra_migration_ready": "n/a",
    "docs_dx_updated": "pass"
  },
  "issues": [
    {
      "id": "Q-001",
      "file": "research/m45_curriculum_reasoning/checkpoints/",
      "category": "dx",
      "severity": "low",
      "summary": "Large model checkpoints (~15GB total) not in .gitignore",
      "fix_hint": "Add research/*/checkpoints/ to .gitignore",
      "evidence": "Model binaries should not be version controlled"
    },
    {
      "id": "Q-002",
      "file": "research/",
      "category": "dx",
      "severity": "low",
      "summary": "No README explaining research directory purpose",
      "fix_hint": "Add research/README.md with reproduction instructions",
      "evidence": "New developers need context for research experiments"
    }
  ]
}
```

---

## Audit Conclusion

**M45 is APPROVED for commit.**

The milestone delivers exactly what was scoped:
- Curriculum training runs end-to-end
- Trace structure differences documented
- Full provenance and reproducibility
- Zero impact on M42 submission

**Recommended pre-commit actions:**
1. Add checkpoints to .gitignore (prevents 15GB+ repo bloat)
2. Add research/README.md (improves DX)

**Post-commit:** Ready for M46 planning (structured self-correction).

