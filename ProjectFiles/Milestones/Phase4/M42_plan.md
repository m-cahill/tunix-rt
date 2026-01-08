# M42 Milestone Plan: Final Submission Package

**Milestone:** M42  
**Status:** ✅ COMPLETE  
**Date:** 2026-01-08  
**Objective:** Create the final submission package for the Tunix Hackathon

---

## Scope Summary

M42 is a **packaging and documentation-only milestone**. No feature code changes allowed.

### Allowed Changes
- ✅ Packaging scripts
- ✅ Documentation updates
- ✅ Evidence capture helpers
- ✅ Deterministic build scripts

### Not Allowed
- 🚫 Feature changes
- 🚫 Training logic changes
- 🚫 Refactors
- 🚫 Behavior changes

---

## Deliverables

| # | Deliverable | Location | Status |
|---|-------------|----------|--------|
| 1 | VIDEO_CHECKLIST.md | `docs/submission/VIDEO_CHECKLIST.md` | ✅ Complete |
| 2 | README.md polish | `README.md` | ✅ Complete |
| 3 | GPU/nightly documentation | `CONTRIBUTING.md` | ✅ Complete |
| 4 | Package submission script update | `backend/tools/package_submission.py` | ✅ Complete |
| 5 | Final test run | `submission_runs/m42_v1/test_run_outputs/` | ✅ Complete |
| 6 | Evidence index | `submission_runs/m42_v1/evidence_index.md` | ✅ Complete |
| 7 | pip freeze evidence | `submission_runs/m42_v1/pip_freeze_backend.txt` | ✅ Complete |
| 8 | Submission ZIP | `tunix_rt_m42_2026-01-08_e54267b.zip` | ✅ Complete |
| 9 | tunix-rt.md update | `tunix-rt.md` | ✅ Complete |

---

## Execution Plan

### Phase 1: Documentation Prep (Deliverables 1-3)

#### Task 1.1: Create VIDEO_CHECKLIST.md
**File:** `docs/submission/VIDEO_CHECKLIST.md`

Contents:
- Explicit statement: "This checklist + docs/DEMO.md is the source of truth"
- Pre-recording checklist (services running, data seeded)
- Shot order with timing (reference DEMO.md scenes)
- Post-recording checklist (upload, verify, add link to README)
- Fallback commands for each demo step

#### Task 1.2: Polish README.md
**File:** `README.md`

Update ONLY these sections:
1. **Top summary** — Judge-facing language, why this exists
2. **Quickstart (Local)** — Streamlined for reproduction
3. **Demo Flow** — Points to `docs/DEMO.md`
4. **Evidence & Reproducibility** — Points to `submission_runs/`
5. **Training Paths** — JAX (TPU) vs PyTorch (Local GPU) clarity

Add video URL placeholder:
```markdown
🎥 **Demo Video:** (link will be added upon submission)
```

#### Task 1.3: Document GPU Environment
**File:** `CONTRIBUTING.md`

Add GPU Development section:
- Why `.venv-gpu` is separate
- PyTorch nightly cu128 installation
- CUDA 12.8 requirement for RTX 5090 (sm_120)
- Known fragility of nightly wheels
- Validated throughput (31.7 samples/sec)

---

### Phase 2: Packaging Infrastructure (Deliverable 4)

#### Task 2.1: Update package_submission.py
**File:** `backend/tools/package_submission.py`

Changes:
1. Update `ARCHIVE_PREFIX` from `"tunix_rt_m36"` to `"tunix_rt_m42"`
2. Update `generate_submission_readme()` version string
3. Add M40/M41/M42 evidence folders to evidence file list
4. Verify BUNDLE_FILES list is complete

Assessment: Script is **clean, deterministic, low-risk** → reuse with updates.

---

### Phase 3: Evidence Capture (Deliverables 5-7)

#### Task 3.1: Run Final Test Suite
**Commands:**
```powershell
# Backend tests
cd backend
uv run pytest --cov=tunix_rt_backend --cov-branch -v 2>&1 | Out-File ../submission_runs/m42_v1/test_run_outputs/backend_tests.txt

# Frontend tests
cd ../frontend
npm run test 2>&1 | Out-File ../submission_runs/m42_v1/test_run_outputs/frontend_tests.txt
```

**Output:** `submission_runs/m42_v1/test_run_outputs/`
- `backend_tests.txt`
- `frontend_tests.txt`

#### Task 3.2: Capture pip freeze (GPU environment)
**Commands:**
```powershell
# Standard environment
cd backend
uv run pip freeze > ../submission_runs/m42_v1/pip_freeze_backend.txt

# GPU environment (if available)
.\.venv-gpu\Scripts\Activate.ps1
pip freeze > submission_runs/m42_v1/pip_freeze_training_pt_gpu.txt
```

#### Task 3.3: Create evidence_index.md
**File:** `submission_runs/m42_v1/evidence_index.md`

Contents:
- What each folder (m40_v1, m41_v1, m42_v1) proves
- What was run and on which machine
- Any known benign warnings
- Link to authoritative docs (DEMO.md, CONTRIBUTING.md)

---

### Phase 4: Package & Finalize (Deliverables 8-9)

#### Task 4.1: Run Package Script
**Command:**
```powershell
cd C:\coding\tunix-rt
python backend/tools/package_submission.py --output submission_runs/m42_v1 --run-dir submission_runs/m42_v1
```

**Output:** `submission_runs/m42_v1/tunix_rt_m42_<date>_<sha>.zip`

Move/rename to: `submission_runs/m42_v1/tunix_rt_submission_m42.zip`

#### Task 4.2: Update tunix-rt.md
Add M42 enhancements section with:
- VIDEO_CHECKLIST.md created
- README polish (5 sections)
- GPU/nightly documentation in CONTRIBUTING.md
- Evidence index created
- Final submission ZIP packaged

---

## Quality Gates

| Gate | Acceptance Criteria | Status |
|------|---------------------|--------|
| VIDEO_CHECKLIST exists | `docs/submission/VIDEO_CHECKLIST.md` present with shot list | ✅ PASS |
| README updated | Video URL placeholder + 5 sections polished | ✅ PASS |
| GPU docs added | CONTRIBUTING.md has GPU Development section | ✅ PASS |
| Backend tests pass | 384 passed, 75.79% coverage | ✅ PASS |
| Frontend tests pass | 75 passed, 7 test files | ✅ PASS |
| Evidence index exists | `submission_runs/m42_v1/evidence_index.md` present | ✅ PASS |
| pip freeze captured | `pip_freeze_backend.txt` present | ✅ PASS |
| ZIP created | `tunix_rt_m42_2026-01-08_e54267b.zip` (104.8 KB) | ✅ PASS |
| tunix-rt.md updated | M42 enhancements documented | ✅ PASS |

---

## Required Evidence Folders

Per M42 answers, these are required in final submission:

| Folder | Purpose | Status |
|--------|---------|--------|
| `submission_runs/m40_v1/` | GPU enablement proof (RTX 5090, CUDA 12.8) | ✅ Exists |
| `submission_runs/m41_v1/` | Frontend polish proof (clean tests) | ✅ Exists |
| `submission_runs/m42_v1/` | Final test runs, pip freeze, evidence index | ⏳ Create |

---

## Locked Decisions (from M42_answers.md)

1. **Video:** Manual recording, `docs/DEMO.md` is authoritative
2. **Submission:** ZIP archive to `submission_runs/m42_v1/`
3. **Dependencies:** `uv.lock` authoritative, document nightly PyTorch
4. **README:** Update 5 sections only, video URL as placeholder
5. **Evidence:** m40_v1 + m41_v1 + m42_v1 required, create evidence_index.md
6. **Scope:** Packaging/docs only, NO code changes
7. **Test pass:** Mandatory, capture to test_run_outputs/

---

## Estimated Time

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 1 | Documentation (3 tasks) | 45 min |
| Phase 2 | Packaging script update | 15 min |
| Phase 3 | Evidence capture (3 tasks) | 30 min |
| Phase 4 | Package & finalize | 20 min |
| **Total** | | **~2 hours** |

---

## Recovery Protocol

Per M42 answers, if session disrupts:
1. Check `M42_toolcalls.md` for last CHECKPOINT
2. Resume from next uncompleted task
3. Restate what was in progress before continuing

---

## Next After M42

Upon M42 completion:
1. **FREEZE** — No further code changes
2. **Record video** — Following VIDEO_CHECKLIST.md
3. **Submit** — Upload to Kaggle
4. **M43 (optional)** — Full production training run

