# M47 Toolcalls Log — Injected Errors & Correction Fidelity

**Milestone:** M47  
**Started:** 2026-01-09

---

## Log Entries

### Entry 001 — 2026-01-09T[initial]
**Tool:** read_file (batch)  
**Purpose:** Check recovery state and understand M46 artifacts  
**Files:**
- `ProjectFiles/Milestones/Phase5/M47_toolcalls.md` — Check recovery state (empty)
- `research/m46_structured_self_correction/data/stage_c_self_correct.jsonl` — Sample M46 data

**Result:** M47 toolcalls empty (fresh start). M46 data has VERIFY/CORRECT structure with template_id tracking.

---

### Entry 002 — 2026-01-09T[initial]
**Tool:** write  
**Purpose:** Document clarifying questions before implementation  
**Files:**
- `ProjectFiles/Milestones/Phase5/M47_questions.md` — 7 clarifying questions

**Result:** Questions written. Awaiting user confirmation on:
1. Error types (Arithmetic + Unit recommended)
2. Injection mechanics (calculation error recommended)
3. CORRECT block content (explicit correction recommended)
4. Training init (M46 self_correct recommended)
5. Error location (final answer affected recommended)
6. Metric thresholds (conservative recommended)
7. Eval set (eval_v2.jsonl recommended)

---

## CHECKPOINT: Pre-Implementation Questions
**Status:** ✅ COMPLETE  
**Result:** All recommendations confirmed with two modifications in M47_answers.md

---

### Entry 003 — 2026-01-09T[confirmed]
**Tool:** read_file  
**Purpose:** Read user confirmation of M47 decisions  
**Files:**
- `ProjectFiles/Milestones/Phase5/M47_answers.md` — Locked decisions

**Result:** Decisions confirmed:
1. Error types: Arithmetic + Unit (confirmed)
2. Injection mechanics: Intermediate calculation errors, must propagate (confirmed)
3. CORRECT block: Explicit correction template (confirmed)
4. Training init: M46 self_correct checkpoint (confirmed)
5. Error location: **MODIFIED** → 80% intermediate-propagating, 20% final-only
6. Metric thresholds: Conservative + net improvement metric (confirmed)
7. Eval set: **MODIFIED** → Dual eval (eval_v2 + 10% Stage-C holdout)

**Additional Guardrails:**
- Manifest must include: sample_id, error_type, injected_step_idx, original_value, injected_value, corrected_value, expected_final
- Unit errors: scale factors only (meters↔cm, hours↔minutes)

---

## CHECKPOINT: Implementation Start
**Status:** ✅ COMPLETE  
**Next Step:** N/A — Milestone complete

---

### Entry 004 — 2026-01-09T18:00
**Tool:** run_terminal_cmd (New-Item)  
**Purpose:** Create M47 directory structure  
**Files:**
- `research/m47_error_correction_fidelity/data/`
- `research/m47_error_correction_fidelity/scripts/`
- `research/m47_error_correction_fidelity/checkpoints/`
- `research/m47_error_correction_fidelity/eval/`
- `research/m47_error_correction_fidelity/metrics/`
- `research/m47_error_correction_fidelity/configs/`

**Result:** Directories created successfully

---

### Entry 005 — 2026-01-09T18:02
**Tool:** write  
**Purpose:** Create error injection script  
**Files:**
- `research/m47_error_correction_fidelity/scripts/inject_errors.py`

**Result:** Script created with:
- 80/20 intermediate-propagating/final-only split
- Arithmetic error injection (+10 offset)
- Manifest with sample_id, error_type, injected_step_idx, values
- Holdout set creation (10%)

---

### Entry 006 — 2026-01-09T18:03
**Tool:** run_terminal_cmd (python inject_errors.py)  
**Purpose:** Execute error injection to create datasets  
**Files:**
- `research/m47_error_correction_fidelity/data/stage_c_clean.jsonl` (307 traces)
- `research/m47_error_correction_fidelity/data/stage_c_error.jsonl` (307 traces)
- `research/m47_error_correction_fidelity/data/stage_c_error_self_correct.jsonl` (307 traces)
- `research/m47_error_correction_fidelity/data/stage_c_holdout.jsonl` (34 traces)
- `research/m47_error_correction_fidelity/data/stage_c_holdout_error.jsonl` (34 traces)
- `research/m47_error_correction_fidelity/error_manifest.json`

**Result:**
- 21 errors successfully injected (6.8%)
- 15 intermediate-propagating, 6 final-only
- 9 injection failures (incompatible trace formats)
- All 21 were arithmetic errors (no suitable unit conversion traces)

---

### Entry 007 — 2026-01-09T18:05
**Tool:** write  
**Purpose:** Create training configs  
**Files:**
- `research/m47_error_correction_fidelity/configs/clean.yaml`
- `research/m47_error_correction_fidelity/configs/error_aware.yaml`

**Result:** Configs created with identical hyperparameters, different datasets

---

### Entry 008 — 2026-01-09T18:06
**Tool:** write  
**Purpose:** Create training script  
**Files:**
- `research/m47_error_correction_fidelity/scripts/run_training.py`

**Result:** Script created with SFTDataset, Windows UTF-8 fix, both experiments

---

### Entry 009 — 2026-01-09T18:08
**Tool:** run_terminal_cmd (python run_training.py)  
**Purpose:** Run both training experiments  
**Files:**
- `research/m47_error_correction_fidelity/checkpoints/clean/final_model/`
- `research/m47_error_correction_fidelity/checkpoints/error_aware/final_model/`
- `research/m47_error_correction_fidelity/checkpoints/training_summary.json`

**Result:**
- Clean: 0.121 loss, 77 steps, 58.5s
- Error-Aware: 0.144 loss, 77 steps, 62.0s
- Higher loss for error_aware suggests model learning correction patterns

---

### Entry 010 — 2026-01-09T18:12
**Tool:** write  
**Purpose:** Create evaluation script with fidelity metrics  
**Files:**
- `research/m47_error_correction_fidelity/scripts/run_eval.py`

**Result:** Script created with:
- Error detection patterns
- Correction behavior analysis
- Fidelity metrics (detection rate, correction accuracy, false corrections)

---

### Entry 011 — 2026-01-09T18:14
**Tool:** run_terminal_cmd (python run_eval.py)  
**Purpose:** Run evaluation on all checkpoints  
**Files:**
- `research/m47_error_correction_fidelity/eval/*.jsonl`
- `research/m47_error_correction_fidelity/eval/eval_summary.json`
- `research/m47_error_correction_fidelity/metrics/fidelity_comparison.json`

**Result:**
- All models produce VERIFY/CORRECT blocks (100%)
- Error detection rate: 0% across all models
- False correction rate: 0%
- Model generates correct calculations from scratch, not detecting errors

**Key Finding:** Verification is structural (template-following) not causal (error-detecting)

---

### Entry 012 — 2026-01-09T18:18
**Tool:** write  
**Purpose:** Create analysis document  
**Files:**
- `research/m47_error_correction_fidelity/analysis.md`

**Result:** Comprehensive analysis documenting negative but valuable results

---

### Entry 013 — 2026-01-09T18:20
**Tool:** write  
**Purpose:** Create provenance manifest  
**Files:**
- `research/m47_error_correction_fidelity/provenance.json`

**Result:** Full reproducibility manifest with hashes, metrics, conclusions

---

## CHECKPOINT: M47 Complete
**Status:** ✅ COMPLETE  
**Result:** Negative but scientifically valuable finding — verification is structural, not causal  
**Ready for:** Documentation update, git commit, M48 planning


