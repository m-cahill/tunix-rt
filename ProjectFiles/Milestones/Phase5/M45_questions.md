# M45 Clarifying Questions

**Date:** 2026-01-08  
**Status:** Awaiting Response

---

## Context

After analyzing the project, I found that `dev-reasoning-v2` has **rich existing metadata** that could inform curriculum partitioning:

| Category | Count | % | Step Range | Notes |
|----------|-------|---|------------|-------|
| **reasoning** | 385 | 70% | 3-4 steps | Multi-step math, verification |
| **synthetic** | 110 | 20% | 1-2 steps | Simple arithmetic, string ops |
| **golden_style** | 35 | 6% | 2 steps | Text repetition, transforms |
| **edge_case** | 20 | 4% | 1-6 steps | Pipeline hardening examples |

The traces already have a `meta.category` field that correlates with complexity.

---

## Questions

### 1. Partitioning Strategy — Category vs. Trace Length

The plan states to partition based on **trace length and complexity**. However, the dataset already has labeled categories that roughly align with complexity:

**Option A — Category-Based (Recommended):**
- **Stage A (Low):** `synthetic` + `golden_style` (145 traces) — 1-2 steps, direct answers
- **Stage B (Medium):** `reasoning` traces with 3 steps (majority)
- **Stage C (Full):** `reasoning` traces with 4+ steps + `edge_case` — explicit verification steps

**Option B — Pure Trace Length:**
- **Stage A:** `len(steps) <= 2`
- **Stage B:** `len(steps) == 3`
- **Stage C:** `len(steps) >= 4`

**Question:** Should I use the existing `meta.category` as the primary signal (Option A), or compute pure trace length (Option B)? Option A is more semantically meaningful but less "pure."

---

### 2. Exact Epoch Counts

The plan says "2–3 epochs" per stage. For reproducibility, I need exact numbers:

| Stage | Proposed Epochs |
|-------|----------------|
| A | 2 |
| B | 2 |
| C | 3 |

**Question:** Do you confirm **2-2-3** epoch distribution? Or should it be uniform (2-2-2 or 3-3-3)?

---

### 3. Baseline Checkpoint for Comparison

The plan says to compare against "Baseline flat SFT checkpoint (from M44)". 

**M44 Evidence:**
- Model: `google/gemma-2b` (PyTorch)
- Training: 3 epochs on full `dev-reasoning-v2` (414 steps)
- Final loss: 0.72
- Checkpoint: `submission_runs/m44_v1/training_output/final_model/`

**Question:** Should I use the **M44 final checkpoint** as the flat SFT baseline for comparison? Or should I train a fresh flat baseline as part of M45 to ensure identical hyperparameters?

---

### 4. Training Backend — PyTorch or JAX?

M44 used **PyTorch** on RTX 5090. The JAX path has TPU/Linux limitations.

**Question:** Do you confirm we proceed with **PyTorch** for M45 curriculum training (same as M44)?

---

### 5. Evaluation Set

The plan says to use a "fixed evaluation set." We have:
- `training/evalsets/eval_v2.jsonl` — 100 items with section/category/difficulty labels

**Question:** Do you confirm `eval_v2.jsonl` as the evaluation set for all checkpoints?

---

### 6. Research Directory — Existing Structure?

The plan specifies creating `research/m45_curriculum_reasoning/`. 

**Question:** Should I create this as a new top-level directory, or place it under `submission_runs/m45_v1/` to follow existing convention?

---

## Summary of Proposed Defaults (If No Response)

If you don't respond to specific questions, I will proceed with:

1. **Partitioning:** Option A (category-based with trace length secondary)
2. **Epochs:** 2-2-3 distribution (A=2, B=2, C=3)
3. **Baseline:** M44 final checkpoint for comparison
4. **Backend:** PyTorch (same as M44)
5. **Eval Set:** `eval_v2.jsonl`
6. **Directory:** `research/m45_curriculum_reasoning/` (new top-level)

---

**Awaiting your confirmation or adjustments before proceeding.**

