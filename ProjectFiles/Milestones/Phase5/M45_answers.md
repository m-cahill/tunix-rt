# M45 Answers & Locked Decisions

**Date:** 2026-01-08  
**Status:** ✅ LOCKED — Proceed with execution

---

## Authoritative Decisions

### 1. Partitioning Strategy

**Decision:** ✅ **Option A — Category-Based (APPROVED)**

Category is the **primary signal**, trace length is **secondary within `reasoning`**.

| Stage | Inclusion Rules | Expected Count |
|-------|-----------------|----------------|
| **Stage A (Low)** | `synthetic` + `golden_style` | ~145 |
| **Stage B (Medium)** | `reasoning` where `len(steps) == 3` | ~majority of 385 |
| **Stage C (Full)** | `reasoning` where `len(steps) >= 4` + all `edge_case` | remaining + 20 |

**Rationale:** Using semantic labels is **more honest**, not "less pure". Author-supplied structure strengthens interpretability and reproducibility.

---

### 2. Epoch Counts

**Decision:** ✅ **2-2-3 CONFIRMED**

| Stage | Epochs | Reason |
|-------|--------|--------|
| A | 2 | Warm-up, pattern grounding |
| B | 2 | Structured reasoning stabilization |
| C | 3 | Long-trace + verification emphasis (smaller, denser) |

Asymmetry is a **feature**, not a confound.

---

### 3. Baseline Checkpoint

**Decision:** ✅ **Use M44 final checkpoint**

- Location: `submission_runs/m44_v1/training_output/final_model/`
- Do NOT retrain a new flat baseline
- Statement: "M44 represents the best flat SFT achievable under identical conditions prior to curriculum ordering."

---

### 4. Training Backend

**Decision:** ✅ **PyTorch CONFIRMED**

- Same backend as M44
- Same hardware (RTX 5090)
- JAX is explicitly out-of-scope for M45

---

### 5. Evaluation Set

**Decision:** ✅ **`training/evalsets/eval_v2.jsonl` CONFIRMED**

- Use unchanged
- Evaluate all checkpoints: M44, Post-A, Post-B, Post-C
- No filtering, cherry-picking, or reweighting

---

### 6. Directory Structure

**Decision:** ✅ **New top-level directory**

```
research/m45_curriculum_reasoning/
  data/
  configs/
  checkpoints/
  eval/
  analysis.md
  provenance.json
```

Do NOT nest under `submission_runs/`.

---

## Framing Guidance (for Analysis)

**Avoid:**
- ❌ "curriculum improves accuracy"
- ❌ "better performance"

**Prefer:**
- ✅ "curriculum reshapes reasoning structure"
- ✅ "ordering alters trace emergence"
- ✅ "verification behaviors appear earlier"

---

## Summary: Locked M45 Configuration

| Parameter | Value |
|-----------|-------|
| Partitioning | Category-first + trace-length within reasoning |
| Epochs | A=2, B=2, C=3 |
| Baseline | M44 final checkpoint |
| Backend | PyTorch (RTX 5090) |
| Eval Set | `eval_v2.jsonl` |
| Directory | `research/m45_curriculum_reasoning/` |

---

**Status:** ✅ CLEARED TO PROCEED

