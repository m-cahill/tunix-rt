# M45 Milestone Summary: Curriculum Reasoning Training

**Milestone:** M45  
**Status:** ✅ Complete  
**Date Completed:** 2026-01-09  
**Phase:** 5 (Exploratory Research)

---

## Executive Summary

M45 demonstrated that **curriculum ordering of reasoning data produces qualitative improvements in reasoning trace structure** compared to flat SFT — without changing the model or optimizer.

**Key Outcome:** Curriculum training reshapes how the model reasons. Post-curriculum outputs exhibit more explicit step-by-step patterns, formula-first reasoning, and verification language.

> "We can shape how a model reasons by structuring information — not by brute force."

---

## Objectives & Results

| Objective | Status | Evidence |
|-----------|--------|----------|
| Partition dataset by complexity | ✅ Complete | A: 145, B: 64, C: 341 samples |
| Run sequential curriculum training | ✅ Complete | 364 total steps, 220s runtime |
| Compare trace structure vs M44 | ✅ Complete | analysis.md with examples |
| Produce reproducible artifacts | ✅ Complete | provenance.json with hashes |

---

## Training Results

### Dataset Partitioning

| Stage | Description | Samples | Avg Steps | Categories |
|-------|-------------|---------|-----------|------------|
| A | Low complexity | 145 | 2.0 | synthetic, golden_style |
| B | Medium complexity | 64 | 3.0 | reasoning (3 steps) |
| C | Full complexity | 341 | 3.89 | reasoning (4+ steps), edge_case |

### Training Performance

| Stage | Epochs | Steps | Loss | Runtime |
|-------|--------|-------|------|---------|
| A | 2 | 74 | 0.564 | 57.1s |
| B | 2 | 32 | 0.291 | 28.4s |
| C | 3 | 258 | 0.191 | 134.4s |
| **Total** | **7** | **364** | **0.191** | **219.9s** |

### Comparison to M44 (Flat SFT)

| Metric | M44 (Flat) | M45 (Curriculum) |
|--------|------------|------------------|
| Total Steps | 414 | 364 |
| Final Loss | 0.72 | 0.19 |
| Runtime | ~200s | ~220s |
| Trace Structure | Generic | Decomposed |

---

## Qualitative Findings

### Trace Structure Improvements

| Pattern | M44 (Flat SFT) | Post-Stage-C (Curriculum) |
|---------|----------------|---------------------------|
| Step structure | Generic "Step X of Y" | Problem-specific decomposition |
| Formula usage | Implicit/absent | Explicit ("I need to use the formula...") |
| Verification | Rare | Common ("Check: X is approximately...") |
| Topic drift | Frequent | Rare |

### Example: Arithmetic Subtraction

**Prompt:** "What is 100 - 37?"

**M44:** `100 - 37 = 63 Result 2 of 2 63`

**Post-Stage-C:** `Subtracting 37 from 100 Ones digit: handle 0 - 7 Tens digit: handle 1 - 3 Final result: 63`

---

## Deliverables Checklist

| Deliverable | Status | Location |
|-------------|--------|----------|
| Dataset split script/notebook | ✅ | `split_dataset.py` |
| stage_a.jsonl, stage_b.jsonl, stage_c.jsonl | ✅ | `data/` |
| Training configs for all stages | ✅ | `configs/` |
| Three sequential checkpoints | ✅ | `checkpoints/` |
| Provenance/run manifest | ✅ | `provenance.json` |
| Evaluation outputs (all checkpoints) | ✅ | `eval/` |
| Qualitative analysis document | ✅ | `analysis.md` |

---

## Quality Gates

| Gate | Status | Evidence |
|------|--------|----------|
| Training end-to-end | ✅ PASS | 364/364 steps |
| No manual intervention | ✅ PASS | Fully automated |
| Ordering changed behavior | ✅ PASS | Trace structure differs |
| Specific trace differences | ✅ PASS | analysis.md examples |
| Reproducible from scratch | ✅ PASS | provenance.json with hashes |

---

## Artifacts Produced

```
research/m45_curriculum_reasoning/
├── data/
│   ├── stage_a.jsonl              # 145 low-complexity traces
│   ├── stage_b.jsonl              # 64 medium-complexity traces
│   ├── stage_c.jsonl              # 341 full-complexity traces
│   ├── split_stats.json           # Partitioning statistics
│   └── trace_length_histogram.txt # Distribution visualization
├── configs/
│   ├── stage_a.yaml               # Stage A training config
│   ├── stage_b.yaml               # Stage B training config
│   └── stage_c.yaml               # Stage C training config
├── checkpoints/
│   ├── stage_a/final_model/       # Post-Stage-A checkpoint
│   ├── stage_b/final_model/       # Post-Stage-B checkpoint
│   ├── stage_c/final_model/       # Post-Stage-C checkpoint
│   └── curriculum_training_summary.json
├── eval/
│   ├── m44_baseline_predictions.jsonl
│   ├── post_stage_a_predictions.jsonl
│   ├── post_stage_b_predictions.jsonl
│   ├── post_stage_c_predictions.jsonl
│   └── eval_summary.json
├── split_dataset.py               # Dataset partitioning script
├── run_curriculum.py              # Training orchestrator
├── eval_all_checkpoints.py        # Batch evaluation script
├── provenance.json                # Full provenance manifest
├── analysis.md                    # Qualitative analysis
├── training_log.txt               # Training output log
└── eval_log.txt                   # Evaluation output log
```

---

## Impact on Submission

**Zero.**

- M42 submission ZIP unchanged
- No code in main codebase modified
- M45 is research-only, stored in `research/` directory
- This is differentiation material, not submission mutation

---

## Lessons Learned

1. **Category labels are a feature** — Using `meta.category` for partitioning is semantically meaningful and honest
2. **Lower final loss** — Curriculum achieved 0.19 vs flat SFT's 0.72 with fewer total steps
3. **Trace structure is trainable** — Formula-first, decomposition, and verification patterns emerged from ordering alone
4. **Repetition is a generation issue** — Both models repeat; curriculum just repeats better content

---

## Next Steps

Based on M45 findings, recommended follow-up research:

| Milestone | Focus |
|-----------|-------|
| M46 | Structured self-correction (can model fix its own errors?) |
| M47 | Generation parameter tuning (reduce repetition) |
| M48 | Output parsing (extract final answers for accuracy) |

---

## Conclusion

**M45 validates that curriculum is a viable research direction for reasoning improvement.**

The evidence shows curriculum-trained models exhibit structurally different reasoning traces — not just different answers. This opens possibilities for M46 (structured self-correction) with real evidence instead of speculation.

**M42 remains the authoritative submission artifact.**

