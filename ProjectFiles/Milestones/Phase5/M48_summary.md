# M48 Summary — Reasoning Failure Topology

**Status:** ✅ Complete  
**Date:** 2026-01-09  
**Type:** Analysis Milestone (Research, Phase 5)  
**Git Commit:** `52b90b4`

---

## Executive Summary

M48 answered the central question from M47's failure:

> **Why does verification become ritual instead of diagnostic?**

### Primary Finding

**Ritual Verification accounts for 97-100% of all verification behavior** across both M46 (baseline) and M47 (error-aware) checkpoints.

Verification is **structural** (template-following) rather than **causal** (error-detecting). The model produces VERIFY/CORRECT blocks as sequence completions, not as inspection operations that compare computational states.

---

## Objective

M48 was an **analysis milestone** — the goal was understanding, not improvement.

Tasks:
1. Define a failure taxonomy for verification behavior
2. Classify M46 and M47 predictions using structural + regex heuristics
3. Produce contrastive examples showing where errors should have been noticed
4. Create reasoning graphs showing structural disconnection
5. Synthesize findings into a mechanistic explanation

---

## Classification Results

### Failure Distribution by Source

| Source | Ritual Verification | Local Error Blindness | Computation Reset | Total |
|--------|--------------------:|----------------------:|------------------:|------:|
| M46 self_correct (error holdout) | 33 (97%) | 1 (3%) | 0 (0%) | 34 |
| M46 self_correct (clean holdout) | 33 (97%) | 0 (0%) | 1 (3%) | 34 |
| M47 error_aware (error holdout) | 34 (100%) | 0 (0%) | 0 (0%) | 34 |
| M47 error_aware (clean holdout) | 34 (100%) | 0 (0%) | 0 (0%) | 34 |

### Key Observations

1. **Ritual Verification dominates** — The model emits template text ("Check by inverse...") without referencing actual computed values
2. **Error-aware training did not change failure topology** — M47 has identical (or slightly worse) patterns compared to M46
3. **Zero successful detections** — No traces showed correct error identification
4. **Zero correction hallucinations** — The model doesn't "over-correct" clean traces (false positive rate = 0%)
5. **Zero detection without localization** — The model never even vaguely acknowledges errors

---

## Failure Taxonomy (6 Classes)

### 1. Ritual Verification (Dominant: 97-100%)

**Definition:** VERIFY block is present but contains only templated language with no reference to actual computation or prior reasoning steps.

**Example:**
```
VERIFY: Check by inverse: divide distance by time to verify speed
CORRECT: No correction needed
```

**Detection:** Template match + absence of specific numbers from reasoning trace.

---

### 2. Computation Reset (Rare: 0-3%)

**Definition:** The model re-solves the problem from the prompt instead of inspecting prior reasoning.

**Mechanistic interpretation:** No "diff operator" — cannot compare current state to prior state.

---

### 3. Local Error Blindness (Rare: 0-3%)

**Definition:** Detects global structure (e.g., "this is a distance problem") but misses specific local errors (e.g., arithmetic miscalculation).

**Mechanistic interpretation:** Verification operates at semantic/structural level without arithmetic grounding.

---

### 4. Detection without Localization (Not observed: 0%)

**Definition:** Vague acknowledgment of error without identifying which step.

---

### 5. Correction Hallucination (Not observed: 0%)

**Definition:** "Corrects" something that was not wrong.

---

### 6. Verification Collapse (Not observed: 0%)

**Definition:** VERIFY block degenerates into restatement of the answer.

---

## Root Cause Analysis

### 1. Why Template Learning Dominates

**Training signal:** 
```
[reasoning steps] → VERIFY: [template] → CORRECT: No correction needed → [answer]
```

This is a **sequence completion task**, not an **inspection task**. The model learns:
- "When I reach step N, emit VERIFY template"
- "After VERIFY, emit CORRECT with default text"

There is no conditional logic based on computational content.

---

### 2. Why Low Error Density Fails

M47 injected errors into only 6.8% of training traces (21/307).

**Prior imbalance:**
- P(error) ≈ 0.07
- P(no error) ≈ 0.93

The optimal strategy under this prior is: **always predict "No correction needed"**.

The model is *calibrated to the training distribution* but not *sensitive to error signals*.

---

### 3. Why Absence of Contrastive Pairs Matters

M47 training showed the model traces with errors AND corrections, but never showed:
- The same problem **without** errors (for comparison)
- Explicit (before, after, diff) triplets

Without contrastive examples, the model cannot learn:
- What "correct" looks like as a reference point
- How to detect divergence from a baseline
- When to trigger correction vs. default

---

### 4. Why Verification Lacks State Comparison

**Architectural gap:** Transformer-based generation is autoregressive. Each token is generated based on previous tokens, but there is no explicit "working memory" for intermediate values or structured comparison.

**Training gap:** The training data never demonstrates:
- Explicit value extraction ("Step 2 produced X")
- Explicit comparison ("X ≠ Y, therefore error")
- Explicit localization ("Error is in Step 2")

---

## Mechanistic Model of Failure

```
┌─────────────────┐     ┌─────────────────────┐     ┌────────────────────┐
│ Reasoning Trace │ ──▶ │ Template Selection  │ ──▶ │ Default Output     │
│                 │     │ (based on problem   │     │ "No correction     │
│                 │     │  type, not content) │     │  needed"           │
└─────────────────┘     └─────────────────────┘     └────────────────────┘
```

**What's missing:**
```
┌─────────────────┐     ┌─────────────────────┐     ┌────────────────────┐
│ Expected Value  │ ──▶ │ State Comparison    │ ──▶ │ Conditional Output │
│ vs Actual Value │     │ (diff operator)     │     │ based on mismatch  │
└─────────────────┘     └─────────────────────┘     └────────────────────┘
```

The verification pathway is **unconditional** — it always produces the same output regardless of computational content.

---

## Cross-Model Comparison (M46 vs M47)

| Metric | M46 self_correct | M47 error_aware | Delta |
|--------|------------------|-----------------|-------|
| Ritual Verification | 97% | 100% | +3% |
| Successful Detection | 0% | 0% | 0% |
| False Corrections | 0% | 0% | 0% |

**Conclusion:** Error-aware training (M47) did not improve error detection. If anything, it slightly increased ritual verification.

---

## Contrastive Analysis (5 Representative Examples)

Each example showed the same pattern:
1. ✅ VERIFY block present with template text
2. ❌ No reference to specific computed values
3. ❌ Error in trace not mentioned
4. ✅ CORRECT says "No correction needed"

**Key observation:** The VERIFY block references problem *type* ("Check by inverse: divide distance by time") but never references problem *values* (the actual numbers computed).

---

## Reasoning Graph (Mermaid)

### Expected vs Actual Verification

**Expected:** VERIFY should compare computed values to expected values via a state-difference operation.

**Actual:** VERIFY attaches to the final answer as a post-hoc appendage with no connection to intermediate computations.

The structural gap is that verification operates as a **sequence completion** (what comes after reasoning) rather than as an **inspection operation** (what compares states).

---

## Implications for Future Work

### What Would Be Needed for Functional Self-Correction

| Requirement | M47 State | Needed for Success |
|-------------|-----------|-------------------|
| Error density | 6.8% | 30-50% |
| Contrastive pairs | None | (error, clean) for same problem |
| Value grounding | Template text | VERIFY references specific numbers |
| Comparison training | None | (before, after, diff) triplets |
| Conditional logic | None | Branch on mismatch detection |

### This is a Training Design Failure

M48 reveals that self-correction failure is not about model capability — it's about training signal design. The model learned exactly what the data taught:

> "Emit verification structure without verification function."

---

## Framing (Per M48 Plan)

M48 uses **mechanistic language**, not anthropomorphic claims:

- ❌ "The model fails to reason"
- ❌ "The model doesn't understand errors"
- ✅ "Verification lacks a state-difference operator"
- ✅ "Reasoning steps are not compared against prior state"
- ✅ "Self-correction operates as a post-hoc ritual"

This framing is **alignment-adjacent, novel, and credible**.

---

## Artifacts Produced

| Artifact | Path | Description |
|----------|------|-------------|
| Failure Taxonomy | `taxonomy/failure_taxonomy.md` | 6-class taxonomy with definitions, heuristics, examples |
| Classification Script | `scripts/classify_failures.py` | Structural + regex heuristics |
| Failure Labels | `metrics/failure_labels.json` | Per-trace classifications |
| Failure Counts | `metrics/failure_counts.json` | Aggregate counts by source |
| Counts Table | `taxonomy/failure_counts_table.md` | Markdown table summary |
| Contrastive Examples | `taxonomy/contrastive_examples.md` | 5 representative examples with annotations |
| Reasoning Graph | `taxonomy/reasoning_graph.md` | Mermaid diagrams showing structural disconnection |
| Synthesis Analysis | `analysis.md` | Comprehensive explanation of why M47 failed |
| Provenance | `provenance.json` | Full reproducibility manifest with hashes |

All artifacts are located in `research/m48_reasoning_failure_topology/`.

---

## Conclusion

M48 successfully characterized the failure topology of self-correction:

> **Verification operates as a post-hoc ritual because training teaches template completion, not state comparison.**

This is a **strong capstone** for the Phase 5 research track. We now have:
- M45: Curriculum reshapes reasoning structure ✅
- M46: Verification behavior is trainable ✅
- M47: Verification is structural, not causal ✅
- M48: Failure topology mapped with mechanistic explanation ✅

The path forward (if pursued) would require fundamentally different training design:
- High error density (30-50%)
- Contrastive pairs with explicit state comparison
- Value-grounded verification (not templates)

M48 provides the analytical foundation for any future work on functional self-correction.

---

**Author:** M48 Reasoning Failure Topology Milestone  
**Generated:** 2026-01-09  
**Git Commit:** `52b90b4`

