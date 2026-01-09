# M48 Analysis — Why Self-Correction Becomes Ritual

**Milestone:** M48 Reasoning Failure Topology  
**Status:** Complete  
**Date:** 2026-01-09

---

## Executive Summary

M48 analyzed the failure modes of model self-correction to answer:

> **Why does verification become ritual instead of diagnostic?**

### Key Finding

**Ritual Verification accounts for 97-100% of all verification behavior** across both M46 and M47 checkpoints.

Verification is **structural** (template-following) rather than **causal** (error-detecting). The model produces VERIFY/CORRECT blocks as sequence completions, not as inspection operations.

---

## 1. Classification Results

### Failure Distribution

| Source | Ritual Verification | Local Error Blindness | Computation Reset | Total |
|--------|--------------------:|----------------------:|------------------:|------:|
| M46 self_correct (error) | 33 (97%) | 1 (3%) | 0 | 34 |
| M46 self_correct (clean) | 33 (97%) | 0 | 1 (3%) | 34 |
| M47 error_aware (error) | 34 (100%) | 0 | 0 | 34 |
| M47 error_aware (clean) | 34 (100%) | 0 | 0 | 34 |

### Interpretation

1. **Ritual Verification dominates** — The model emits template text ("Check by inverse...") without referencing actual values
2. **Error-aware training did not change failure topology** — M47 has identical patterns to M46
3. **No successful detections** — Zero traces showed correct error identification
4. **No correction hallucination** — The model doesn't "over-correct" clean traces

---

## 2. Why Template Learning Dominates

### Training Signal Analysis

From M45/M46 training data:
- 100% of traces have VERIFY blocks
- 93%+ of CORRECT blocks say "No correction needed"
- VERIFY text uses fixed templates per problem type

The model learns a simple pattern:
```
[reasoning steps] → VERIFY: [template] → CORRECT: No correction needed → [answer]
```

This is a **sequence completion task**, not an **inspection task**.

### Position Encoding Effect

VERIFY/CORRECT appear at fixed positions (end of trace). The model learns:
- "When I reach step N, emit VERIFY template"
- "After VERIFY, emit CORRECT with default text"

There is no conditional logic based on computational content.

---

## 3. Why Low Error Density Fails

M47 injected errors into only 6.8% of training traces (21/307).

This creates a **prior imbalance**:
- P(error) ≈ 0.07
- P(no error) ≈ 0.93

The optimal strategy under this prior is: **always predict "No correction needed"**.

The model learned exactly this strategy — it is *calibrated to the training distribution* but not *sensitive to error signals*.

---

## 4. Why Absence of Contrastive Pairs Matters

M47 training showed the model traces with errors AND corrections, but never showed:
- The same problem **without** errors (for comparison)
- Explicit (before, after, diff) triplets

Without contrastive examples, the model cannot learn:
- What "correct" looks like as a reference point
- How to detect divergence from a baseline
- When to trigger correction vs. default

### The Missing Diff Operator

Verification requires comparing two states:
```
expected_value vs actual_value → mismatch?
```

The model has no mechanism for this comparison. It generates verification text as a monologue, not as a comparison between states.

---

## 5. Why Verification Lacks State Comparison

### Architectural Gap

Transformer-based generation is **autoregressive**:
- Each token is generated based on previous tokens
- There is no explicit "working memory" for intermediate values
- Comparison requires holding two values and computing difference

When the model generates VERIFY, it has access to:
- The full context window (prompt + reasoning + answer)
- But no structured representation of "values to compare"

### Training Gap

The training data never demonstrates:
- Explicit value extraction ("Step 2 produced X")
- Explicit comparison ("X ≠ Y, therefore error")
- Explicit localization ("Error is in Step 2")

VERIFY blocks use semantic templates ("Check by inverse") without instantiating them with values.

---

## 6. Mechanistic Model of Failure

```
Reasoning Trace → Verification Block
                  ↓
        Template Selection (based on problem type)
                  ↓
        Default Output: "No correction needed"
```

There is no branch for:
```
        Comparison: computed vs expected
                  ↓
        Error Detection: mismatch found
                  ↓
        Localization: which step
                  ↓
        Correction: fix and recompute
```

The verification pathway is **unconditional** — it always produces the same output regardless of computational content.

---

## 7. What This Means for Self-Correction Research

### Negative Result, Positive Insight

M47 failed to produce error-sensitive verification. But M48 reveals *why*:

1. **Verification structure ≠ Verification function** — Learning to emit VERIFY blocks does not imply learning to verify
2. **Prior calibration ≠ Error sensitivity** — Models calibrate to training distribution, defaulting to majority class
3. **Sequential generation ≠ State comparison** — Autoregressive generation lacks explicit diff operators

### Implications for Future Work

To achieve functional self-correction, training would need:

| Requirement | Current State | Needed |
|-------------|---------------|--------|
| Error density | 6.8% | 30-50% |
| Contrastive pairs | None | (error, clean) for same problem |
| Value grounding | Template text | VERIFY references specific numbers |
| Comparison training | None | Explicit (before, after, diff) triplets |
| Conditional logic | None | Branch on mismatch detection |

---

## 8. Conclusion

M48 successfully characterized the failure topology of self-correction:

> **Verification operates as a post-hoc ritual because the training signal teaches template completion, not state comparison.**

This is not a model failure — it is a training design failure. The model learned exactly what the data taught: emit verification structure without verification function.

The path forward is not "more training" but "different training" — specifically, contrastive examples with explicit state comparison.

---

## Artifacts Produced

| Artifact | Path |
|----------|------|
| Failure Taxonomy | `taxonomy/failure_taxonomy.md` |
| Failure Labels | `metrics/failure_labels.json` |
| Failure Counts | `metrics/failure_counts.json` |
| Counts Table | `taxonomy/failure_counts_table.md` |
| Contrastive Examples | `taxonomy/contrastive_examples.md` |
| Reasoning Graph | `taxonomy/reasoning_graph.md` |
| This Analysis | `analysis.md` |

---

**Author:** M48 Reasoning Failure Topology Milestone  
**Generated:** 2026-01-09

