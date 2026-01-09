# M48 Summary — Reasoning Failure Topology

**Status:** ✅ Complete  
**Date:** 2026-01-09  
**Type:** Analysis Milestone

---

## Objective

Map the failure modes of self-correction to answer:

> **Why does verification become ritual instead of diagnostic?**

---

## Key Finding

**Ritual Verification accounts for 97-100% of all verification behavior.**

| Source | Ritual Verification | Other | Total |
|--------|-------------------:|------:|------:|
| M46 self_correct (error) | 97% | 3% | 34 |
| M46 self_correct (clean) | 97% | 3% | 34 |
| M47 error_aware (error) | 100% | 0% | 34 |
| M47 error_aware (clean) | 100% | 0% | 34 |

Error-aware training (M47) did **not** change the failure topology.

---

## Failure Taxonomy (6 Classes)

1. **Ritual Verification** — Template text with no computational reference (97-100%)
2. **Computation Reset** — Re-solves from scratch instead of inspecting (0-3%)
3. **Local Error Blindness** — Detects structure, misses specific errors (0-3%)
4. **Detection without Localization** — Vague error acknowledgment (0%)
5. **Correction Hallucination** — "Fixes" a non-error (0%)
6. **Verification Collapse** — VERIFY degenerates to answer restatement (0%)

---

## Why M47 Failed

### 1. Template Learning Dominates

Training signal: `[reasoning] → VERIFY: [template] → CORRECT: No correction needed`

The model learns sequence completion, not inspection.

### 2. Low Error Density

6.8% error rate → Model calibrates to majority class ("No correction needed")

### 3. No Contrastive Pairs

Model never sees (error, clean) pairs for the same problem → Cannot learn state comparison

### 4. Missing Diff Operator

Verification requires: `expected vs actual → mismatch?`

The model has no mechanism for this comparison.

---

## Mechanistic Model

```
Reasoning Trace → Template Selection → Default Output
                        ↓                    ↓
              "Check by inverse..."   "No correction needed"
```

There is no conditional branch based on computational content.

---

## Implications for Future Work

| Requirement | Current State | Needed |
|-------------|---------------|--------|
| Error density | 6.8% | 30-50% |
| Contrastive pairs | None | (error, clean) for same problem |
| Value grounding | Template text | VERIFY references specific numbers |
| Comparison training | None | (before, after, diff) triplets |

---

## Artifacts

| Artifact | Path |
|----------|------|
| Failure Taxonomy | `taxonomy/failure_taxonomy.md` |
| Classification Labels | `metrics/failure_labels.json` |
| Contrastive Examples | `taxonomy/contrastive_examples.md` |
| Reasoning Graph | `taxonomy/reasoning_graph.md` |
| Analysis | `analysis.md` |
| Provenance | `provenance.json` |

---

## Conclusion

M48 successfully characterized the failure topology:

> **Verification operates as a post-hoc ritual because training teaches template completion, not state comparison.**

This is a training design failure, not a model capability failure. The path forward requires contrastive training with explicit state comparison.

