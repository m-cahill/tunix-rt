# Phase 5 Timeline — Reasoning Research Arc

**Phase:** 5 (Post-Submission Research)  
**Duration:** M45 → M49  
**Status:** Complete

---

## Milestone Timeline

| Milestone | Question Asked | Answer Found | Key Metric |
|-----------|----------------|--------------|------------|
| **M45** | Can curriculum ordering reshape reasoning structure? | Yes — stage-based training produces qualitatively different trace patterns | Loss improved 25% (0.095 → 0.072) with curriculum |
| **M46** | Can verification behavior be explicitly trained? | Yes — VERIFY/CORRECT blocks appear when trained on augmented data | Verification frequency: 5% → 97% (+92pp) |
| **M47** | Does trained verification actually detect errors? | No — verification is structural (template-following), not causal (error-detecting) | Error detection rate: 0% |
| **M48** | Why does self-correction fail? | Lack of state-difference operator; training teaches sequence completion, not inspection | Ritual verification: 97-100% of traces |
| **M49** | Can error detection be externalized to an observer? | Yes — a simple observer model achieves non-zero detection | Observer: 50% detection, AUC 0.969 (validation) |

---

## Narrative Arc

### Act 1: Structure (M45)

**Question:** Does training order matter for reasoning?

**Method:** Three-stage curriculum (basic → intermediate → advanced arithmetic)

**Finding:** Yes. Curriculum-trained models produce more structured traces with clearer step delineation. The model learns *how* to organize reasoning, not just *what* to compute.

---

### Act 2: Form (M46)

**Question:** Can we train a model to produce self-correction markers?

**Method:** Augment Stage-C traces with VERIFY and CORRECT blocks

**Finding:** Yes. The model reliably produces verification structure when trained on augmented data. Verification behavior is trainable as a *form*.

---

### Act 3: Function (M47)

**Question:** Does verification actually work?

**Method:** Inject errors into traces, measure detection rate

**Finding:** No. Despite producing VERIFY/CORRECT blocks, the model detects 0% of injected errors. It says "No correction needed" regardless of whether an error exists.

**Critical insight:** Verification is learned as *sequence completion*, not as *error inspection*.

---

### Act 4: Diagnosis (M48)

**Question:** Why does verification fail?

**Method:** Classify failure modes, analyze verification behavior

**Finding:** Ritual Verification accounts for 97-100% of all verification. The model lacks:
- State-comparison operator (cannot compare computed vs expected)
- Contrastive training signal (never sees error/clean pairs)
- Conditional generation (no branch on mismatch)

---

### Act 5: Separation (M49)

**Question:** Can detection be separated from generation?

**Method:** Train a lightweight observer on answer-comparison features

**Finding:** Yes. A logistic regression observer achieves 50% recall (vs generator's 0%) with validation AUC 0.969. Error detection is a separable function.

---

## Timeline Summary

```
M45 (Structure)  →  M46 (Form)  →  M47 (Function?)  →  M48 (Diagnosis)  →  M49 (Separation)
     ↓                  ↓                ↓                   ↓                    ↓
  Curriculum        Verification      Test fails       Map failure           Observer
  improves          appears           (0% detect)      topology              succeeds
  structure         in output                          (97% ritual)          (50% detect)
```

---

## Key Numbers (Summary)

| Milestone | Primary Metric | Value |
|-----------|----------------|-------|
| M45 | Curriculum loss improvement | 25% |
| M46 | Verification frequency increase | +92pp (5% → 97%) |
| M47 | Generator error detection | 0% |
| M48 | Ritual verification rate | 97-100% |
| M49 | Observer error detection | 50% (AUC 0.969 val) |

---

## What Phase 5 Proved

1. **Verification structure is trainable** — Models can learn to produce VERIFY/CORRECT blocks
2. **Verification function is not trainable (naively)** — Structural output ≠ causal behavior
3. **Self-correction fails architecturally** — Generation lacks state-comparison capability
4. **Error detection is separable** — An external observer can detect what the generator cannot

---

**Generated:** 2026-01-09  
**Source:** M45-M49 analysis.md, provenance.json, and metrics files

