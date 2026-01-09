# M50 Analysis — Recursive System Post-Mortem

**Milestone:** M50 Recursive System Post-Mortem  
**Phase:** 5 (Complete)  
**Date:** 2026-01-09

---

## What Did the System Learn?

Phase 5 asked a sequence of precise questions about reasoning, verification, and error detection. This document synthesizes the findings into a coherent architectural understanding.

### The Central Question

> **Can a language model be trained to verify and correct its own reasoning?**

### The Answer

**Partially.** A model can be trained to produce verification *structure* (VERIFY/CORRECT blocks), but not verification *function* (actual error detection). Error detection requires architectural separation — an external observer that compares computed values against expected values.

---

## Why Naïve Self-Correction Fails

This section explains the mechanisms behind verification failure, grounded in M47 and M48 findings.

### 1. Autoregressive Generation

Language models generate text autoregressively: each token depends only on previous tokens. This creates a fundamental limitation:

- The model cannot "look back" at its reasoning to inspect it
- Verification becomes another sequence to complete, not an operation to perform
- There is no native mechanism to hold values for comparison

**Evidence:** M48 showed 97-100% of verification follows the same pattern regardless of content.

### 2. Absence of State-Difference Operator

Verification requires comparing two values:

```
expected_value vs computed_value → mismatch?
```

The generator has no such operator. When it produces "VERIFY: Check by inverse," it:
- Does not extract its computed value
- Does not compute the inverse
- Does not compare the results

The VERIFY block references the *type* of check without *instantiating* it.

**Evidence:** M48 classified this as "Ritual Verification" — template text without computational grounding.

### 3. Training as Sequence Completion

M46 training data taught:

```
[reasoning] → VERIFY: [template] → CORRECT: No correction needed → [answer]
```

The model learned: "After reasoning, emit VERIFY template, then CORRECT with default text."

This is a *sequence completion* task, not an *inspection* task. The model learns what tokens come next, not what behavior to perform.

**Evidence:** M46 achieved 97% verification frequency, but M47 showed 0% error detection.

### 4. Imbalanced Error Priors

M47 injected errors into 6.8% of training traces. The optimal strategy under this prior is:

```
P(error) ≈ 0.07 → always predict "No correction needed"
```

The model is calibrated to the training distribution, not sensitive to error signals.

**Evidence:** M47 error-aware model achieved identical 0% detection as the control.

### 5. Lack of Contrastive Supervision

The training data never showed:
- The same problem with and without errors
- Explicit (error, clean) pairs for comparison
- Different CORRECT outputs for different conditions

Without contrastive examples, the model cannot learn:
- What "correct" looks like as a reference
- When to trigger correction vs. default

**Evidence:** M48 identified "No contrastive pairs" as a root cause of failure.

---

## What the System Discovered Instead

Phase 5 did not solve self-correction. It discovered something more precise: **why self-correction fails and what architectural changes would be required**.

### 1. Curriculum as Structural Shaping (M45)

Training order affects reasoning structure. Stage-based curriculum (basic → intermediate → advanced) produces traces with:
- Clearer step delineation
- More consistent formatting
- Better loss convergence (25% improvement)

**Insight:** Reasoning structure is malleable through curriculum design.

### 2. Verification as Learnable Form (M46)

Explicit verification markers (VERIFY/CORRECT) can be trained into model outputs. When trained on augmented data:
- 97% of outputs include VERIFY blocks
- 84% include CORRECT language
- Structure is stable and reproducible

**Insight:** Verification *form* is trainable, even if *function* is not.

### 3. Observation as Separable Function (M49)

Error detection works when separated from generation. A simple observer model:
- Compares generated vs. expected answers
- Achieves 50% recall (vs. generator's 0%)
- Validation AUC: 0.969

**Insight:** Detection is a different capability than generation. Architectural separation enables it.

### 4. Error Detection as Classification (M49)

The observer treats error detection as binary classification:
- Input: generated text + expected answer
- Output: error present (0/1)
- Features: answer match, numeric difference

This is fundamentally different from the generator's task (sequence completion).

**Insight:** Reformulating verification as classification enables success.

---

## Limits & Non-Claims

Phase 5 does **not** claim:

| Non-Claim | Why |
|-----------|-----|
| "We solved reasoning" | We characterized failure, not success |
| "Models can now self-correct" | Self-correction still fails at 0% |
| "The observer is production-ready" | 68 samples, simple features, demonstration only |
| "This generalizes to all domains" | Tested only on arithmetic reasoning |
| "More training would fix it" | The failure is architectural, not statistical |

### What Would Be Required to Go Further

1. **Higher error density:** 30-50% instead of 6.8%
2. **Contrastive training:** (error, clean) pairs for same problem
3. **Value grounding:** VERIFY must reference specific numbers
4. **Comparison training:** Explicit (before, after, diff) triplets
5. **Architectural changes:** Memory, critic networks, or tool use

Phase 5 mapped the gap. Closing it would require new experiments beyond the current scope.

---

## What RediAI Is (and Is Not)

### What RediAI Is

RediAI is a **reasoning systems laboratory**:
- A framework for designing, probing, and analyzing reasoning behavior
- A methodology for asking precise questions about model capabilities
- A system that produces falsifiable, documented experiments

### What RediAI Is Not

RediAI is **not**:
- A silver-bullet model that "solves reasoning"
- A leaderboard entry optimized for benchmarks
- A production system ready for deployment
- A claim that reasoning is solved

### The Value Proposition

RediAI's value is in *making reasoning legible*:
- Explicitly testing what works and what doesn't
- Documenting failure modes with precision
- Providing architectural insights for future work

---

## Synthesis: The Phase 5 Thesis

Phase 5 established a coherent thesis:

> **Self-correction fails not because models "cannot reason," but because generation lacks a state-comparison operator. Verification behavior is trainable as form, but not as function. Error detection succeeds when architecturally separated from generation.**

This is a mechanistic, falsifiable, and actionable insight. It explains observed behavior without anthropomorphizing model capabilities, and it suggests concrete architectural directions for future work.

---

## Artifacts Consulted

This synthesis drew from:

| Milestone | Key Artifacts |
|-----------|---------------|
| M45 | `analysis.md`, `provenance.json`, training metrics |
| M46 | `analysis.md`, behavioral metrics, eval predictions |
| M47 | `analysis.md`, error_manifest.json, fidelity metrics |
| M48 | `analysis.md`, failure_taxonomy.md, contrastive_examples.md |
| M49 | `analysis.md`, observer_metrics.json, contrastive_demo.md |

Full provenance is documented in `provenance.json`.

---

**Author:** M50 Recursive System Post-Mortem  
**Generated:** 2026-01-09  
**Status:** Phase 5 Complete

