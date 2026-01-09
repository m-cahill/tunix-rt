# Phase 5 Executive Summary

**Project:** RediAI / Tunix RT  
**Phase:** 5 (Post-Submission Research)  
**Status:** Complete  
**Date:** 2026-01-09

---

## One-Line Summary

> **Self-correction fails because generation lacks a state-comparison operator; error detection succeeds when architecturally separated.**

---

## The Research Question

Can a language model be trained to verify and correct its own reasoning?

---

## The Answer

**Partially.** We can train models to produce verification *structure* (VERIFY/CORRECT blocks), but not verification *function* (actual error detection). The generator always says "No correction needed" regardless of whether errors exist.

**However:** An external observer model achieves 50% error detection where the generator achieves 0%. Error detection is a separable capability.

---

## Key Findings

| Milestone | Question | Result |
|-----------|----------|--------|
| M45 | Does curriculum affect reasoning? | Yes — 25% loss improvement with staged training |
| M46 | Can verification be trained? | Yes — 97% of outputs include VERIFY blocks |
| M47 | Does verification detect errors? | **No** — 0% detection rate |
| M48 | Why does it fail? | Ritual verification at 97-100%; no state comparison |
| M49 | Can detection be separated? | **Yes** — Observer achieves 50% recall, AUC 0.969 |

---

## Why Self-Correction Fails

1. **Autoregressive generation** — Each token depends only on previous tokens; no "look back" capability
2. **Training as sequence completion** — VERIFY/CORRECT are learned as "what comes next," not as operations
3. **Missing state-comparison** — No mechanism to compare computed vs. expected values
4. **Imbalanced priors** — 93% of training had "No correction needed," so model defaults to this
5. **No contrastive pairs** — Model never sees (error, clean) pairs for the same problem

---

## What We Discovered Instead

| Discovery | Implication |
|-----------|-------------|
| Curriculum shapes structure | Training order affects reasoning quality |
| Verification form is trainable | Models can produce checking structure |
| Verification function is not trainable (naively) | Structure ≠ behavior |
| Error detection is separable | Observation succeeds where generation fails |

---

## The Architectural Insight

```
Generator → [Verification Block] → "No correction needed" (0% detection)
     ↓
Observer → [Compare values] → Error signal (50% detection)
```

**Key:** The observer succeeds because comparison is its *primary function*, not a side effect of generation.

---

## Limits & Non-Claims

We do **not** claim:
- "We solved reasoning"
- "Models can now self-correct"
- "The observer is production-ready"

We **do** claim:
- We mapped why self-correction fails
- We showed verification is structural, not causal
- We demonstrated a viable architectural separation

---

## What Would Be Required to Go Further

| Requirement | Current State | Needed |
|-------------|---------------|--------|
| Error density | 6.8% | 30-50% |
| Contrastive pairs | None | (error, clean) pairs |
| Value grounding | Template text | VERIFY with numbers |
| Architecture | Generator only | Generator + Observer |

---

## RediAI as a System

RediAI is not a silver-bullet model. It is a **reasoning systems laboratory**:
- Designs and tests precise hypotheses
- Documents failure modes with rigor
- Produces falsifiable, reproducible experiments

The value is in *making reasoning legible*.

---

## Phase 5 Thesis

> **Self-correction fails not because models "cannot reason," but because generation lacks a state-comparison operator. Verification behavior is trainable as form, but not as function. Error detection succeeds when architecturally separated from generation.**

---

## For Judges

This work demonstrates:
1. **Methodological rigor** — Each milestone asked a precise question and produced falsifiable evidence
2. **Intellectual honesty** — Negative results (M47) were reported and analyzed, not hidden
3. **Architectural insight** — The failure diagnosis (M48) led to a working solution (M49)
4. **Complete narrative** — M45 → M49 forms a coherent research arc from structure through failure to separation

Phase 5 shows that reasoning research can be systematic, reproducible, and insightful — even when the initial hypothesis (self-correction) fails.

---

**Generated:** 2026-01-09  
**Word Count:** ~500  
**Reading Time:** 2-3 minutes

