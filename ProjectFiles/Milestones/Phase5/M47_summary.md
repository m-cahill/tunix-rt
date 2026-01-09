# M47 Summary — Error Correction Fidelity

**Status:** ✅ Complete  
**Date:** 2026-01-09  
**Type:** Research Milestone (Negative but Valuable Result)

---

## Objective

Test whether a model trained on explicit error corrections develops the ability to detect and correct novel errors during generation.

---

## Key Finding

**Verification behavior is structural (template-following) rather than causal (error-detecting).**

All models (M46, M47-clean, M47-error-aware) reliably produce VERIFY/CORRECT blocks but none detect errors in their own reasoning.

---

## Experiment Design

### Error Injection
- **21 errors injected** into 307 training traces (6.8%)
- 15 intermediate-propagating (71%), 6 final-only (29%)
- All arithmetic errors (no suitable unit conversion traces)

### Training Runs
| Run | Dataset | Loss | Steps |
|-----|---------|------|-------|
| Clean | stage_c_clean.jsonl | 0.121 | 77 |
| Error-Aware | stage_c_error_self_correct.jsonl | 0.144 | 77 |

### Evaluation
- 34 held-out samples (clean + error-injected)
- Fidelity metrics: detection rate, correction accuracy, false corrections

---

## Results

### Fidelity Metrics (All Models, Error-Injected Holdout)

| Metric | M46 | M47 Clean | M47 Error-Aware |
|--------|-----|-----------|-----------------|
| VERIFY block present | 100% | 100% | 100% |
| Error detection rate | 0% | 0% | 0% |
| Correction attempt rate | 0% | 0% | 0% |
| False correction rate | 0% | 0% | 0% |

### Threshold Assessment (per M47_answers.md)

| Threshold | Target | Actual | Status |
|-----------|--------|--------|--------|
| Detection rate | ≥20% | 0% | ❌ Not met |
| Correction accuracy | ≥10% | 0% | ❌ Not met |
| False corrections | ≤25% | 0% | ✅ Met |

---

## What the Model Generates

Example output (M47 Error-Aware):
```
I need to use the formula: distance = speed x time
Given: speed = 108 km/h, time = 6 hours
Distance = 108 x 6 = 648 km
The train travels 648 km
VERIFY: Check by inverse: divide distance by time to verify speed
CORRECT: No correction needed
648 km
```

The model:
1. ✅ Computes the problem from scratch (often correctly)
2. ✅ Appends VERIFY block with template text
3. ✅ Appends CORRECT block saying "No correction needed"
4. ❌ Does NOT detect when its own computation has an error

---

## Why It Doesn't Work

1. **Weak training signal:** Only 6.8% of traces had errors
2. **Default behavior dominates:** 93% of training had "No correction needed"
3. **No contrastive learning:** Model never sees (error, correct) pairs for same problem

---

## Implications for M48

Per the original plan, M47's failure mode informs M48:

1. **Higher error injection rate:** 30-50% instead of 10%
2. **Contrastive training:** Pairs of (error, correct) traces
3. **Explicit error markers:** Train on `[ERROR DETECTED]` patterns
4. **Chain-of-thought verification:** Model must explain what it's checking

---

## Artifacts

| Artifact | Path |
|----------|------|
| Analysis | `research/m47_error_correction_fidelity/analysis.md` |
| Provenance | `research/m47_error_correction_fidelity/provenance.json` |
| Checkpoints | `research/m47_error_correction_fidelity/checkpoints/` |
| Error manifest | `research/m47_error_correction_fidelity/error_manifest.json` |
| Eval predictions | `research/m47_error_correction_fidelity/eval/*.jsonl` |

---

## Conclusion

M47 is a **negative but scientifically valuable result**. It proves that:
1. Verification structure transfers reliably
2. Error-awareness does NOT emerge from exposure to corrections alone
3. Future work requires contrastive/high-dosage error training

The experiment provides clear direction for M48.

