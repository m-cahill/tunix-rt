# M46 Clarifying Questions — Structured Self-Correction

**Date:** 2026-01-08  
**Status:** Awaiting User Response

---

## Pre-Implementation Questions

Before proceeding with M46 implementation, I need answers to the following:

---

### Q1: Marker Format Confirmation

The M46 plan proposes this marker format:

```
ANSWER:
REASONING:
VERIFY:
CORRECT (if needed):
FINAL:
```

**Question:** Should I use this exact format, or do you prefer a variation?

**Options:**
1. **Exactly as shown above** — 5-block structure with optional CORRECT
2. **Simplified** — Just `VERIFY:` and `CORRECT:` appended to existing trace
3. **Custom** — Something else (please specify)

**My recommendation:** Option 2 (simplified) — inject `VERIFY:` and `CORRECT:` blocks after the existing reasoning steps, rather than restructuring the entire trace. This minimizes transformation complexity and preserves the original trace structure from M45.

---

### Q2: Injected Errors Strategy

The plan allows for "a small number of known errors" (<10% of samples) with explicit corrections.

**Question:** How should I approach injected errors?

**Options:**
1. **Zero injected errors** — Only restructure existing traces with VERIFY/CORRECT blocks (CORRECT: "No correction needed" for most)
2. **Minimal injection (~5%)** — Manually craft ~17 samples with deliberate calculation errors + corrections
3. **Programmatic injection (~10%)** — Automatically introduce errors in arithmetic samples (e.g., off-by-one errors)

**My recommendation:** Option 1 (zero injected errors) for the first run. Injecting errors adds complexity and could confound results. If the model learns verification behavior from structure alone, that's cleaner evidence. We can add injected errors in M47 if needed.

---

### Q3: Verification Content Generation

**Question:** How should I generate the `VERIFY:` block content?

**Options:**
1. **Template-based** — Rules like "For arithmetic: 'Check: X ± Y = Z using inverse operation'"
2. **Copy-based** — `VERIFY:` restates the answer from the last step
3. **Structural** — `VERIFY: Confirm steps follow template pattern`

**My recommendation:** Option 1 (template-based) — Create verification templates per problem type (arithmetic, percentage, unit conversion, etc.) based on the existing trace categories.

---

### Q4: Training Duration

The plan says "1–2 epochs" for each run.

**Question:** What should the exact training duration be?

**Options:**
1. **1 epoch each** — Quick signal detection (~85 steps per run)
2. **2 epochs each** — More saturation (~170 steps per run)
3. **Match M45 Stage-C** — 3 epochs each for direct comparison

**My recommendation:** Option 1 (1 epoch each) — We're testing whether the model *picks up* verification behavior, not maximizing loss. Shorter runs = faster iteration.

---

### Q5: Control Dataset Treatment

The plan calls for a "Control" dataset with "Original Stage-C traces (unchanged)".

**Question:** Should the control dataset be truly unchanged, or should we restructure both datasets identically except for VERIFY/CORRECT?

**Options:**
1. **Unchanged** — Control uses raw `stage_c.jsonl` exactly as-is
2. **Parallel structure** — Both datasets restructured with markers, but control uses `VERIFY: [placeholder]` and no CORRECT

**My recommendation:** Option 1 (unchanged) — Keeps the comparison clean: "Does adding verification structure change behavior?"

---

### Q6: Evaluation Focus

**Question:** What's the primary metric we care about?

**Options:**
1. **Verification frequency** — Does the model produce more VERIFY-like language?
2. **Correction accuracy** — When the model corrects, is it right?
3. **Final answer accuracy** — Does self-correction improve end results?
4. **All of the above** — Comprehensive behavioral analysis

**My recommendation:** Option 1 (verification frequency) as primary, with Option 4 analysis in the write-up. M46's thesis is about *whether verification behavior can be trained*, not whether it improves accuracy.

---

## Summary

| Question | My Recommendation |
|----------|-------------------|
| Q1: Marker format | Simplified (VERIFY/CORRECT appended) |
| Q2: Injected errors | Zero for V1 |
| Q3: Verification content | Template-based per problem type |
| Q4: Training duration | 1 epoch each |
| Q5: Control dataset | Unchanged (raw stage_c.jsonl) |
| Q6: Evaluation focus | Verification frequency as primary |

---

**Please confirm or modify these recommendations before I proceed.**

Once I have answers, I will:
1. Log pre-implementation toolcalls to `M46_toolcalls.md`
2. Create the directory structure under `research/m46_structured_self_correction/`
3. Build the trace transformation script
4. Execute training and evaluation
5. Produce analysis document


