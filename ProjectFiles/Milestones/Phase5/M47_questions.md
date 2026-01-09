# M47 Clarifying Questions — Injected Errors & Correction Fidelity

**Date:** 2026-01-09  
**Status:** Awaiting User Response

---

## Pre-Implementation Questions

Before proceeding with M47 implementation, I need answers to the following:

---

### Q1: Error Types Selection

The plan allows 2-3 error types. Recommended options:
1. Arithmetic Slip (off-by-one, sign error, miscalculation)
2. Unit/Conversion Error (wrong scale factor)
3. Logic Step Omission (skipped step)

**Question:** Which 2-3 error types should I implement?

**Options:**
1. **All three** — Maximum coverage but more complex
2. **Arithmetic Slip + Unit Error** — More mechanical, easier to verify
3. **Arithmetic Slip only** — Simplest, clearest signal

**My recommendation:** Option 2 (Arithmetic Slip + Unit Error) — These are the most common error types in the dataset and easiest to programmatically inject/verify. Logic Step Omission requires more complex trace manipulation.

---

### Q2: Error Injection Mechanics

**Question:** How should I inject arithmetic errors?

**Options:**
1. **Off-by-one** — Change final digit (e.g., 51 → 52)
2. **Sign flip** — Negate the result (e.g., 51 → -51)
3. **Wrong operation** — Use addition instead of subtraction in the reasoning step
4. **Calculation error** — Introduce error in intermediate calculation (e.g., "60 + 80 = 150" instead of 140)

**My recommendation:** Option 4 (Calculation error in intermediate step) — This is more realistic and tests whether the model can catch an error in reasoning, not just the final answer. Combined with a corrected final answer so the error is in the trace, not the answer.

---

### Q3: CORRECT Block Content for Error Cases

In M46, all CORRECT blocks said "No correction needed" because there were no errors.

**Question:** For error-injected traces with VERIFY/CORRECT structure, what should the CORRECT block contain?

**Options:**
1. **Explicit correction** — "CORRECT: The calculation should be 60 + 80 = 140, not 150. Final answer is 148."
2. **Error acknowledgment** — "CORRECT: Error detected. Recalculating..."
3. **Ground-truth answer** — "CORRECT: The correct answer is 148"

**My recommendation:** Option 1 (Explicit correction) — This gives the model a template for what a real correction looks like, including acknowledging the specific error and providing the fix.

---

### Q4: Training Init Checkpoint

The plan says to init from "M46 Self-Correct checkpoint."

**Question:** Should I use:
1. The M46 `self_correct/final_model` checkpoint (after 1 epoch on VERIFY/CORRECT structure)
2. The M45 Stage-C checkpoint (curriculum-trained, no VERIFY/CORRECT)

**My recommendation:** Option 1 (M46 self_correct checkpoint) — This tests whether the already-learned verification behavior can extend to error detection. Starting from M45 would test learning both behaviors simultaneously.

---

### Q5: Error Location Strategy

**Question:** Where in the trace should errors be injected?

**Options:**
1. **Intermediate step** — Error in "calculation" or "add_tens" step, but correct final answer
2. **Final answer** — Error propagates to final answer
3. **Both** — Mix of both types

**My recommendation:** Option 2 (Final answer affected) — If the error doesn't affect the final answer, it's harder to measure whether correction helped. The model should learn that errors in reasoning lead to wrong answers.

---

### Q6: Metric Thresholds

The plan mentions stop conditions (>25% false corrections) but not success thresholds.

**Question:** What counts as "meaningful" correction fidelity?

**Options:**
1. **Conservative** — Error detection rate >20%, correction accuracy >10%
2. **Moderate** — Error detection rate >40%, correction accuracy >25%
3. **Aggressive** — Error detection rate >60%, correction accuracy >50%

**My recommendation:** Option 1 (Conservative) — M47 is exploratory. Any non-trivial signal is valuable. We can set higher bars for M48+.

---

### Q7: Eval Set Design

The plan mentions evaluating on "Error-injected eval set."

**Question:** Should I inject errors into:
1. **eval_v2.jsonl** (100 examples) — Same eval set as M45/M46
2. **A subset of stage_c** held out from training
3. **New synthetic traces** with known errors

**My recommendation:** Option 1 (eval_v2.jsonl) — Maintains consistency with M45/M46 evaluation. Inject errors into ~10% (~10 examples) for comparison.

---

## Summary

| Question | My Recommendation |
|----------|-------------------|
| Q1: Error types | Arithmetic Slip + Unit Error |
| Q2: Injection mechanics | Calculation error in intermediate step |
| Q3: CORRECT block content | Explicit correction with error acknowledgment |
| Q4: Training init | M46 self_correct checkpoint |
| Q5: Error location | Final answer affected |
| Q6: Metric thresholds | Conservative (>20% detection, >10% correction) |
| Q7: Eval set | eval_v2.jsonl with 10% injected errors |

---

**Please confirm or modify these recommendations before I proceed.**

Once I have answers, I will:
1. Log pre-implementation toolcalls to `M47_toolcalls.md`
2. Create the directory structure under `research/m47_error_correction_fidelity/`
3. Build the error injection script
4. Execute training and evaluation
5. Produce analysis document


