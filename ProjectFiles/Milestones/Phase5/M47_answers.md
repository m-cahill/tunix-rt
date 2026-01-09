Locked. Your recommendations are almost perfect — I’m going to **confirm most of them** and make **two targeted adjustments** that will make the experiment cleaner and more “scientifically persuasive.”

---

# ✅ M47 Clarifying Decisions — **AUTHORITATIVE / LOCKED**

## Q1: Error Types Selection

**Decision:** ✅ **Option 2 — Arithmetic Slip + Unit/Conversion Error**

* Best signal-to-complexity ratio
* Easy to inject + verify deterministically
* Generalizes beyond arithmetic-only toy effects

**Do NOT include logic-step omission in M47.** Save it for M48 (failure topology).

---

## Q2: Error Injection Mechanics (Arithmetic)

**Decision:** ✅ **Your Option 4, with a constraint**

✅ Inject a **calculation error in an intermediate step**
…but ensure it **propagates to the final answer** (see Q5).

**Guardrail:** inject errors that are:

* **single-point**
* **localized**
* **mechanically checkable**

**Examples of allowed arithmetic injections:**

* “60 + 80 = 150” (should be 140)
* “9 × 7 = 56” (should be 63)

Avoid sign flips unless the original task naturally includes negatives.

---

## Q3: CORRECT Block Content for Error Cases

**Decision:** ✅ **Option 1 — Explicit correction**

Use the full correction template:

> Identify the wrong step → provide corrected step → recompute final answer

Keep it short and structured (no essays). Example:

`CORRECT: Step 2 is wrong: 60 + 80 = 140 (not 150). Recompute: 140 + 8 = 148. Final: 148`

This teaches what “real correction” looks like.

---

## Q4: Training Init Checkpoint

**Decision:** ✅ **Option 1 — Initialize from M46 self_correct/final_model**

Exactly right. M47 is testing **extension of an already-learned verification behavior** into **error sensitivity**.

Starting from M45 would confound by re-learning formatting.

---

## Q5: Error Location Strategy

**Decision:** 🔁 **Modify your recommendation: choose a MIX, not final-only**

✅ Use **Option 3 — Both**, but with strict proportions:

* **80%** of injected errors: **intermediate step that propagates to final**
* **20%** of injected errors: **final-answer-only** (sanity check for “does it check the end”)

**Why this is better than final-only:**

* Final-only errors can be “caught” without understanding the reasoning
* Intermediate-propagating errors test whether `VERIFY` is actually tied to computation

This gives a stronger story about **process-level correction** rather than answer policing.

---

## Q6: Metric Thresholds (Meaningful Signal)

**Decision:** ✅ **Option 1 — Conservative thresholds**

Define “meaningful” as:

* **Detection rate ≥ 20%**
* **Correction accuracy ≥ 10%**
* **False corrections ≤ 25%** (stop condition remains)

That’s enough to justify M48+.

Also add one more sanity metric:

* **Net improvement rate ≥ 10%** on injected-error cases (optional, but compelling)

---

## Q7: Eval Set Design

**Decision:** 🔁 **Slight adjustment: Use BOTH eval_v2 + a held-out Stage-C slice**

✅ Do **Option 1** (inject into `eval_v2.jsonl`, ~10%) **AND** add a small held-out set:

* Hold out **10% of stage_c** *before* training selection
* Inject errors into that held-out set too

**Why:**
`eval_v2` is only 100 items; it’s valuable for continuity, but may be too small/noisy.
A held-out Stage-C slice ensures **distribution match** and strengthens claims.

This is still within scope and does not add new data.

---

# 🔒 Final Locked Summary (M47)

| Question | Decision                                                          |
| -------- | ----------------------------------------------------------------- |
| Q1       | Arithmetic + Unit errors                                          |
| Q2       | Intermediate calculation errors (mechanical), must propagate      |
| Q3       | Explicit correction template (wrong step → fix → recompute final) |
| Q4       | Init from M46 self_correct checkpoint                             |
| Q5       | Mix: 80% intermediate-propagating, 20% final-only                 |
| Q6       | Conservative: ≥20% detect, ≥10% correct, ≤25% false corrections   |
| Q7       | Dual eval: eval_v2 (10% injected) + 10% Stage-C holdout injected  |

You are **fully cleared to proceed**.

---

## Two Quick Implementation Guardrails (Worth Doing)

1. **Manifest everything**

   * `error_manifest.json` must include: sample_id, error_type, injected_step_idx, original_value, injected_value, corrected_value, expected_final

2. **Keep unit errors very simple**

   * Only scale factor mistakes like:

     * meters ↔ centimeters (×100 / ÷100)
     * hours ↔ minutes (×60)
   * Avoid obscure conversions (too hard to auto-verify reliably)

---

If Cursor follows this, M47 will produce a genuinely persuasive result *even if the model fails*, because the failure will be precisely measurable and will feed directly into M48 (“why does verification become ritual vs causal?”).
