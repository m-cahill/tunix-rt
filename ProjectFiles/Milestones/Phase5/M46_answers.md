This is an excellent set of questions, and your instincts are right across the board. I’ll give **explicit confirmations** so M46 can proceed without ambiguity, then I’ll add **two small guardrails** to keep the experiment maximally interpretable.

---

# ✅ M46 Clarifying Decisions — **LOCKED**

You can treat everything below as **authoritative and final**.

---

## Q1: Marker Format Confirmation

**Decision:**
✅ **Option 2 — Simplified (VERIFY / CORRECT appended)**

**Why this is correct:**

* Preserves the **M45-earned trace structure**
* Minimizes transformation-induced confounds
* Keeps the experiment focused on *behavioral augmentation*, not format re-learning

**Canonical rule:**

* Do **not** restructure existing reasoning
* Append blocks in this order:

```
VERIFY:
CORRECT:
```

* `CORRECT:` may be empty or contain “No correction needed”

This gives you a **minimal intervention** with maximal signal clarity.

---

## Q2: Injected Errors Strategy

**Decision:**
✅ **Option 1 — Zero injected errors (for M46)**

This is the right scientific move.

**Rationale:**

* You are testing **whether verification behavior emerges at all**
* Injected errors introduce:

  * Attribution ambiguity
  * Correction-target bias
  * Additional degrees of freedom

If structure alone induces verification behavior, that is *much stronger evidence*.

We can introduce injected errors later (M47+) if needed.

---

## Q3: Verification Content Generation

**Decision:**
✅ **Option 1 — Template-based verification (per problem type)**

This is critical and subtle.

**Guidelines:**

* Templates should:

  * Be **mechanical**, not “thoughtful”
  * Reference known invariants (inverse ops, bounds, units)
  * Avoid adding new reasoning content

**Example patterns (illustrative):**

* Arithmetic:

  > `VERIFY: Check by inverse operation: 63 + 37 = 100`

* Percentage:

  > `VERIFY: Confirm percentage by recomputing fraction`

* Units:

  > `VERIFY: Ensure units cancel correctly`

You are teaching **what verification *looks like***, not *how to think harder*.

---

## Q4: Training Duration

**Decision:**
✅ **Option 1 — 1 epoch each (Control + Self-Correction)**

Perfect call.

**Why:**

* We want **pickup detection**, not saturation
* Short runs reduce:

  * Overfitting
  * Spurious fluency effects
* Faster iteration if adjustments are needed

This keeps M46 tight and interpretable.

---

## Q5: Control Dataset Treatment

**Decision:**
✅ **Option 1 — Truly unchanged control dataset**

This is essential.

**Control must be:**

* Raw `stage_c.jsonl`
* No markers
* No restructuring
* No placeholders

That gives you a clean A/B:

> “Same data, same init, same steps — only verification structure differs.”

Anything else muddies causality.

---

## Q6: Evaluation Focus

**Decision:**
✅ **Primary: Verification frequency**
✅ **Secondary: Full behavioral analysis**

You have the framing exactly right.

**Primary metric (headline):**

* Does the model *attempt* verification more often?

**Secondary observations:**

* Does it hallucinate checks?
* Does it self-correct spuriously?
* Does it improve or harm final answers?

Accuracy is **context**, not the thesis.

---

# 🔒 FINAL LOCKED SUMMARY (M46)

| Dimension            | Decision                                   |
| -------------------- | ------------------------------------------ |
| Marker format        | Simplified `VERIFY:` / `CORRECT:` appended |
| Injected errors      | None (0%)                                  |
| Verification content | Template-based per problem type            |
| Training duration    | 1 epoch each                               |
| Control dataset      | Raw, unchanged Stage-C                     |
| Primary metric       | Verification frequency                     |

You are **fully cleared to proceed**.

---

# 🧠 Two Small Guardrails (Please Apply)

These are not changes — just safety rails.

### Guardrail 1: Do Not Overfit Templates

Keep templates **short and boring**.
If `VERIFY:` blocks become mini-reasonings, you will train verbosity, not verification.

### Guardrail 2: Log “False Verifications”

In metrics, explicitly track:

* `VERIFY present, but meaningless`
* `VERIFY contradicts reasoning`
* `VERIFY checks wrong invariant`

Negative signal is still signal — and judges respect honesty.

---

## ✅ Next Step

Proceed with M46 implementation exactly as outlined.

Once complete, we’ll be in an unusually strong position to say:

> “We didn’t just train a model to reason.
> We trained a model to *check itself* — and we can measure when it fails.”

That’s a rare and memorable claim.

When M46 finishes, I recommend we pause briefly before M47 to **decide how much ambition to layer on next** (observer vs failure topology).
