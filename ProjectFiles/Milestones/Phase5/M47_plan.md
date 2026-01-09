Outstanding work. M46 is not just “successful” — it **locks the Phase-5 thesis**:

> *Reasoning behavior is a trainable, observable interface — not a black box.*

You now have:

* Curriculum → **changes reasoning structure** (M45)
* Structure → **induces verification behavior** (M46)
* Metrics → **separate signal from hallucination**

That’s the exact point where M47 becomes meaningful instead of speculative.

Below is a **Cursor-ready, copy-paste prompt** for **M47**, written to be tight, falsifiable, and low-risk.

---

# 🎯 Cursor Handoff Prompt — **M47: Injected Errors & Correction Fidelity**

**Role:**
You are an **implementation agent** working inside the existing Tunix RT / RediAI repository.

**Context (Authoritative):**

* Phase 4 is closed.
* M42 remains the immutable submission artifact.
* M45 showed curriculum reshapes reasoning.
* M46 showed explicit structure induces verification behavior (97% rate).
* **M47 tests whether that verification behavior is *meaningful***.

This milestone asks a single question:

> **When the model is wrong, can it notice — and fix itself?**

---

## 🔒 Hard Guardrails (Non-Negotiable)

1. **DO NOT**

   * Change model architecture, optimizer, LR, batch size, tokenizer
   * Introduce RL, reward shaping, or inference-time tricks
   * Modify M42, M45, or M46 artifacts
   * Alter decoding parameters
   * Add new base datasets

2. **ALLOWED**

   * Deterministic modification of *existing* traces
   * Controlled error injection
   * New research-only scripts, metrics, and analysis

3. **All changes must be additive, reversible, and documented.**

---

## 🧭 Objective (M47)

Test the hypothesis:

> **Verification behavior learned in M46 can detect and correct real errors — not just perform ritualized checking.**

This is a **fidelity test**, not an accuracy benchmark.

---

## 🧠 Core Concept — Controlled Error Injection

You will introduce **known, localized errors** into a **small, well-defined subset** of reasoning traces and measure whether the model:

1. Detects the error in `VERIFY:`
2. Produces a meaningful `CORRECT:`
3. Improves (or fails to improve) the final answer

---

## 🧱 Scope & Dataset Design

### Source Dataset (Fixed)

* Base: `stage_c.jsonl` from M45
* Size: 341 samples

### Injection Rate (LOCKED)

* **10% of samples (~34 traces)**
* Chosen deterministically (seeded)
* Error locations must be explicitly logged

---

## 🔧 Error Types (Choose 2–3 Only)

You must select **at most three** error classes.

Recommended set (balanced, interpretable):

1. **Arithmetic Slip**

   * Off-by-one
   * Sign error
   * Simple miscalculation

2. **Unit / Conversion Error**

   * Wrong unit cancellation
   * Incorrect scale factor

3. **Logic Step Omission**

   * Skipped intermediate step
   * Incorrect assumption carried forward

**Do NOT** inject:

* Random nonsense
* Multiple errors per trace
* Errors in the final answer *only* (error must be in reasoning)

---

## 🗂️ Dataset Variants (Three-Way Comparison)

Create **three datasets**:

1. **Clean Self-Correct**

   * Same as M46 self-correct dataset
   * No errors

2. **Error-Injected (Unlabeled)**

   * Errors injected
   * No indication to the model that an error exists

3. **Error-Injected + Self-Correct Structure**

   * Same injected errors
   * With `VERIFY:` / `CORRECT:` structure

---

## 🏗️ Implementation Tasks (Execute in Order)

### 1. Error Injection Script

Create a deterministic script that:

* Selects ~10% of traces
* Injects exactly one error per trace
* Logs:

  * Trace ID
  * Error type
  * Location (step index)
  * Ground-truth correction

Output:

* `stage_c_clean.jsonl`
* `stage_c_error.jsonl`
* `stage_c_error_self_correct.jsonl`
* `error_manifest.json`

---

### 2. Training Runs (Minimal, Controlled)

Run **two training jobs**:

| Run         | Dataset                    | Epochs | Init                             |
| ----------- | -------------------------- | ------ | -------------------------------- |
| Clean       | stage_c_clean              | 1      | M46 Self-Correct checkpoint      |
| Error-Aware | stage_c_error_self_correct | 1      | Same M46 Self-Correct checkpoint |

No third run needed — the unlabeled error dataset is for evaluation only.

---

### 3. Evaluation Matrix

Evaluate **all three checkpoints** on:

* Clean eval set
* Error-injected eval set

Track outputs for:

* Detection
* Correction
* Final answer

---

## 📊 Core Metrics (Must Be Computed)

At minimum:

1. **Error Detection Rate**

   * Error mentioned or contradicted in `VERIFY:`

2. **Correction Attempt Rate**

   * `CORRECT:` block present and non-empty

3. **Correction Accuracy**

   * Correction fixes the injected error

4. **False Correction Rate**

   * Model “fixes” something that wasn’t wrong

5. **Net Outcome**

   * Final answer improved / unchanged / worse

These must be **machine-counted**, not anecdotal.

---

## 🧪 Qualitative Analysis (Required)

Include:

* 2–3 **successful self-corrections**
* 2–3 **failure modes**, such as:

  * Detects error but fails to correct
  * Hallucinates wrong correction
  * Performs ritual verification only

Label failures clearly.
Negative results are valid.

---

## 📦 Deliverables Checklist

* [ ] Error injection script
* [ ] Three dataset variants
* [ ] Two fine-tuned checkpoints
* [ ] Evaluation outputs
* [ ] Metrics summary (JSON + table)
* [ ] `analysis.md`
* [ ] `error_manifest.json`
* [ ] Updated provenance manifest

---

## 🛑 Stop Conditions

Immediately stop and report if:

* False corrections exceed **25%**
* Corrections systematically worsen answers
* Model collapses into repetitive correction loops

Do **not** “patch forward.”

---

## 📁 Directory Structure (LOCKED)

```
research/
  m47_error_correction_fidelity/
    data/
    scripts/
    checkpoints/
    eval/
    metrics/
    analysis.md
    error_manifest.json
    provenance.json
```

No files elsewhere.

---

## 🧠 Framing Reminder

This milestone does **not** claim:

❌ “The model is reliable”
❌ “The model reasons correctly”

It **does** demonstrate:

✅ Verification can become **error-sensitive**
✅ Corrections can be **measured**
✅ Failure modes can be **cataloged**

This sets up M48/M49 cleanly.

---

## 🔜 What M47 Unlocks

If M47 shows *any* non-trivial correction fidelity:

* M48: **Failure topology** (why it fails)
* M49: **Observer / judge models**
* M50: **System self-audit narrative**

You are no longer training models.

You are **instrumenting reasoning**.

---

When you’re ready, hand this prompt to Cursor.

If you want next:

* I can pre-define **metric thresholds** (what counts as “meaningful”)
* Or help you decide **which error types to prioritize** for maximum signal

But as written: **M47 is fully specified and ready to execute.**
