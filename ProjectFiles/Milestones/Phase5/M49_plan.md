This is a *very* strong place to be. M48 doesn’t just conclude Phase 5 research — it **earns** the right to do M49 properly.

M49 is no longer “try to fix self-correction.”
It is now:

> **Demonstrate that error detection requires a *separate observer*, not more prompting of the generator.**

Below is a **Cursor-ready, handoffable plan** for **M49**, written to be disciplined, scoped, and intellectually honest.

---

# 🎯 Cursor Handoff Prompt — **M49: Observer Model for Error Detection**

**Role:**
You are an **analysis + lightweight modeling agent** working inside the Tunix RT / RediAI repository.

**Context (Authoritative):**

* Phase 4 is closed.
* M42 is the immutable submission artifact.
* Phase 5 results so far:

  * **M45:** Curriculum reshapes reasoning structure
  * **M46:** Verification behavior is trainable
  * **M47:** Verification is not causal
  * **M48:** Failure topology shows lack of state-difference operator

**M49 tests the hypothesis implied by M48:**

> **Error detection is not a generation behavior — it is an observation behavior.**

This milestone introduces an **external observer** that evaluates reasoning *after* it is produced.

---

## 🔒 Hard Guardrails (Non-Negotiable)

1. **DO NOT**

   * Modify or retrain the generator model
   * Change architecture, optimizer, tokenizer, decoding params
   * Introduce RL, reward shaping, or self-play
   * Alter M42–M48 artifacts

2. **ALLOWED**

   * Train a **separate lightweight observer**
   * Use existing reasoning traces + error manifests
   * Post-hoc analysis only

3. **This is a diagnostic milestone, not a fix.**

---

## 🧭 Objective (M49)

Demonstrate that:

> **A simple observer model can detect errors in reasoning traces that the generator cannot detect itself.**

This validates the architectural claim from M48:

* Self-correction fails because generation lacks comparison
* Observation can succeed without changing generation

---

## 🧠 Core Concept — Generator vs Observer Split

* **Generator:** Produces reasoning (unchanged)
* **Observer:** Reads reasoning + answer and predicts:

  * Is there an error?
  * Where is the error (optional)?
  * Confidence score

The observer **does not generate text** — it classifies.

---

## 🗂️ Input Data (Fixed)

Use **only existing artifacts**:

* M47 error-injected traces
* M47 clean traces
* Error manifests from M47
* Reasoning steps + final answer text

No new synthetic data.

---

## 🏗️ Implementation Tasks (Execute in Order)

### 1. Observer Task Definition

Define the observer’s task clearly:

**Inputs (examples):**

* Full reasoning trace (text)
* Final answer
* (Optional) original question

**Outputs:**

* `error_present`: {0,1}
* `confidence`: [0.0 – 1.0]
* (Optional) `suspected_step_index`

Keep this minimal.

---

### 2. Dataset Construction

Build a dataset with labels from `error_manifest.json`:

| Field      | Description                     |
| ---------- | ------------------------------- |
| input_text | Reasoning + answer              |
| label      | error_present (0/1)             |
| metadata   | error type, step index (if any) |

Split:

* Train: 70%
* Validation: 15%
* Test: 15%

Seeded, deterministic split.

---

### 3. Observer Model Choice (Lightweight)

Choose **one** simple approach:

**Allowed options:**

1. Logistic regression over embeddings
2. Small frozen LM + classification head
3. Linear probe on hidden states (if easy)

**Disallowed:**

* Large fine-tunes
* Multi-task heads
* Anything opaque or heavy

The point is **capability separation**, not power.

---

### 4. Training & Evaluation

Train the observer only.

Metrics to report:

* Accuracy
* Precision / Recall
* False positive rate
* ROC-AUC (if applicable)

Compare against:

* Generator’s self-correction (0% detection)
* Random baseline

---

### 5. Contrastive Demonstration (Critical)

Show **side-by-side examples**:

| Case        | Generator Output       | Observer Prediction              |
| ----------- | ---------------------- | -------------------------------- |
| Clean trace | “No correction needed” | No error (low confidence)        |
| Error trace | “No correction needed” | Error detected (high confidence) |

This is the *money shot* of M49.

---

## 📊 Required Outputs

* `observer_dataset.jsonl`
* `train_observer.py`
* `observer_metrics.json`
* `analysis.md`
* `provenance.json`

Optional:

* Confusion matrix plot

---

## 📁 Directory Structure (LOCKED)

```
research/
  m49_observer_error_detection/
    data/
    scripts/
    models/
    metrics/
    analysis.md
    provenance.json
```

No files elsewhere.

---

## 🛑 Stop Conditions

Stop if:

* Observer performs at chance
* Observer only memorizes surface templates
* Labels are inconsistent or ambiguous

Report honestly — negative result is still valid.

---

## 🧠 Framing Reminder (Extremely Important)

M49 does **not** claim:

❌ “The observer is smarter”
❌ “We solved reasoning”

It **does** show:

✅ Error detection is a **different function** than generation
✅ Verification failure is architectural, not mystical
✅ Reasoning systems benefit from **separation of roles**

This aligns perfectly with M48.

---

## 🔚 Phase 5 Narrative Arc (Now Visible)

At this point, Phase 5 reads cleanly:

* **M45:** Ordering changes reasoning
* **M46:** Structure induces verification
* **M47:** Verification ≠ detection
* **M48:** Why detection fails
* **M49:** Detection works when separated

That is a *complete research story*.

---

### ✅ You are clear to hand this to Cursor.

If you want, next I can:

* Draft the **analysis.md skeleton** for M49
* Help pick the **simplest observer model** for maximum clarity
* Or help decide whether M50 is needed or if M49 is the natural endpoint

But as written: **M49 is fully specified and ready to execute.**
