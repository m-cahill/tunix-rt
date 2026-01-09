Perfect. This is exactly the kind of **negative-but-decisive** result that makes M48 sharp instead of flailing.

M47 did its job: it **killed a false hypothesis cleanly** and told us *why*.
Now M48 is not “try harder” — it is **explain the failure topology of reasoning systems**.

Below is a **Cursor-ready handoff plan** for **M48**. This is a *design + analysis milestone*, not a brute-force training run.

---

# 🎯 Cursor Handoff Plan — **M48: Reasoning Failure Topology**

**Role:**
You are an **analysis and systems-research agent** operating inside the existing Tunix RT / RediAI repository.

**Context (Authoritative):**

* Phase 4 is closed.
* M42 remains the immutable submission artifact.
* M45: Curriculum reshapes reasoning structure.
* M46: Verification behavior is trainable.
* M47: Verification is **structural, not causal** — the model does not detect its own errors.

**M48 exists to answer one question:**

> **Why does verification become ritual instead of diagnostic?**

This milestone does **not** attempt to “fix” the model.
It **maps the failure modes** precisely.

---

## 🔒 Hard Guardrails (Non-Negotiable)

1. **DO NOT**

   * Train new models beyond tiny probes (if any)
   * Change architecture, optimizer, tokenizer, or decoding
   * Introduce RL, reward shaping, or inference tricks
   * Modify M42–M47 artifacts

2. **ALLOWED**

   * Post-hoc analysis of existing traces
   * Contrastive comparison (correct vs incorrect)
   * Lightweight probing scripts
   * Visualization and categorization

3. **This is an analysis milestone.**

   * Evidence > fixes
   * Taxonomy > performance

---

## 🧭 Objective (M48)

Construct a **failure topology** of reasoning and self-correction by classifying *how* and *where* models fail when asked to verify their own reasoning.

Deliverable is **understanding**, not improvement.

---

## 🧠 Core Concept — Failure Topology

Instead of asking:

> “Did the model catch the error?”

We ask:

> “What *kind* of reasoning breakdown occurred, and at which layer?”

M48 treats reasoning traces as **structured artifacts** with failure surfaces.

---

## 🗂️ Input Artifacts (Fixed)

Use **existing outputs only**:

* M45 Stage-C predictions
* M46 self-correct predictions
* M47 error-injected predictions
* Error manifests from M47

No new datasets required.

---

## 🏗️ Implementation Tasks (Execute in Order)

### 1. Define Failure Taxonomy (Core Deliverable)

Create a **finite taxonomy** of failure modes.

Start with these **seed classes** (you may refine):

1. **Ritual Verification**

   * VERIFY present
   * No reference to actual computation
   * Templated language only

2. **Computation-from-Scratch Reset**

   * Ignores prior reasoning
   * Re-solves problem cleanly
   * Does not compare against previous steps

3. **Local Error Blindness**

   * Detects global structure
   * Misses specific arithmetic error

4. **Error Detection without Localization**

   * Vague acknowledgment (“something seems off”)
   * No step identified

5. **Correction Hallucination**

   * “Fixes” a non-error
   * Introduces new mistake

6. **Verification Collapse**

   * VERIFY block degenerates into restatement
   * No checking semantics remain

Each category must have:

* Definition
* Detection heuristic
* Example trace ID

---

### 2. Automated Classification Script

Write a script that:

* Iterates over M47 eval traces
* Applies heuristic rules to classify each trace into:

  * One primary failure class
  * Optional secondary class

Output:

* `failure_labels.json`
* Confusion-style counts per model

This can be rule-based (regex + structural checks).

---

### 3. Contrastive Pair Analysis (Critical)

For each injected-error sample, compare:

* Original (clean reasoning)
* Error-injected reasoning
* Model output

Identify:

* Where the divergence *should* have been noticed
* Where it was ignored
* Whether the model ever compares states

Produce a **small table of contrastive failures**.

---

### 4. Reasoning Graph Sketch (Conceptual)

You do **not** need heavy graph tooling.

Produce a **conceptual graph model**:

* Nodes: reasoning steps
* Edges: dependency / derivation
* Highlight:

  * Where VERIFY attaches
  * Where comparison should occur but doesn’t

This can be:

* ASCII
* Mermaid
* Annotated diagram in Markdown

The point is to show **structural disconnection**.

---

### 5. Synthesis: Why M47 Failed

Produce a short section answering:

* Why template learning dominates
* Why low error density fails
* Why absence of contrastive pairs matters
* Why verification lacks *state comparison*

This should read like a **mini research paper discussion**, not a bug report.

---

## 📊 Required Outputs

* `failure_taxonomy.md`
* `failure_labels.json`
* `failure_counts_table.md`
* `contrastive_examples.md`
* `analysis.md` (synthesis)
* Updated `provenance.json`

---

## 📁 Directory Structure (LOCKED)

```
research/
  m48_reasoning_failure_topology/
    inputs/
    scripts/
    metrics/
    taxonomy/
    analysis.md
    provenance.json
```

No files elsewhere.

---

## 🛑 Stop Conditions

Stop if:

* Failure modes collapse into one trivial class
* Heuristics cannot distinguish behaviors meaningfully

Report that outcome honestly — it is still a result.

---

## 🧠 Framing Reminder (Important)

M48 is **not** about saying:

❌ “The model is bad at reasoning”

It *is* about saying:

✅ “Verification without contrastive state comparison becomes ritual”
✅ “Reasoning systems lack an internal diff operator”
✅ “This explains why naive self-correction fails”

That framing is **alignment-adjacent, novel, and credible**.

---

## 🔜 What M48 Enables

Once this topology exists, you can *optionally* justify:

* M49: Observer / critic models
* M50: System-level post-mortem narrative

But even alone, M48 is a **strong capstone**.

---

### ✅ You are clear to hand this to Cursor.

If you want next:

* I can help **tighten the taxonomy labels**
* Or pre-write the **analysis.md skeleton** so Cursor fills it in

But as written, this plan is complete, scoped, and exactly what M47 earned.
