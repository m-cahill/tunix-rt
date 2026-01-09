Below is a **Cursor-ready, copy-paste prompt** to execute **M45 — Curriculum Reasoning Training** exactly as scoped.
It is written to be **unambiguous, low-risk, auditable, and end-to-end verifiable**, with explicit guardrails and stop conditions.

---

## 🎯 Cursor Handoff Prompt — **M45: Curriculum Reasoning Training**

**Role:**
You are an **implementation agent** working inside the existing Tunix RT / RediAI submission repository.

**Context (Authoritative):**

* **Phase 4 is CLOSED.**
* **M42 is the immutable Kaggle submission artifact. Do not modify or overwrite it.**
* M44 validated full production training on RTX 5090 using PyTorch + Gemma-2B.
* Flat SFT has reached diminishing returns.
* **This milestone is exploratory research**, not submission replacement.

**This milestone must be fully reproducible, auditable, and reversible.**

---

## 🔒 Hard Guardrails (Non-Negotiable)

1. **DO NOT**

   * Change model architecture
   * Change optimizer, LR schedule, batch size, or tokenizer
   * Add new datasets
   * Modify M42 artifacts
   * Introduce “magic” heuristics or adaptive training logic

2. **ALLOWED**

   * Dataset *partitioning only*
   * Sequential training runs
   * New configs, logs, and analysis artifacts
   * New code limited to **orchestration + analysis**

3. **All changes must be additive.**

   * No breaking refactors
   * No deletion of existing paths

---

## 🧭 Objective (M45)

Demonstrate that **curriculum ordering of reasoning data** produces **qualitative improvements in reasoning traces** compared to flat SFT — *without changing the model or optimizer*.

Success is measured by **trace structure**, not loss alone.

---

## 🧱 Milestone Scope

### Dataset (Fixed)

* Source dataset: `dev-reasoning-v2`
* Total samples unchanged
* No augmentation
* No synthetic generation

### Curriculum Partitioning

Partition the dataset **once**, deterministically, into **three tiers** based on **trace length and complexity**.

You must document the exact criteria used.

#### Required Tiers

**Stage A — Low Reasoning**

* Short answers
* Minimal or no explicit reasoning
* Direct responses

**Stage B — Medium Reasoning**

* Multi-step explanations
* Explicit justification
* Moderate trace length

**Stage C — Full Reasoning / Edge Cases**

* Long chains of thought
* Ambiguity handling
* Self-checking language
* Failure-prone examples

> If trace length is the only available signal, use it.
> If multiple signals exist, document priority order.

---

## 🏗️ Implementation Tasks (Execute in Order)

### 1. Dataset Analysis & Split

* Write a **single script or notebook** that:

  * Loads `dev-reasoning-v2`
  * Computes:

    * Trace length distribution
    * Any available metadata signals
  * Outputs:

    * `stage_a.jsonl`
    * `stage_b.jsonl`
    * `stage_c.jsonl`

**Artifacts Required**

* Histogram of trace lengths
* Counts per stage
* Split criteria documented in Markdown

---

### 2. Training Orchestration (Sequential)

Run **three sequential training phases**:

| Stage | Dataset | Epochs | Init Weights |
| ----- | ------- | ------ | ------------ |
| A     | stage_a | 2–3    | Base Gemma   |
| B     | stage_b | 2–3    | Checkpoint A |
| C     | stage_c | 2–3    | Checkpoint B |

**Rules**

* Identical optimizer + hyperparameters
* Save checkpoint after *each* stage
* No interleaving
* No retries unless infra failure

---

### 3. Logging & Provenance

For each stage:

* Save:

  * Training config
  * Exact dataset hash
  * Checkpoint ID
  * Loss curves
  * Wall-clock time

Create a **single provenance file** tying all three stages together.

---

### 4. Evaluation & Comparison

Using a **fixed evaluation set**:

* Generate model outputs for:

  * Baseline flat SFT checkpoint (from M44)
  * Post-Stage-A
  * Post-Stage-B
  * Post-Stage-C

**Do NOT cherry-pick examples.**

---

### 5. Qualitative Analysis (Critical)

Produce a short analysis artifact answering:

* How do traces change *structurally*?
* Evidence of:

  * Earlier verification language
  * Reduced hallucinated steps
  * Better step ordering
* Where the curriculum helps
* Where it fails

This can be:

* Markdown + tables
* Annotated examples (minimal)

---

## 📦 Deliverables Checklist (Must All Exist)

* [ ] Dataset split script/notebook
* [ ] `stage_a.jsonl`, `stage_b.jsonl`, `stage_c.jsonl`
* [ ] Training configs for all three stages
* [ ] Three sequential checkpoints
* [ ] Provenance / run manifest
* [ ] Evaluation outputs (all checkpoints)
* [ ] Short qualitative analysis document

---

## 🧪 Definition of Done (M45)

M45 is complete when:

* Curriculum training runs end-to-end without manual intervention
* Artifacts prove **ordering**, not hyperparameters, changed behavior
* We can point to **specific trace differences**
* Everything is reproducible from scratch

---

## 🛑 Stop Condition

If:

* Training diverges
* Outputs regress catastrophically
* Dataset split is ambiguous

**STOP and report findings. Do not “fix” silently.**

---

## 📌 Naming & Organization

Create a top-level directory:

```
research/
  m45_curriculum_reasoning/
    data/
    configs/
    checkpoints/
    eval/
    analysis.md
    provenance.json
```

No files go elsewhere.

---

## 🧠 Reminder

This is **not leaderboard chasing**.

This is about demonstrating:

> *We can shape how a model reasons by structuring information — not by brute force.*

Proceed carefully.
Produce evidence.
Prefer clarity over cleverness.

---

When you’re ready, I can next:

* Lock **exact split thresholds**
* Draft the **analysis rubric**
* Or prepare **M46 (structured self-correction)** immediately after M45 completes
