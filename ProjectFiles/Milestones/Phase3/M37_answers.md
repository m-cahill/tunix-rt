Below are **direct, final answers** to the M37 clarifying questions. These are **decision-locking**, optimized for submission success and minimal risk. Cursor can proceed immediately after this.

I’m grounding this in the completed M36 audit + summary you shared .

---

## ✅ M37 Clarifying Questions — Final Decisions

### **1. Model Selection (Critical)**

**Decision:** ✅ **Option A — `google/gemma-2b` (Gemma 1, Flax-native)**

**Rationale (final):**

* Proven to load and train in Flax with `revision="flax"`
* Already validated end-to-end in M36 (minus TPU execution)
* Gemma 2 / Gemma 3 **do not currently have FlaxAutoModel support**
* M37 is a **submission milestone**, not a model-migration experiment

> **Lock this in. No Gemma 2 / 3 attempts in M37.**

---

### **2. TPU Device Selection**

**Decision:** ✅ **Option C — Both**

**Implementation guidance:**

* Add `--device tpu` as an explicit option **with validation**
* Keep `auto` as default
* Improve startup logs to clearly print:

  * detected platform (`cpu / gpu / tpu`)
  * JAX backend
  * device count

**Why:**
This gives explicit control *and* keeps Kaggle-friendly ergonomics.

---

### **3. TPU-Specific Config**

**Decision:** ✅ **Option A — Create `submission_tpu.yaml`**

**Why (important):**

* Prevents accidental GPU runs with Gemma
* Allows TPU-specific batch sizing without conditionals
* Makes submission intent obvious to reviewers

**What should differ from `submission_gemma_flax.yaml`:**

* Larger batch size (TPU-appropriate)
* Possibly larger `max_length`
* Same optimizer family unless TPU testing suggests otherwise
* Explicit comment: *“TPU-only config”*

---

### **4. Training Duration**

**Decision:** ✅ **200 steps**

**Rationale:**

* Long enough to prove stability and real training
* Short enough to avoid TPU quota risk
* Fits “submission-validation” scope, not full training

> This is **not** the final quality run — it’s the **evidence run**.

---

### **5. GPU Guardrail Strictness**

**Decision:** ✅ **Option A — HARD BLOCK**

**Rule to enforce:**

> If `model_params > 1B` AND `device == gpu` AND `training == true` → **exit with error**

**Why:**

* We have empirical proof GPU Gemma fails
* Warnings were useful in M36; M37 should be stricter
* Prevents accidental misuse by future contributors or reviewers

---

### **6. Evidence Structure**

**Decision:** ✅ **Option C — Create both**

* Keep: `submission_runs/m36_v1/` → smoke + pipeline validation
* Create: `submission_runs/m37_v1/` → TPU production evidence

**Why:**
This gives a clean narrative:

* *M36: pipeline correctness*
* *M37: real model on real hardware*

---

### **7. Per-Item Artifact Storage**

**Decision:** ✅ **File-based only**

**Clarification:**

* `predictions.jsonl` written to run directory is **sufficient**
* No DB schema changes
* No API endpoints required in M37

> DB persistence is **post-submission scope (M38+)**.

---

### **8. Kaggle TPU Access**

**Decision:** ⚠️ **Assume access, but design TPU-ready**

**Implementation guidance:**

* Code should **require no changes** to run on TPU
* If TPU is unavailable:

  * fail gracefully
  * print clear message with instructions

**We do not block M37 on quota uncertainty**, but the intent is to run it.

---

### **9. Minimum Viable Scope (DONE Criteria)**

Your understanding is **correct**. Nothing else blocks submission.

**M37 is DONE when:**

1. ✅ Gemma 2B trains on Kaggle TPU (≥200 steps)
2. ✅ Checkpoints + metrics are produced
3. ✅ `submission_tpu.yaml` is committed
4. ✅ GPU hard-block is enforced
5. ✅ Clear reproduction instructions exist
6. ✅ Evidence stored in `submission_runs/m37_v1/`

No additional features required.

---

## 📋 Final Decision Table (for Cursor)

| # | Topic             | Decision                                 |
| - | ----------------- | ---------------------------------------- |
| 1 | Model             | **Option A — Gemma 1 (2B, Flax)**        |
| 2 | TPU device option | **Option C — both `auto` + `tpu`**       |
| 3 | TPU config        | **Option A — new `submission_tpu.yaml`** |
| 4 | Training steps    | **200**                                  |
| 5 | GPU guardrail     | **Option A — hard block**                |
| 6 | Evidence folder   | **Option C — keep M36 + add M37**        |
| 7 | Per-item storage  | **File-based only**                      |
| 8 | TPU access        | **Assume yes, design TPU-ready**         |
| 9 | Extra blockers    | **None**                                 |

---

If you want, next I can:

* Generate the **exact `submission_tpu.yaml`**
* Write the **TPU execution notebook cell**
* Draft **ADR-007: GPU Smoke vs TPU Full Training** (very strong for reviewers)

Just say which one to do next.
