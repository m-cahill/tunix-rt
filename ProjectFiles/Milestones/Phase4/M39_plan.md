Below is a **Cursor-ready M39 prompt** that incorporates the full M38 audit + summary and explicitly pivots the project to a **local RTX 5090 run**, with an **optional PyTorch migration path**. This is written in a *“good-enough, get-it-done”* mode, not enterprise-grade.

I’ve grounded this directly in the M38 findings and constraints  so anyone reviewing the diff can see *why* this pivot is justified.

---

# 📌 Cursor Prompt — **M39: Local GPU Execution (RTX 5090) + Optional PyTorch Pivot**

## Milestone: M39

**Title:** Local GPU Execution on RTX 5090 (Close Kaggle TPU Path)
**Status:** Planned
**Predecessor:** M38 (TPU HBM OOM — closed as hardware-blocked)
**Philosophy:** *Good-enough > perfect*. Optimize for **successful execution + evidence**, not infrastructure purity.

---

## 🔴 Context (Do Not Skip)

M38 conclusively demonstrated that **full fine-tuning Gemma-2B (256K vocab)** is **not feasible on Kaggle TPU v5e-8** due to a hard HBM ceiling (16GB). Even with:

* `batch_size = 1`
* `seq_len = 64`
* `bfloat16`
* Adafactor
* Forced code-level overrides

XLA compilation still exceeds HBM by ~12–13GB.

**This is a hardware constraint, not a code bug.**
All TPU-side mitigations are already correct and should be preserved as documentation, not extended further.

👉 **Decision:** TPU path is now **closed**.
👉 **New execution target:** **Local RTX 5090 (32GB VRAM)**.

---

## 🎯 M39 Goals

### Primary Goal (Required)

1. **Successfully run training locally on RTX 5090**
2. **Produce real training artifacts** (loss, logs, checkpoints)
3. **Populate evidence files** currently left as templates in M38

### Secondary Goal (Optional, time-boxed)

4. **Evaluate PyTorch migration feasibility**

   * Only migrate if it *reduces friction*
   * No multi-milestone refactors unless strictly necessary

---

## 🧱 Phase 1 — Close the TPU Path (Documentation Only)

**Do NOT delete TPU code. Do NOT refactor further.**

Instead:

* Add a short **comment block** in:

  * `training/train_jax.py`
  * `training/configs/submission_tpu.yaml`
* State clearly:

  > “TPU execution blocked due to Gemma 256K vocab + 16GB HBM limit. See M38 audit.”

This preserves the engineering narrative for judges/reviewers.

---

## 🚀 Phase 2 — Local GPU (RTX 5090) Execution (JAX Path)

### 2.1 Environment Assumptions

* CUDA + cuDNN installed locally
* RTX 5090 ≈ 32GB VRAM
* Single-GPU execution is acceptable

### 2.2 Config Work

Create a **new config**:

```text
training/configs/submission_local_gpu.yaml
```

Base it on:

* `submission_tpu.yaml` (memory-safe defaults)
* Relax constraints **only where safe**

Suggested starting values:

```yaml
device: gpu
batch_size: 1
max_seq_length: 128   # try 256 only if 128 passes
gradient_accumulation_steps: 4
dtype: bfloat16
optimizer: adafactor
```

⚠️ Do **not** increase batch size initially. VRAM is precious due to logits size.

---

### 2.3 Code Changes (Minimal)

In `train_jax.py`:

* Allow GPU runs to **skip TPU-forced overrides**
* Keep TPU overrides intact

No architectural changes. No abstractions. No flags explosion.

---

### 2.4 Execution

Run locally:

```bash
python training/run_train_jax.py \
  --config training/configs/submission_local_gpu.yaml \
  --dataset dev-reasoning-v2 \
  --device gpu \
  --output ./output/local_gpu_run
```

If this runs:

* Capture logs
* Capture loss
* Save checkpoint (even partial is fine)

---

## 📦 Phase 3 — Evidence Population (Critical)

Populate **for real**:

```
submission_runs/m39_v1/
├── run_manifest.json
├── eval_summary.json
├── training_log.txt
```

Required fields:

* Commit SHA
* Hardware = RTX 5090
* VRAM
* Config snapshot
* Training steps completed
* Final loss (even if partial)

This is **mandatory**. No placeholders.

---

## 🔄 Phase 4 — PyTorch Migration (OPTIONAL, Time-Boxed)

⚠️ Only attempt if JAX GPU run is painful or unstable.

### Evaluation Criteria

Switch **only if**:

* JAX GPU OOMs unpredictably
* Debug velocity is poor
* PyTorch offers faster iteration (likely)

### If Switching:

* Create **parallel** path:

  ```
  training_pt/
  ```
* Do **not** delete JAX code
* Use:

  * `AutoModelForCausalLM`
  * `torch.cuda.amp`
  * Gradient checkpointing if needed

### Scope Limit

* 1–2 milestones max
* No trainer framework rewrites
* No distributed training
* Single-GPU only

If PyTorch works faster → keep it.
If not → stop immediately.

---

## 🧠 Explicit Non-Goals (Do NOT Do These)

* ❌ No LoRA / PEFT (unless RTX 5090 still OOMs)
* ❌ No model parallelism
* ❌ No refactoring training abstractions
* ❌ No CI changes
* ❌ No Kaggle retries

---

## ✅ Definition of Done (M39)

M39 is **DONE** when:

* [ ] Training runs locally on RTX 5090
* [ ] Real logs + loss exist
* [ ] Evidence files are populated (no nulls)
* [ ] TPU path is documented as blocked
* [ ] PyTorch decision is explicitly recorded (yes/no)

---

## 🧭 Final Note to Cursor

> This milestone is about **execution closure**, not elegance.
> The architecture is already strong.
> The contest does not reward suffering.
> Ship a real run, document the constraint, and move on.

---

If you want, next we can:

* Estimate **exact milestone count** for a clean PyTorch switch (spoiler: probably **M40–M41 max**), or
* Draft a **short “judge-friendly” explanation** of why the TPU path failed and why that’s reasonable.
