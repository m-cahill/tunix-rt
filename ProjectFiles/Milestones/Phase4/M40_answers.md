Great set of clarifying questions — these are exactly the right ones to ask before touching anything. I’ll answer **decisively** so Cursor can proceed without ambiguity.

I’ll also bias toward **lowest risk, least churn**, and **fastest path to a clean GPU win**.

---

## ✅ M40 Clarifications — Final Decisions

### **1. Training Script Path Discrepancy**

**Decision:** **(B) Use the existing `training_pt/train.py` directly**

**Rationale:**

* M39 already validated this script
* Creating a wrapper adds risk and time for zero benefit
* For M40, we only care that **PyTorch runs on CUDA**, not architectural symmetry with JAX

📌 **Instruction for Cursor:**
Do **not** create `run_train_torch.py`. Invoke `training_pt/train.py` directly.

---

### **2. Virtual Environment Strategy**

**Decision:** **(B) Create a separate `.venv-gpu`**

**Rationale:**

* Nightly PyTorch + CUDA is inherently unstable
* We want a clean rollback path
* Preserves M39 CPU success as a safety net

📌 **Instruction for Cursor:**
Create `.venv-gpu` and do all M40 work inside it. Do **not** modify `.venv`.

---

### **3. CUDA Driver Version Confirmation**

**Decision:** Cursor should **assume CUDA ≥ 12.8 is required**, but **verify before installing PyTorch**.

📌 **Instruction for Cursor:**

1. Run:

   ```bash
   nvidia-smi
   ```
2. Capture:

   * Driver Version
   * CUDA Version (reported by driver)
3. Log this to `submission_runs/m40_*/env_info.txt`

If CUDA < 12.8 → **stop and report**, do not proceed blindly.

---

### **4. PyTorch Nightly Index URL**

**Decision:** **(A) Use `cu128` nightly if available**, otherwise fall back **once**, then stop.

**Order of operations:**

1. Try:

   ```bash
   pip install --pre torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
2. If unavailable → try `cu124`
3. If both fail → **do NOT build from source yet** (that’s a separate milestone if needed)

📌 **Important:**
Do **not** silently fall back to CPU wheels. Explicitly check `torch.cuda.is_available()`.

---

### **5. Config File to Use**

**Decision:** **(A) GPT-2 first for validation**, then optional Gemma

**Rationale:**

* GPU enablement ≠ production training
* GPT-2 proves:

  * CUDA kernels compile
  * Memory allocation works
  * Training loop is GPU-safe
* Gemma 2B can be validated **after** success

📌 **Instruction for Cursor:**

* Use `submission_local_gpu_gpt2.yaml` (or equivalent)
* Batch size = small
* Steps = 10–50 max
* This is a **smoke test**, not a benchmark

---

### **6. Evidence Directory Naming**

**Decision:** **Use existing convention: `submission_runs/m40_v1/`**

**Rationale:**

* Matches prior milestones
* Easier to reason about progression
* No semantic ambiguity

📌 **Required contents:**

```
submission_runs/m40_v1/
├── env_info.txt
├── torch_version.txt
├── cuda_check.txt
├── training_log.txt
├── nvidia_smi.txt
└── README.md
```

---

## 🔒 Final Cursor Execution Summary (Authoritative)

Cursor should proceed with the following **locked decisions**:

| Area            | Decision                               |
| --------------- | -------------------------------------- |
| Training script | Use `training_pt/train.py`             |
| Virtualenv      | Create `.venv-gpu`                     |
| CUDA            | Verify via `nvidia-smi`, require ≥12.8 |
| PyTorch         | Nightly `cu128` → fallback `cu124`     |
| Model           | GPT-2 first                            |
| Evidence        | `submission_runs/m40_v1/`              |

---

## 🎯 Strategic Note (Why This Is the Right Call)

You are doing exactly what strong competitors do:

* **De-risk infrastructure first**
* **Prove hardware utilization**
* **Delay polish until fundamentals are locked**
* **Avoid yak-shaving symmetry or purity**

Most teams will **fail silently on CPU** or get stuck fighting TPU quotas.
You’re eliminating that entire class of risk.

---

If you want, next I can:

* Convert this into a **single “BEGIN M40” Cursor command**
* Pre-write the **README.md template** for evidence
* Plan **M41 frontend polish** in a way that pairs cleanly with the demo video

Just tell me.
