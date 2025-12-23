Great set of questions. I’ll answer them **directly and decisively** so you can unblock Cursor and move M12 forward cleanly, without scope creep or architectural regret.

Below are **authoritative answers** you can treat as *project decisions*, not suggestions.

---

## M12 Decisions & Answers

---

### **1. Tunix Installation Source**

**Decision:** **Mock-first. Do NOT depend on a real Tunix install in M12.**

* **Do not** assume `google-tunix` is available or stable on PyPI.
* **Do not** pin to a Git SHA yet.
* **Do not** import Tunix runtime APIs.

**What to do instead (M12 scope):**

* Treat Tunix as an **external consumer** of artifacts we generate.
* Emit **JSONL + manifest files** that are *Tunix-compatible*, without importing Tunix.

📌 **Rationale**

* This exactly mirrors the successful UNGAR pattern.
* Keeps default CI green.
* Avoids guessing about Google-internal packaging.
* Allows users with Tunix installed to run artifacts manually.

➡️ **Real Tunix runtime integration belongs in M13+, not M12.**

---

### **2. Tunix “Run Manifest” Structure**

**Decision:** **Manifest is a lightweight, best-effort config — not a validated Tunix CLI contract.**

Treat the manifest as:

* A **developer convenience artifact**
* Human-readable
* Executable *by convention*, not enforcement

**Manifest format: YAML (preferred)**

**Required fields (M12):**

```yaml
version: "1.0"
runner: tunix
mode: sft

model:
  model_id: <string>

dataset:
  format: tunix_sft
  path: <path to jsonl>

training:
  learning_rate: float
  num_epochs: int
  batch_size: int
  max_seq_length: int

output:
  output_dir: <path>
```

📌 **Rationale**

* This is enough for any reasonable Tunix SFT workflow.
* Avoids locking into undocumented CLI flags.
* Future-proof: can be evolved without breaking M12.

➡️ **Do NOT attempt to validate this against Tunix CLI yet.**

---

### **3. JSONL Export Format**

**Decision:** **Reuse `tunix_sft`. No new format.**

Choose:

✅ **Option A — reuse `tunix_sft`**

* Already Gemma-aligned
* Already reasoning-aware
* Already tested and covered
* Already used in M09/M10

❌ Do **not** create `tunix_export`
❌ Do **not** default to `training_example`

📌 **Rule**

> M12 exports existing high-quality data, it does not invent new schema.

---

### **4. Frontend Panel Scope**

**Decision:** **UNGAR-level minimalism. No dataset browser.**

Include **only**:

* Tunix availability status
* Text input:

  * Trace IDs **or**
  * Dataset ID (choose **dataset ID**, since M10 established datasets)
* Model ID input
* Button: **Export JSONL**
* Button: **Generate Manifest**

**Defaults allowed and encouraged.**

❌ No hyperparameter UI beyond defaults
❌ No dataset browsing UI
❌ No preview tables

📌 **Rationale**

* This is a power-user bridge, not a product UI.
* UNGAR panel precedent applies perfectly.

---

### **5. Hyperparameters for Manifest**

**Decision:** **Exactly four, with defaults. No more.**

Include:

| Param            | Default |
| ---------------- | ------- |
| `learning_rate`  | `2e-5`  |
| `num_epochs`     | `3`     |
| `batch_size`     | `8`     |
| `max_seq_length` | `2048`  |

📌 **Rules**

* Optional overrides via request payload
* Frontend may expose **none or all**, your choice
* Backend must apply defaults if omitted

---

### **6. CI Workflow Timing**

**Decision:** **Create the optional Tunix workflow early — non-blocking.**

Choose:

✅ **Option A**

* Add `.github/workflows/tunix-integration.yml`
* Mark job with `continue-on-error: true`
* Guard with `if: inputs.run_tunix == true` or similar

📌 **Rationale**

* Makes integration visible immediately
* No risk to main CI
* Matches UNGAR strategy
* Enables early adopters

---

### **7. Completion Expectation**

**Decision:** **Phases 0–4 only. Stop before CI + docs finalization.**

**M12 deliverable cutoff:**

* Backend endpoints
* Manifest generation
* JSONL export
* Frontend panel

⛔ **Pause before:**

* CI workflow hardening
* Long-form documentation

📌 This gives you a **review checkpoint** before locking public contracts.

---

### **8. Tunix Import Strategy**

**Decision:** **Mock-first, zero Tunix runtime imports.**

Choose:

✅ **Option B**

Implementation rules:

* No `import tunix`
* No runtime execution
* No CLI invocation
* No dependency pinning

Backend behavior:

* If Tunix not installed → still works
* Export + manifest generation **never** require Tunix
* Availability endpoint may simply say:

  > “Artifacts compatible with Tunix. Runtime not required.”

📌 **This is the most important M12 architectural decision.**

---

## Final M12 Scope (Locked)

### M12 IS:

* A **bridge milestone**
* A **data + config emitter**
* A **handoff layer** between Tunix-RT and Tunix

### M12 IS NOT:

* A training runner
* A Tunix client
* A TPU orchestrator
* A benchmarking milestone

---

## What Happens After M12

* **M13**: Real Tunix execution hooks (optional)
* **M14**: Training result ingestion
* **M15**: Evaluation loop closes (trace → train → compare)

---

## Cursor Instruction Summary (Paste This)

> Implement M12 as a **mock-first Tunix integration**:
>
> * Reuse `tunix_sft` export format
> * Generate YAML run manifests (no Tunix runtime dependency)
> * Add backend endpoints + minimal frontend panel
> * Keep default CI green
> * Follow UNGAR optional-dependency pattern exactly
> * Stop after Phases 0–4 and await review

If you want, next I can:

* Convert this into a **Cursor-ready step-by-step TODO**
* Draft the **exact endpoint signatures**
* Write the **manifest builder function signature**
* Or outline **M13+ roadmap** once M12 lands

You’re in a very strong architectural position right now.
