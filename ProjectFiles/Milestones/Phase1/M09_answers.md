Below are **clear, implementation-level answers to Q1–Q13** so Cursor can proceed without ambiguity. These choices deliberately **extend M8 instead of reinventing it**, keep CI green, and make M09 a *real but bounded* training milestone.
(Reference context: your attached **M09 questions** )

---

## Phase Structure & Overlap with M8

### **Q1: Schema overlap (TrainingExample vs Dataset/Trace schemas)**

**Decision: A — create `training/schema.py` with `TrainingExample` (new abstraction).**

**Why**

* Dataset manifests (`schemas/dataset.py`) describe *collections*.
* Traces (`schemas/trace.py`) describe *reasoning artifacts*.
* **TrainingExample** is a *derived, training-time unit* (prompt/response pair).
* Keeping it separate avoids polluting runtime schemas with training-only concerns.

**Rule**

* `TrainingExample` is *produced from* traces/datasets, never stored in DB.

---

### **Q2: Gemma IT formatter overlap**

**Decision: A — extend/refactor existing `training/renderers.py`.**

**Why**

* M8 already introduced `render_tunix_sft_prompt()` correctly.
* Creating a parallel `gemma_format.py` would fragment logic.
* Instead:

  * Keep **high-level renderers** in `renderers.py`
  * Add **low-level helpers** (token blocks, role formatting) *inside the same module* or a submodule.

**Rule**

* One place owns Gemma formatting logic.
* Tests should snapshot outputs so changes are explicit.

---

### **Q3: Export functionality overlap**

**Decision: B — extend the existing dataset build/export system.**

**Why**

* M8 already solved dataset selection, manifests, and export paths.
* M09 should **reuse**:

  * `POST /api/datasets/build`
  * `GET /api/datasets/{dataset_key}/export.jsonl`
* M09 adds:

  * a *new export format* (training examples)
  * not a new pipeline.

**Rule**

* No standalone exporters that bypass dataset manifests.

---

## Phase 3: Training Runner Structure

### **Q4: Training folder location**

**Decision: C — hybrid approach.**

* **`backend/training/`**
  → library code (schema, renderers, exporters, eval helpers)
* **Top-level `training/`**
  → *scripts* (`train_sft_tunix.py`, configs, README)

**Why**

* Keeps backend package clean and importable.
* Keeps scripts runnable without app context.
* Matches your “portable, well-structured” rule.

---

### **Q5: Tunix installation source**

**Decision**

* **Install Tunix from GitHub**, pinned to a **commit SHA**.
* Treat it exactly like UNGAR (optional, reproducible).

**Rationale**

* Tunix is not reliably distributed via PyPI yet.
* Commit pinning preserves determinism and auditability.

**Doc guidance**

```bash
pip install git+https://github.com/google-deepmind/tunix.git@<PINNED_SHA>
```

---

## Phase 4: Evaluation & Import

### **Q6: Post-training trace import**

**Decision: B — add a bulk endpoint `POST /api/traces/batch`.**

**Why**

* Eval produces *many* traces at once.
* Reusing single-trace POST would be slow and noisy.
* File-upload import is overkill for M09.

**Scope**

* Minimal batch endpoint:

  * accepts list of trace payloads
  * validates each
  * inserts transactionally

---

### **Q7: Evaluation storage strategy**

**Decision: C — files first, DB later (M10).**

**Why**

* M09 goal is *loop validation*, not analytics.
* Files in `artifacts/training_runs/<run_id>/` are:

  * auditable
  * reproducible
  * zero schema churn

---

## Recipe & Format Details

### **Q8: Training recipe definition**

**Decision: C — store recipe version in the dataset manifest.**

**Implementation**

* Manifest includes:

  ```json
  {
    "recipe": "trace_sft_v1",
    "instruction": "Show your work as steps"
  }
  ```

**Why**

* Avoid hardcoding.
* Enables future recipes (e.g., policy-only, critique-style).

---

### **Q9: System role handling**

**Decision: B — configurable via training config (YAML).**

**Why**

* Gemma IT has no `system` role.
* Instructions must be injected into the first user turn.
* Config-driven lets you experiment without code churn.

**Rule**

* Formatter *reads* config; it does not invent instructions.

---

## Determinism & Reproducibility

### **Q10: Seed management**

**Decision: B — auto-generate if not provided, then record.**

**Rule**

* Seed is mandatory in manifests.
* If user omits it:

  * generate once
  * persist everywhere (dataset, training run, eval).

---

## Scope Confirmation

### **Q11: Actual training in M09**

**Decision: B — run a *real* tiny SFT job (10–50 steps).**

**Why**

* M09’s value is proving the loop is *real*, not theoretical.
* Keep it:

  * optional dependency
  * local/manual or optional CI
* This aligns with the M8 audit’s recommendation.

**Guardrail**

* Never block main CI on training success.

---

### **Q12: Eval set source**

**Decision: A — static `eval_v1.jsonl`.**

**Why**

* Deterministic.
* Auditable.
* Decoupled from DB state and migration history.

You can add DB-driven evals in M10.

---

## CI / Coverage Strategy

### **Q13: Coverage impact of training code**

**Decision: A — omit `training/` from default coverage.**

**Why**

* Training scripts are not runtime-critical.
* You already established the precedent in M8.
* Add coverage via optional workflows if needed.

**Rule**

* Core backend coverage remains meaningful and gated.
* Training code is validated via unit tests + smoke workflows.

---

## Final Direction (TL;DR)

* **Extend M8, don’t fork it**
* **Introduce TrainingExample as a derived abstraction**
* **Run real (tiny) Tunix SFT in M09**
* **Files > DB for eval in v1**
* **Determinism everywhere**
* **CI stays green, optional depth via side workflows**

If you want, next I can:

* convert these answers into a **single Cursor TODO list**, or
* generate the **exact file tree + stub code** for M09 so Cursor can start coding immediately.
