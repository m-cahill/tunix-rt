Perfect set of questions. I’ll answer **decisively** so Cursor can proceed without ambiguity, and I’ll explicitly mark **authoritative choices** vs **optional helpers**.

---

# M42 Clarifications — Authoritative Answers

## 1. Video Recording

### **A1.1 — Manual vs automation**

**Manual recording.**
Cursor should **not** attempt to record video.

However, Cursor **should**:

* Prepare **video checklist**
* Prepare **shot order + timing**
* Prepare **fallback screenshots / commands** in case something flakes live

➡️ Treat video as **human-executed, AI-prepared**.

---

### **A1.2 — Recording tool**

You will handle tooling manually (OBS / Loom / etc.).

➡️ Cursor does **not** need to configure or test recording software.

---

### **A1.3 — Which doc is authoritative**

**`docs/DEMO.md` is authoritative.**

The older:

* `docs/video_script_m31.md`
* `docs/video_shotlist_m31.md`

➡️ **Do not update or revive them.**
They are historical artifacts.

Cursor should:

* Create **`docs/submission/VIDEO_CHECKLIST.md`**
* Explicitly state: *“This checklist + docs/DEMO.md is the source of truth.”*

---

## 2. Submission Package

### **A2.1 — Target format**

**ZIP archive is the authoritative submission artifact.**

Kaggle-side notebook / UI actions are **out of scope** for Cursor.

➡️ Deliverable:

```
submission_runs/m42_v1/tunix_rt_submission_m42.zip
```

---

### **A2.2 — Existing packaging script**

Yes — **inspect `backend/tools/package_submission.py`**.

Decision rule:

* If it is **clean, deterministic, and low-risk** → reuse / lightly update
* If it is **complex, stale, or risky** → **do NOT refactor**
  Create a **new minimal script** under `scripts/`

➡️ Bias toward **lowest-risk path**, not reuse-for-principle.

---

### **A2.3 — Deadline**

No hard external deadline enforced in-repo.

Treat M42 as:

* **Submission-ready immediately upon completion**
* No time-based pressure that justifies cutting corners

---

## 3. Dependency Locking

### **A3.1 — Backend (`uv.lock`)**

* **Yes**, `uv.lock` is authoritative
* **No**, do **not** add redundant pins to `pyproject.toml`

➡️ Instead:

* Document that `uv.lock` is the source of truth
* Capture `pip freeze` as **evidence**, not enforcement

---

### **A3.2 — Frontend**

`package-lock.json` is sufficient.

➡️ No additional locking needed.

---

### **A3.3 — GPU / PyTorch nightly**

Yes — **must be documented explicitly**.

Do **not** try to “lock” nightly in a traditional sense.

Required:

* Separate documentation section explaining:

  * Why nightly is required
  * Which CUDA (`cu128`)
  * Known fragility
* Evidence snapshot:

  ```
  submission_runs/m42_v1/pip_freeze_training_pt_gpu.txt
  ```

➡️ This is **declared instability**, not hidden instability.

---

## 4. README Polish

### **A4.1 — What to update**

Update **only** these sections (no scope creep):

1. **Top summary** (why this exists, in judge language)
2. **Quickstart (Local)**
3. **Demo Flow** → points to `docs/DEMO.md`
4. **Evidence & Reproducibility**
5. **Training Paths (JAX vs PyTorch)**

➡️ No deep architecture essay.
➡️ Clarity > completeness.

---

### **A4.2 — Video URL**

**Yes, but as a placeholder.**

Add:

```
🎥 Demo Video: (link will be added upon submission)
```

Do **not** block README merge on the URL existing.

---

## 5. Evidence Audit

### **A5.1 — Required folders**

Required for final submission ZIP:

* ✅ `submission_runs/m40_v1/`
* ✅ `submission_runs/m41_v1/`
* ✅ `submission_runs/m42_v1/`

Nothing earlier is required.

---

### **A5.2 — Master evidence index**

**Yes — required.**

Create:

```
submission_runs/m42_v1/evidence_index.md
```

Contents:

* What each folder proves
* What was run
* On which machine
* Any known benign warnings

➡️ Judges should never have to guess *why* a folder exists.

---

## 6. Scope Boundaries

### **A6.1 — Code changes**

Allowed **only if** they are:

* Packaging
* Documentation
* Deterministic build scripts
* Evidence capture helpers

🚫 Not allowed:

* Feature changes
* Training logic changes
* Refactors
* Behavior changes

---

### **A6.2 — Final test pass**

**Yes — mandatory.**

Cursor should:

* Run the **standard CI-equivalent local test commands**
* Capture outputs into:

  ```
  submission_runs/m42_v1/test_run_outputs/
  ```

Even if CI already passed.

➡️ This is **belt-and-suspenders credibility**.

---

## Cursor Serialization / Recovery Strategies

### Decision: **Yes, implement the following three**

#### ✅ 1. PowerShell / Terminal Log Discipline

Approve adding guidance to `.cursorrules`.

#### ✅ 2. CHECKPOINT markers

Yes — add `CHECKPOINT:` entries to `M42_toolcalls.md`.

#### ❌ 3. Session State JSON

**Not needed.**
Adds overhead with limited benefit given toolcalls + checkpoints.

---

## Final Instruction to Cursor

You may now:

* Proceed to generate **`M42_plan.md`**
* Execute M42 **without further clarification**
* Treat all answers above as **locked decisions**

When M42 is complete, we will **freeze**, submit, and only *then* consider M43 (full training).

If you want, next I can:

* Pre-review the **expected M42_plan.md structure**
* Or write a **“Cursor execution contract”** to reduce drift even further
