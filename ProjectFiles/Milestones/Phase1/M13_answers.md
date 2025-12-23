Below are **clear, binding answers** to the **M13 clarifying questions**, aligned with how you’ve been running this project so far: incremental, enterprise-grade, CI-safe, and deliberately stopping *before* TPU / long-running orchestration.

I’m answering **decisively** so Cursor can execute without further back-and-forth.

---

## ✅ Architecture & Design Decisions (LOCKED FOR M13)

### **1. Tunix Execution Modes**

**Supported modes in M13: exactly two.**

#### ✅ Dry-run (DEFAULT)

* **Behavior:**

  * Validate manifest + dataset
  * Validate Tunix availability *only structurally*
  * **Do NOT** invoke Tunix CLI
* **Purpose:** CI-safe, always available, deterministic
* **Required:** Yes

#### ✅ Local

* **Behavior:**

  * Executes `tunix train --config manifest.yaml` via subprocess
  * Captures stdout / stderr / exit code
* **Purpose:** Developer verification only
* **Required:** Yes (single verified path)

❌ **No other modes in M13**

* No remote
* No TPU
* No Ray
* No background workers

Those start **M14+**, not now.

---

### **2. Tunix Runtime Dependency**

Follow **UNGAR pattern exactly**.

✅ Decisions:

* Add **`backend[tunix]`** optional extra
* If Tunix not installed and `dry_run=false`:

  * Return **501 Not Implemented**
* Implement:

  ```python
  tunix_available() -> bool
  ```

  * Checks importability and CLI availability

This keeps default CI **clean and fast**.

---

### **3. Execution Metadata Storage**

❌ **NO persistence in M13**

Explicitly **do not**:

* Add DB tables
* Store run history
* Track runs beyond request lifecycle

✅ Response payload only:

* run_id (UUID)
* status
* timestamps
* stdout / stderr (truncated)
* exit_code

Persistence begins **M14**.

---

### **4. Endpoint Design**

✅ **Option B — Parameters-first (LOCKED)**

```json
POST /api/tunix/run
{
  "dataset_key": "my_dataset-v1",
  "model_id": "google/gemma-2b-it",
  "dry_run": true
}
```

**Why:**

* Matches Tunix mental model
* Avoids raw YAML in API
* Keeps backend as source of truth
* Manifest generation already exists (M12)

Cursor should **not** support manifest upload in M13.

---

### **5. Local Execution Details**

#### Execution

* Use `subprocess.run()`
* Capture stdout/stderr
* Return exit code

#### Timeout

* **Dry-run:** 10s
* **Local execution:** 30s hard timeout

If it times out:

```json
status = "failed"
reason = "timeout"
```

#### Execution Model

* **Synchronous / blocking**
* No async jobs
* No polling endpoints

This is intentional. Async comes later.

---

### **6. Frontend Integration**

✅ **Minimal UI only**

* Add **“Run with Tunix (Local)”** button
* Location: **existing Tunix panel**
* Display:

  * Spinner
  * Final status
  * Collapsible stdout/stderr

❌ No:

* Streaming logs
* Training history
* Progress bars

---

### **7. CI Workflow**

✅ **Separate workflow**

* New: `tunix-runtime.yml`
* **Manual dispatch only**
* **Never blocks merge**

#### What it tests:

* Dry-run path only
* Manifest + dataset validation
* No real training

❌ No scheduled runs
❌ No local execution in CI

---

### **8. Stop Criteria (CONFIRMED)**

M13 is **DONE** when:

✅ Dry-run works
✅ Local execution works once (developer verified)
✅ 501 when Tunix unavailable
✅ Default CI green without Tunix
✅ Optional CI dry-run passes

❌ No persistence
❌ No result ingestion
❌ No metrics
❌ No history UI

Yes — your interpretation is **100% correct**.

---

## 🧪 Testing Strategy

### **9. Test Coverage**

Match UNGAR exactly.

* Default tests:

  * Dry-run path
  * 501 behavior
* Optional tests:

  * `@pytest.mark.tunix`
  * Local execution smoke test

Target:

* **+10–15 tests**
* Coverage neutral or positive

---

### **10. Local Execution Tests**

For `@pytest.mark.tunix`:

* Dataset: **1–2 traces**
* Model: smallest available (documented)
* Epochs: 1
* Timeout: **5 seconds**
* Purpose: *prove plumbing works*, not training quality

---

## 🔧 Compatibility & Dependencies

### **11. Tunix Installation**

* Document installation in M13 docs
* Assume **public availability**
* Do **not** pin version yet
* Installation documented, not enforced

---

### **12. TPU Assumptions**

✅ Explicitly **NO TPU**

* CPU/GPU only
* No JAX TPU config
* No TPU flags
* TPU orchestration deferred

---

## 📚 Documentation Scope (REQUIRED)

Create all of the following:

* `docs/M13_BASELINE.md`
* `docs/M13_TUNIX_EXECUTION.md`
* `docs/M13_SUMMARY.md`
* Update:

  * `tunix-rt.md`
  * `README.md`

This mirrors M12 discipline.

---

## 🧭 Final Confirmation Summary

| Question        | Answer                    |
| --------------- | ------------------------- |
| Execution modes | Dry-run + Local only      |
| Persistence     | ❌ No                      |
| Endpoint        | Parameters-first          |
| Frontend        | Minimal button + logs     |
| CI              | Separate, manual, dry-run |
| Stop criteria   | Exactly as stated         |

---

## 🚦Cursor Is Clear to Proceed

These answers are **authoritative for M13**.
Cursor should **not reinterpret or extend scope** beyond this.

If you want, next I can:

* Generate the **exact M13 Cursor prompt**
* Or draft the **M13 ADRs up-front**
* Or outline **M14+ roadmap** now that Tunix execution exists

📎 **Reference:** 
