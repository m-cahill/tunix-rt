Below are **authoritative, locked answers** for **M14**, aligned with how you’ve successfully run M11–M13: forward-compatible, CI-safe, enterprise-grade, and deliberately boring where it matters.

These answers are **binding**—Cursor should implement exactly this, without interpretation.

---

# ✅ M14 Decisions (Final)

## Database Schema

### **1. Stdout / Stderr Truncation**

* **Decision:** Keep **10KB truncation**, exactly as in M13.
* **Implementation:** Truncate at capture time; store as-is in `TEXT`.
* **Rationale:** Predictable size, avoids DB bloat, consistent audit trail.

---

### **2. Status State Machine**

* **Decision:** **Include `pending` in schema**, but do **not** write it in M14.
* **Runtime behavior in M14:**

  * Create run with `status="running"`
  * Transition to `completed | failed | timeout`
* **Rationale:** Forward compatibility for M15 async without migration churn.

---

### **3. Indexing**

* **Decision:**

  * Index `dataset_key`
  * Index `created_at`
* **Rationale:** Required for pagination + recent-run queries.

---

### **4. Mode Field Naming**

* **Decision:** Use **`dry-run`** (with hyphen).
* **Rationale:** Must match M13 API + frontend schema exactly.

---

### **5. Nullable Fields**

* **exit_code:**

  * `NULL` for dry-run and timeout
* **completed_at:**

  * `NULL` only if process never completed (crash)
* **duration_seconds:**

  * Always set if `completed_at` exists
* **Action:** Document these cases in migration docstring.

---

## Service Layer

### **6. Run Record Creation Timing**

* **Decision:**

  * Create run **immediately** with `status="running"`
  * Update at end
* **Rationale:** Matches future async semantics; no refactor later.

---

### **7. Dry-Run Persistence**

* **Decision:** **Persist dry-runs**.
* **Rationale:** Full audit trail; dry-run is still a meaningful action.

---

### **8. DB Failure Handling**

* **Decision:**

  * **Never fail the user request** due to DB write failure after execution.
  * Log error; return execution result.
* **Rationale:** Execution success > persistence guarantees.

---

### **9. Output Truncation**

* **Decision:** No additional truncation in M14.
* **Rationale:** Single truncation source of truth (M13).

---

## API Design

### **10. Pagination Defaults**

* **Decision:** Reuse `/api/traces` pattern.

  * `limit=20`
  * `max=100`
* **Rationale:** Consistency + predictable frontend UX.

---

### **11. Filtering Semantics**

* **Decision:**

  * Query params (`status`, `dataset_key`, `mode`)
  * **AND logic**
  * All filters optional
* **Rationale:** Simple, predictable, index-friendly.

---

### **12. Detail Response Schema**

* **Decision:** Reuse **`TunixRunResponse`**.
* **Rationale:** Avoid schema explosion; frontend already understands it.

---

### **13. Run ID Type**

* **Decision:** Use **Postgres UUID type**.
* **Rationale:** Native support, indexing efficiency, correctness.

---

## Frontend

### **14. Placement**

* **Decision:**

  * New **collapsible section** under existing Tunix panel.
* **Rationale:** No navigation changes; low UX risk.

---

### **15. Refresh Strategy**

* **Decision:** **Manual refresh button only**.
* **Rationale:** Polling belongs to M15 (async phase).

---

### **16. Scope of Runs Displayed**

* **Decision:**

  * Show **all runs**
  * Include `dataset_key` column
  * Allow frontend filtering
* **Rationale:** Enables cross-dataset auditing.

---

### **17. Detail View UX**

* **Decision:** Inline expandable rows.
* **Rationale:** Matches M13 UX; avoids routing complexity.

---

## Migration

### **18. Migration Naming**

* **Decision:**

  * `alembic revision -m "add_tunix_runs_table"`
* **Rationale:** Follow existing convention.

---

### **19. Downgrade**

* **Decision:**

  * Drop `tunix_runs` table entirely.
* **Rationale:** No FK dependencies; clean rollback.

---

### **20. Foreign Keys**

* **Decision:** **No FK** on `dataset_key`.
* **Rationale:** Datasets are not DB entities (yet).

---

## Testing

### **21. Test Strategy**

* **Decision:**

  * Use real service
  * Use **dry-run mode**
  * No Tunix dependency
* **Rationale:** Deterministic, CI-safe.

---

### **22. Coverage**

* **Decision:**

  * **No exclusions**
  * Maintain existing gates
* **Rationale:** This is core infrastructure now.

---

## Documentation

### **23. Doc Structure**

* **Decision:**

  * Single `M14_RUN_REGISTRY.md`
  * Covers backend + API + frontend
* **Rationale:** One feature, one doc.

---

### **24. README Update**

* **Decision:**

  * Extend existing **Tunix Integration** section.
* **Rationale:** Avoid fragmenting narrative.

---

## Scope Boundaries (Explicit)

### **25. Run Deletion**

* **Decision:** ❌ No DELETE endpoint.
* **Rationale:** Immutability; YAGNI.

---

### **26. Run Retry**

* **Decision:** ❌ No retry endpoint.
* **Rationale:** POST again is sufficient.

---

### **27. Metadata Mutation**

* **Decision:** ❌ No PATCH.
* **Rationale:** Runs are immutable audit records.

---

## Implementation Order

### **28. Development Sequence**

**LOCKED order:**

1. Schema + model
2. Alembic migration
3. Service layer
4. API endpoints
5. Backend tests
6. Frontend UI
7. E2E (if needed)

---

### **29. Stop Criteria (M14 is DONE when):**

* ✅ Runs persist across restarts
* ✅ List + detail endpoints work
* ✅ Frontend shows history
* ✅ Dry-run + local runs both recorded
* ✅ CI green, no coverage regression

---

## 🔑 Key Decisions Summary

| Topic            | Decision                        |
| ---------------- | ------------------------------- |
| Status           | Include `pending`, not used yet |
| Dry-run storage  | Yes                             |
| Run creation     | Immediate with `running`        |
| DB failure       | Do not fail user                |
| Frontend refresh | Manual only                     |
| Deletion / retry | Not in M14                      |

---

## Cursor Is Cleared to Implement M14

These answers close **all M14 ambiguity**.
Cursor should proceed without further clarification.

📎 **Reference:** 

If you want next:

* I can draft **M15 (Async Execution)** now
* Or map **Phase 3 → Phase 4 exit criteria**
* Or prepare **investor / Google-facing architecture summary**

Just say the word.
