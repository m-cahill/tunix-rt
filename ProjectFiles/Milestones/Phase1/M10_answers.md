Here are **clear approvals / decisions for M10 (Q1–Q10)** so Cursor can proceed immediately, with no further back-and-forth.
(Decisions are grounded in your **M10 clarifying questions** document. )

---

## ✅ Final Decisions for M10

### **Q1. Implementation scope & phasing**

**Approved: Option A — complete all phases sequentially.**

* M10 is refactor-heavy but conceptually settled.
* No architectural unknowns remain.
* Pausing mid-refactor would add risk without benefit.

**Directive:** Proceed Phase 0 → Phase 5 in one pass.

---

### **Q2. Service layer pattern**

**Approved: Option A — create `backend/tunix_rt_backend/services/`.**

This is the *right* inflection point to formalize the distinction:

* `helpers/` → small, stateless utilities
* `services/` → business logic / orchestration

This aligns with enterprise layering and will pay dividends in M11–M12.

---

### **Q3. Batch endpoint optimization**

**Approved: Option B — single bulk SELECT after commit.**

* Preserves DB-generated fields (`id`, `created_at`)
* Avoids N refresh calls
* Keeps AsyncSession usage safe and idiomatic

**Guardrail:** No concurrent operations on the same `AsyncSession`.

---

### **Q4. Training script tests (Phase 5)**

**Approved: Option B — defer to M11.**

Correct call.

* M10 is about **runtime app health**
* Training scripts are already isolated and documented
* We don’t want to expand scope unnecessarily

**Directive:** Explicitly mark this as deferred in M10 summary.

---

### **Q5. Coverage expectations during refactor**

**Approved: Yes — temporary dips are acceptable.**

**Hard requirement:**

* Final state must be **≥ baseline (~79% line)**
* CI must be green at the end

Intermediate commits may dip **as long as the branch is not merged early**.

---

### **Q6. Git strategy**

**Approved: Option B — work on a branch (`m10-refactor`).**

* This is a non-trivial refactor.
* Reviewability > speed.
* Merge only when all phases + docs are complete.

---

### **Q7. Commit granularity**

**Approved: Option A — follow the 7-commit sequence.**

This gives you:

* Clean rollback points
* Clear audit trail
* Easier future archaeology

---

### **Q8. Phase 5 scope clarification**

**Approved: Option C — defer entirely to M11.**

No partial implementation.
No stubs.
No half-tests.

Clean deferral is the correct engineering decision.

---

### **Q9. Documentation updates**

**Approved: Yes — create both.**

Required artifacts:

* `docs/M10_BASELINE.md`
* `docs/M10_SUMMARY.md`
* `docs/M10_GUARDRAILS.md`

Consistency across milestones is now a **project invariant**.

---

### **Q10. AsyncSession concurrency guardrail**

**Approved: Option B — document + inline code comment.**

Do **both**:

* Inline comment in batch service explaining why concurrency is forbidden
* Written guardrail in `M10_GUARDRAILS.md`

This prevents future “optimization” bugs.

---

## 📌 Consolidated Directive to Cursor

You are **cleared to proceed with M10 exactly as proposed**, with the following constraints locked:

* Complete **all phases** in one branch
* Introduce a formal `services/` layer
* Optimize batch refresh via **single bulk SELECT**
* **Defer training script tests** to M11 (explicitly documented)
* Maintain or exceed **baseline coverage** by end of milestone
* Use **7 commits**, branch-based workflow
* Produce **baseline + summary + guardrails docs**
* Add **AsyncSession concurrency warning** inline + documented

No further approval needed.

---

## 🔭 Strategic Note (for you, not Cursor)

M10 is the **“make it boring” milestone**:

* thinner controllers
* fewer footguns
* clearer layering
* organic coverage recovery

Once M10 lands, the codebase is structurally ready for:

* M11 (evaluation loop expansion)
* M12 (dataset → training → eval automation)
* M13+ (scaling narratives, investor-grade demos)

This is exactly the right sequencing.

If you want, next I can:

* pre-design **M11** at a high level, or
* draft a **post-M10 architecture snapshot** you can reuse externally.
