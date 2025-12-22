Here are **final, authoritative answers to the M11 clarifying questions**, locking scope and risk so Cursor can proceed without hesitation. These decisions are based on the M11 questions you attached and the current audit state. 

---

## ✅ M11 Decisions (Locked)

### **1. Risk Appetite for Optional Phases (5–6)**

**Decision:** **B — Complete mandatory phases (0–4), then attempt Phase 5, skip Phase 6.**

* Phase 5 (frontend coverage to 70%) is low risk and worth doing if time permits.
* Phase 6 (Vite 7 / Vitest 4 upgrade) is **explicitly deferred** to a future, dedicated milestone.
* No dependency upgrades in M11 beyond what’s strictly required for security/DX fixes.

**Lock:** Phase 6 is *out of scope* for M11.

---

### **2. Training Script Test Strategy**

**Decision:** **A — Subprocess-based smoke tests only.**

* Implement `--dry-run` and validate via `subprocess.run([...])`.
* No refactoring of training scripts into importable units in M11.
* Unit-level testing of training internals is deferred to a later milestone when training stabilizes.

**Rationale:** Fast, deterministic, CI-safe. Matches M10 deferral plan.

---

### **3. Priority Order (If Time Constrained)**

**Decision:** **Approved as proposed**, with one small reorder:

**Final priority order:**

1. **Phase 0** — Baseline (mandatory)
2. **Phase 1** — Fix-first (SBOM, SHA pinning, pre-commit)
3. **Phase 3** — Complete app extraction (core architectural goal)
4. **Phase 4** — Training script dry-run tests (explicit M10 deferral)
5. **Phase 2** — Docs (ADR + production docs)
6. **Phase 5** — Frontend coverage (nice-to-have)
7. **Phase 6** — ❌ Deferred

Docs can slide slightly if needed, but must be complete by milestone end.

---

### **4. Branch Strategy**

**Decision:** **A — Create `m11-stabilize` branch and merge at the end.**

* M11 touches CI, security, architecture, and docs.
* A branch provides safety and reviewability.
* Merge only when **all mandatory phases are complete and CI is green**.

This differs slightly from M10, intentionally.

---

### **5. Documentation Update Timing**

**Decision:** **C — Update twice.**

* **After Phase 3:**
  Update `tunix-rt.md` to reflect architectural changes (services, thinner app).
* **At end of M11:**
  Final summary + links to ADR-006, training docs, SLOs.

This balances cleanliness with accuracy.

---

### **6. SBOM Tool Preference**

**Decision:** **A — Use `cyclonedx-py` CLI.**

* Fastest, lowest-risk fix.
* Avoids the earlier Python module invocation issues.
* Upload JSON SBOM artifact in CI.

No experimentation with alternative tools in M11.

---

### **7. GitHub Actions SHA Pinning**

**Decision:** **A — Manually pin to SHAs + add Dependabot.**

* Pin all `uses:` entries to full SHAs.
* Add/update Dependabot config to keep them fresh.
* Do not rely on one-shot pinning tools.

This is the strongest supply-chain posture.

---

### **8. Training Script `--dry-run` Semantics**

**Decision:** **A — Lightweight validation only.**

`--dry-run` must:

* Load YAML config
* Validate required fields
* Compute output paths
* Validate manifest schema (if applicable)
* Exit **0**
* ❌ No model init, dataset loading, or Tunix calls

Tests must remain fast and deterministic.

---

### **9. Coverage Gate Adjustment**

**Decision:** **A — Keep existing gates unchanged.**

* **Line:** ≥80%
* **Branch:** ≥68%

Do **not** raise gates to current levels.
Temporary dips during refactor are acceptable as long as:

* Final state meets gates
* No silent regressions

---

### **10. Commit Granularity**

**Decision:** **B — Use atomic commits (≈15–20 commits).**

* One logical change per commit.
* Easier rollback, clearer history, better auditability.
* Still follow the *sequence* of the M11 plan, just finer-grained.

---

## 📌 Consolidated Directive for Cursor

Proceed with **M11** using the following locked constraints:

* Mandatory phases: **0–4**
* Optional: **Phase 5 if time allows**
* **Phase 6 explicitly deferred**
* Work on **`m11-stabilize` branch**
* Subprocess-only training script smoke tests
* SBOM via `cyclonedx-py`
* SHA-pin all GitHub Actions + Dependabot
* Keep existing coverage gates
* Atomic commits
* Docs updated **after Phase 3 and at end**

No further clarification needed.

---

## 🔭 Strategic Note (for you)

M11 is the **last “stabilization + hygiene” milestone** before the project shifts back into forward momentum (evaluation expansion, richer training loops, external-facing demos).

After M11:

* Architecture is locked
* CI/security posture is investor-grade
* Training + eval loops are safe to deepen

When you’re ready, the next natural step is **M12: Evaluation Loop Expansion + Trace → Dataset → Score Feedback**, and you’ll be doing that on very solid ground.

If you want, next I can:

* generate a **Cursor-ready TODO checklist** for M11 using these decisions, or
* sketch **M12 options** so you can choose scope early.
