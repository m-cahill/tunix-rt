Great questions — these are exactly the right clarifications to lock **M26** cleanly before execution. Here are **authoritative answers** you can proceed with confidently.

---

## ✅ M26 Clarifications — Final Answers

### 1. **Dependency Management (Orbax)**

**Yes.** Add `orbax-checkpoint` to the **`training` optional dependencies** in `backend/pyproject.toml`.

**Rationale**

* Orbax is the **canonical** checkpointing solution for Flax/JAX.
* It is only needed for **training**, so it belongs in `training` extras.
* CI remains unaffected because training extras are **opt-in**.

**Decision**

```toml
[project.optional-dependencies]
training = [
  "jax",
  "flax",
  "optax",
  "orbax-checkpoint",
]
```

✔️ Approved.

---

### 2. **Dataset Seeding Strategy**

**Create a new generic seeder** and refactor the existing one to use it.

**Correct approach**

* Create: `tools/seed_dataset.py` (generic, future-proof)
* Refactor: `seed_golden_dataset.py` → thin wrapper calling the generic seeder with defaults

**Why**

* You will need this again (`golden-v3`, eval sets, synthetic variants).
* Keeps backward compatibility with existing scripts.
* Avoids one-off tooling debt.

**Decision**
✔️ **Generic seeder + compatibility wrapper**.

---

### 3. **Benchmarking Script Location**

**Yes — standalone `training/bench_jax.py` is preferred.**

**Why**

* Keeps `train_jax.py` focused on correctness + training logic.
* Benchmarking has different concerns (warmup, timers, profiling).
* Makes it easier to compare configs without touching training code.

**Decision**
✔️ Implement `training/bench_jax.py`.

(Optionally: reuse shared helpers from `train_jax.py`, but do not merge responsibilities.)

---

### 4. **Metrics Granularity (DB vs Artifacts)**

**Store only summary metrics in the DB. Use artifacts for time-series.**

**Correct split**

* **DB (`tunix_runs.metrics`)**

  * final loss
  * best loss
  * total steps
  * total tokens
  * avg throughput
  * wall time
* **Artifacts (`metrics.jsonl`)**

  * per-step / per-interval metrics
  * full loss curve
  * throughput over time

**Why**

* DB JSON columns are not meant for unbounded time-series.
* You already have the artifact pipeline — leverage it.
* UI can read artifacts directly for charts.

**Decision**
✔️ **Summary in DB, detailed metrics in artifacts**.

This aligns perfectly with your existing architecture.

---

### 5. **Which `train_jax.py` File**

**Correct.**
You should work with:

```
/training/train_jax.py   (root-level training folder)
```

**Not** anything inside:

```
backend/tunix_rt_backend/
```

**Reason**

* Training is intentionally **out-of-band** from the API/backend runtime.
* Backend orchestrates runs; training scripts execute them.
* This separation is good and should remain.

**Decision**
✔️ Confirmed.

---

## ✅ Final Go-Ahead

You are fully unblocked.

You can now:

* Finalize the M26 plan
* Start implementation
* Hand this directly to Cursor

No further clarification needed.
This is a **clean, well-scoped milestone** and a strong step toward real training runs.

Proceed 🚀
