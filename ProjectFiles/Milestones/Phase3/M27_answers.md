Great set of questions — these are exactly the right things to pin down before executing M27. Below are **clear, authoritative answers** you can proceed with immediately.

---

## ✅ M27 Clarifying Questions — Final Answers

### 1. **Dataset Generation (`golden-v2`)**

**Yes — create `golden-v2`.**
Do **not** mutate `golden-v1` in place.

**Correct approach**

* Keep `golden-v1` frozen (it is now a historical baseline).
* Extend dataset tooling to support `golden-v2` (100 traces).

**Implementation guidance**

* Either:

  * Update `backend/tools/seed_golden_dataset.py` to accept a `--version` or `--dataset` flag
    **or**
  * Create a new thin wrapper (`seed_golden_v2.py`) that calls the generic seeder introduced in M26.

**Key requirement**

* `golden-v2` must be **deterministic** (fixed seed).
* Manifest + DB seeding must stay in sync (no more “manifest exists but DB empty” situations).

✔️ **Decision**: Generate `golden-v2` explicitly; do not overwrite `golden-v1`.

---

### 2. **Training Script CLI (`--data` vs `--dataset`)**

**Add `--dataset` support, but keep `--data` for backward compatibility.**

This is the cleanest UX and avoids breaking existing scripts.

**Recommended behavior**

* `--dataset golden-v2`

  * resolves to: `backend/datasets/golden-v2/dataset.jsonl`
* `--data /path/to/file.jsonl`

  * still works as-is
* Enforce **mutual exclusivity**:

  ```text
  Exactly one of --dataset or --data must be provided
  ```

**Why**

* Matches how the backend, UI, and audits already think about datasets (by key, not file path).
* Keeps “offline” training ergonomic.

✔️ **Decision**: Support both; prefer `--dataset` going forward.

---

### 3. **Evaluation Implementation (`eval_generate.py`)**

**Yes — implement real model loading and inference.**
Mocks are no longer sufficient at this stage.

**Scope (keep it minimal)**

* Load trained checkpoint (Orbax)
* Run deterministic generation / scoring on a small eval split
* Produce:

  * scalar metrics (loss, score, judge proxy, etc.)
  * minimal per-example outputs if helpful

**Guardrails**

* No sampling randomness unless explicitly seeded
* No beam search complexity unless already needed
* Keep evaluation fast and repeatable

✔️ **Decision**: Replace mock logic with real inference against trained checkpoints.

---

### 4. **Evaluation Trigger (Where to Run Eval)**

You’re exactly right to call out the tradeoff here.

**Authoritative decision**

> **Do BOTH, with different responsibilities.**

#### Primary (Authoritative Path)

✅ **Backend worker triggers evaluation after training completes**

* This is the **canonical system behavior**
* Enables:

  * correct run lifecycle (`training → evaluating → completed`)
  * persistence of eval results in DB + artifacts
  * future automation (tuning, comparisons)

#### Secondary (Developer Convenience)

➕ Allow `train_jax.py --eval_after_train` for **offline/manual runs**

* This simply calls the same evaluation function
* Writes artifacts locally
* Does **not** replace the worker-driven path

**What NOT to do**

* Shell-script-only orchestration (too brittle)
* Eval logic duplicated in multiple places

✔️ **Decision**:

* Worker-triggered eval is primary
* Training script can optionally invoke eval for offline runs

---

### 5. **Frontend Location (“Run Detail page”)**

Correct observation: there is no formal `pages/` directory.

**Current reality**

* `src/components/Tuning.tsx` is the **main runs-oriented view**
* It already understands:

  * runs
  * metrics
  * job status

**Recommended approach**

* **Do not invent a new routing system in M27**
* Extend existing components incrementally:

**Concrete plan**

* Add a dedicated subcomponent, e.g.:

  * `RunDetailsPanel.tsx`
  * or `RunEvaluationSummary.tsx`
* Mount it inside:

  * `Tuning.tsx` (or whichever view already shows runs)

This keeps M27 focused on learning validation, not frontend architecture.

✔️ **Decision**: Extend existing run UI; no new top-level “page” yet.

---

## ✅ Summary Table

| Question            | Decision                                          |
| ------------------- | ------------------------------------------------- |
| `golden-v2` dataset | Create explicitly; keep `golden-v1` frozen        |
| Training CLI        | Add `--dataset`, keep `--data`                    |
| Evaluation logic    | Implement real inference                          |
| Eval trigger        | Worker-driven (primary), training script optional |
| Frontend location   | Extend existing run view (`Tuning.tsx`)           |

---

You are now fully unblocked to execute **M27** with confidence.

Once M27 lands, you will have:

* proven learning
* automatic evaluation
* closed-loop training → eval → UI
* a credible foundation for **M28 (tuning + comparison)**

Proceed 🚀
