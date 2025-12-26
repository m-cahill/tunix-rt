Here are the M29 decisions, locked so you can execute without churn. (I also checked Kaggle notebook + JAX/Flax expectations to make sure our “submission path” choices align with how Kaggle actually runs notebooks.) ([Kaggle][1])

---

## 1) Branch creation

Create `milestone/M29-competition-data-and-routers` **from `main`**.

Rationale: M29 is largely refactor + packaging + data pipeline work; it benefits from being directly on the integration trunk with the latest CI constraints and avoids long-lived divergence.

✅ Decision: **branch from `main`**.

---

## 2) Router granularity

Your proposed grouping is solid and aligns with how the code is already conceptually organized.

✅ Decision: **Use your Option A (grouped routers), with your 10-router split**:

* `health.py` — `/api/health`, `/api/redi/health`, `/metrics`
* `traces.py` — traces CRUD + compare
* `datasets.py` — dataset build/export
* `ungar.py` — UNGAR generation/export
* `tunix.py` — tunix status, SFT export/manifest, run execution (non-run endpoints)
* `tunix_runs.py` — run list/details, logs, artifacts, metrics, cancellation
* `evaluation.py` — evaluate trigger/get, evaluations list, leaderboard score aggregation
* `regression.py` — baselines + checks
* `tuning.py` — tuning jobs CRUD
* `models.py` — model registry + versions

Guardrail: **Do not change route paths** in this refactor. This should be a pure extraction + router include.

---

## 3) TODO disposition (3 TODO markers)

These are small and should not be left as code TODOs this late.

✅ Decision: **Implement the two easy ones now; issue the “optional pagination” one.**

* `model_registry.py:57` “Add pagination if needed” → **Convert to GitHub issue** (low urgency; keep API stable).
* `model_registry.py:159` “Populate from Evaluations if available” → **Implement now** (this directly improves leaderboard/registry coherence).
* `regression.py:123` “Support lower is better configuration” → **Implement now** (small config surface, improves correctness).

---

## 4) Dataset builder endpoint strategy

You already have `POST /api/datasets/build`. Don’t replace it.

✅ Decision: **C (both), but staged:**

1. **Enhance existing `/api/datasets/build`** with provenance metadata (schema version, source, counts, build timestamp).
2. Add a **minimal new endpoint** for import, e.g.:

   * `POST /api/datasets/ingest` (imports JSONL → traces DB)
   * then build uses those trace ids (or a query) to create the dataset manifest

Guardrail: Keep the data contract explicit and versioned so we don’t regress into “manifest exists, DB empty”.

---

## 5) “Bigger-than-golden” dataset (`dev-reasoning-v1`)

✅ Decision: **Generate via a deterministic seed script (not UNGAR-only).**

What it should contain (v1):

* **70% reasoning-trace style items** that match the competition’s “show your work” flavor (short decompositions, tool-less reasoning, verification steps)
* **30% structured synthetic tasks** (procedural arithmetic/logic/string transforms) to harden generalization

Why not UNGAR-only:

* UNGAR is great for structured episodes, but competition scoring will hinge on general reasoning trace quality. UNGAR should be *part* of the mix, not the entirety.

Guardrails:

* No scraped/licensing-risk external text.
* Everything procedurally generated or self-authored.
* Deterministic seed + manifest.

---

## 6) Notebook location

✅ Decision: **A — repo root `notebooks/`**

Reason: easiest discovery + mirrors Kaggle mental model. Kaggle notebooks write outputs to `/kaggle/working` and read inputs from `/kaggle/input`, but our repo layout should be simple and obvious. ([Kaggle][2])

---

## 7) Notebook format

✅ Decision: **C — both**

* Authoritative artifact: `notebooks/kaggle_submission.ipynb`
* Companion: `notebooks/kaggle_submission.py` (kept in sync; useful for local runs / diffability)

Kaggle is natively notebook-driven, so `.ipynb` is the primary deliverable; having a `.py` companion reduces friction for CI/dev review. ([Kaggle][1])

---

## 8) Training backend for Kaggle notebook

✅ Decision: **A — use the JAX/Flax path (`train_jax.py`)**

Why:

* Kaggle explicitly supports JAX/Flax and even recommends GPUs for JAX while noting TPU compatibility. ([Kaggle][3])
* You already invested in the JAX pipeline + checkpointing + metrics.
* Keeps the “single-session” story clean.

Keep `train_torch.py` as a local alternative, but the Kaggle “hero path” should be JAX.

---

## 9) Nightly CI tier scope

✅ Decision: **B — basic nightly workflow**

* Runs all tests on schedule
* No extended dataset validation
* No smoke training (yet)

Rationale: low-risk, good audit/rigor signal, minimal maintenance. We can expand in M30 if needed.

---

## 10) Must-haves vs nice-to-haves (if time constrained)

✅ Priority order (final):

1. **Phase 1 — routers refactor** (audit + long-term maintainability)
2. **Phase 3 — dataset ingest/build + dev-reasoning-v1** (competition readiness)
3. **Phase 4 — Kaggle notebook path** (submission packaging)
4. **Phase 2 — TODOs** (do the two small implementations; issue the pagination)
5. **Phase 5 — nightly** (do basic nightly if time; otherwise slip to M30)

---

## 11) E2E/frontend updates if API paths change

✅ Decision: **Same PR, same commit series** if any path changes.

But: **avoid changing paths** at all in router extraction. This should be internal restructuring only. If a path must change (strongly discouraged), you must update:

* frontend client
* e2e tests
* docs
  in the same PR so CI remains a reliable signal.

---

# One extra guardrail for M29 (high ROI)

Because we previously had a “client.ts got overwritten” incident:

✅ Add a tiny unit test in `frontend/src/api/client.test.ts` that asserts **core exports exist** (e.g., `getApiHealth`, `fetchJSON`, etc.). This prevents “file clobbered” regressions from ever landing again.

---

If you want, I can convert the above into a **Cursor-ready checklist prompt** (copy/paste) matching your milestone template exactly—but the decisions above are the key unblockers.

[1]: https://www.kaggle.com/docs/notebooks?utm_source=chatgpt.com "Kaggle's notebook documentation"
[2]: https://www.kaggle.com/code/waldemar/super-easy-introduction-to-kaggle-directories?utm_source=chatgpt.com "Super EASY introduction to Kaggle Directories"
[3]: https://www.kaggle.com/getting-started/315696?utm_source=chatgpt.com "Getting Started with JAX/FLAX"
