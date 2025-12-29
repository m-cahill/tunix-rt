According to a document from **December 27, 2025**, here are my answers to the **M36 clarifying questions**. 

---

## 1) Kaggle execution scope

**Answer: (c) Both.** 

* Keep the **local CPU/smoke rehearsal** path (still valuable for quick validation and CI-friendly evidence generation).
* Also produce a **Kaggle operator runbook + evidence templates** so *you* can execute the real GPU/TPU run and paste in final values.

This matches the M33 “proof-first” pattern and avoids blocking on Kaggle execution capability. 

---

## 2) Notebook eval set default

**Answer: (a) Switch default to `eval_v2.jsonl`, keep `eval_v1.jsonl` as documented fallback.** 

Rationale:

* M36 explicitly requires `eval_v2` (100 items + scorecard structure), and M35 already positioned this as the competition-grade eval set. 

---

## 3) Evidence schema additions (`kaggle_notebook_url`, `kaggle_run_id`)

**Answer: Add them as first-class fields in `run_manifest.json`, but treat `kaggle_run_id` as optional.** 

Recommended shape:

* `kaggle_notebook_url` **required for Kaggle runs** (can be `null` for local runs).
* Add one more field you didn’t list but is *more reliable than “run_id”*:

  * `kaggle_notebook_version` (the saved version number / identifier)
* `kaggle_run_id` **optional** (only if you can confidently obtain it from Kaggle UI/API).

Why:

* Kaggle notebooks are “versioned” via **Save Version / Version History**; capturing a specific version is the most reproducible handle for evidence.
* If you need logs/artifacts programmatically, the Kaggle API supports downloading kernel outputs (which can complement your evidence bundle).

So: don’t bury these in `notes`—make them explicit top-level fields (with clear nullability rules).

---

## 4) Frontend test coverage targets: what to prioritize

**Answer: Your prioritization is good; I’d re-order slightly to maximize audit uplift + regression protection.** 

### `Leaderboard.tsx` (5–10 tests)

Prioritize:

1. Loading / empty / error states (fast, stable, high value)
2. Renders rows + columns correctly
3. Filter inputs update state + Apply/Clear triggers fetch params
4. Scorecard rendering and **primary score % formatting** (important for M35 fidelity) 
5. Pagination (next/prev)
6. Date formatting (lower value, do if cheap)

### `LiveLogs.tsx` (3–5 tests)

Prioritize:

1. Waiting state
2. Renders incoming log lines
3. Connection status indicator + status transitions
4. Auto-scroll behavior
5. (If implemented) reconnect/backoff behavior (nice-to-have)

---

## 5) React `act()` warnings scope

**Answer: (b) Fix all warnings, but timebox it to “warnings-only” changes.** 

Reason: A clean test output is a cheap “audit score” win and prevents future signal loss in CI logs. Do it surgically:

* Prefer `await waitFor(...)`, proper async user-event handling, and wrapping state transitions.
* Avoid broad refactors; if any warning looks non-trivial, defer with a TODO.

---

## 6) Docs update scope (`evaluation.md` vs `M36_KAGGLE_RUN.md`)

**Answer: Do both, with `docs/M36_KAGGLE_RUN.md` as the primary runbook and `docs/evaluation.md` as the durable reference.** 

* `docs/M36_KAGGLE_RUN.md`: step-by-step Kaggle execution + exactly what evidence fields to fill (URL/version/log excerpt), and what artifacts to upload into `submission_runs/m36_v1/`.
* `docs/evaluation.md`: short section describing current “per-item predictions” limitation + what M37 would add (artifact persistence), with a cross-link to the runbook.

This keeps execution operational docs separate from conceptual system docs.

---
