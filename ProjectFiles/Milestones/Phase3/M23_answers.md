Here are the decisions for each M23 question, with the “why” and the most robust implementation choice. 

---

## 1) `predictions.jsonl` contract & where to generate it

**Decision:** implement a **separate inference/generation step in the execution pipeline** (e.g. `tunix_execution.py`), **not** dummy output from `training/train_sft_tunix.py`. 

**Why:**

* In most ML systems, **training artifacts** (weights/checkpoints) and **serving/eval artifacts** (predictions/scores) are separate pipeline stages, because eval must be reproducible and tied to a specific model version + dataset + evaluator. ([martinfowler.com][1])
* If you generate dummy `predictions.jsonl` from a placeholder trainer, you’ll “pass” eval while measuring nothing—bad training signal and easy to regress later.

**Practical shape (minimal, but real):**

* After training finishes (or even for “dry-run”), run a lightweight “predict” pass over the dataset prompts and write:

  * `predictions.jsonl`: `{trace_id, prediction}` one per item
  * optionally `predictions_meta.json`: `{model_id, dataset_key, judge_version, created_at}`
* For CI: keep dataset small (`golden-v1`) so this step stays fast.

---

## 2) Is `AnswerCorrectnessJudge` “stub” logic or just missing inputs?

**Decision:** treat it as **primarily missing/fragile inputs + contract**, not fundamentally wrong comparison logic. 

**What’s “stubby” in practice:**

* The judge can’t be “real” until `predictions.jsonl` is reliably produced and discoverable.
* Error handling needs to be *crisp and actionable* (missing manifest, missing predictions file, empty predictions, trace IDs not found).

**Comparison logic:** your current `_compare` exact-match-with-normalization approach is a perfectly valid **`answer_correctness@v1`** baseline. Keep it deterministic and version it; you can expand later (e.g., regex, numeric parsing, multiple acceptable answers) without breaking historical runs by bumping to `@v2`.

---

## 3) Frontend `act()` warnings: targeted vs sweep

**Decision:** run the **full frontend test suite** and fix whatever appears, but prioritize the tests that touch async state updates (likely `Tuning.test.tsx` / registry-related tests). 

**Why:** `act()` warnings are usually a sign the test is asserting before the UI settles; the correct fix is typically `await userEvent...` + `findBy*`/`waitFor`, not wrapping everything in `act`. ([Kent C. Dodds][2])

**Standard fixes that usually eliminate warnings:**

* `const user = userEvent.setup(); await user.click(...)`
* Use `await screen.findByText(...)` instead of `getByText` after async actions
* Wrap assertions in `await waitFor(() => ...)` when the UI updates later

---

## 4) Locked metrics: schema validation vs runtime enforcement

**Decision:** enforce locked metrics **at runtime in `TuningService`** as the source of truth, and optionally also validate in the schema for faster feedback. 

**Why:**

* Service-layer enforcement ensures the rule holds even if the job is created via a different caller (future CLI, internal calls, tests). That’s consistent with “validate at boundaries, enforce invariants in the domain/service.” ([Stack Overflow][3])
* Schema validation is nice UX, but it’s not a security/consistency boundary by itself.

**Recommended pattern:**

* In `TuningService.start_job()` (or equivalent), check:

  * requested metric ∈ `LOCKED_METRICS`
  * otherwise raise `ValueError` / 422 with a clear message: “Only locked metrics allowed: answer_correctness@v1”

---

If you want a single “Cursor patch plan” for M23 implementing all four decisions, say the word and I’ll format it as a step-by-step checklist with file paths and minimal diffs.

[1]: https://martinfowler.com/articles/cd4ml.html?utm_source=chatgpt.com "Continuous Delivery for Machine Learning"
[2]: https://kentcdodds.com/blog/common-mistakes-with-react-testing-library?utm_source=chatgpt.com "Common mistakes with React Testing Library"
[3]: https://stackoverflow.com/questions/76836799/should-the-data-be-validated-independently-in-each-microservice?utm_source=chatgpt.com "Should the data be validated independently in each ..."
