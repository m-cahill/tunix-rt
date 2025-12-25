Next up should be a **tight “M23: Post-Ship Hardening”** milestone—because your M22 audit shows two things that will bite you the moment you start training:

* **`AnswerCorrectnessJudge.evaluate()` is still a stub/placeholder**, meaning you don’t actually have real correctness scoring yet. 
* **Backend coverage regressed to ~67%** (well below the ≥75% target you wanted as a training gate). 

So: **M23 should finish the evaluation engine + restore coverage gates**, then **M24 can be “first baseline training run”**.

---

## What to do next: M23 (Hardening Sprint)

### M23 goals

1. **Make `answer_correctness@v1` real** (no hardcoded scores). 
2. **Bring backend coverage back to ≥75%** (or at minimum reinstate a non-regression gate). 
3. **Align dataset guardrails + tests** (empty dataset behavior). 
4. **Make “locked metric” guardrail truly enforceable** (block, not warn). 
5. Remove React test **`act()` warnings** (fragility cleanup). 

After M23: you’re genuinely training-ready.

---

## Cursor handoff prompt — M23

**Title:** M23 — Evaluation Engine Completion + Coverage Gate + Test Hygiene

**Context:** M22 shipped evaluation scaffolding, golden dataset, UI metric visibility, and training readiness docs, but audit flags: judge stub + coverage regression + a couple guardrail/test mismatches.  

### Phase 0 — Baseline gate (mandatory)

* Pull `main`, run full suite locally (backend/frontend/e2e). Ensure green before changes.

### Phase 1 — Finish `AnswerCorrectnessJudge` for real scoring (HIGH)

1. Replace the placeholder logic in `AnswerCorrectnessJudge.evaluate()` with real evaluation:

   * Load dataset manifest from `get_datasets_dir() / run.dataset_key / manifest.json`
   * Resolve the dataset items (trace IDs) from DB
   * **Load model predictions** for each item from a run artifact (recommended contract):

     * `predictions.jsonl` containing `{ "trace_id": "...", "prediction": "..." }`
   * Normalize prediction and ground truth and compute mean correctness.
2. Define/implement the minimal **run-output contract** if missing:

   * Ensure the Tunix run/export pipeline writes `predictions.jsonl` for eval to consume.
3. Persist computed metrics + judge version info as you already scaffolded.

**Guardrails**

* If manifest missing → fail with clear message
* If predictions file missing/empty → fail with clear message (don’t silently pass)
* If trace IDs don’t resolve in DB → fail with clear message

### Phase 2 — Add tests for judge + factory (HIGH, coverage driver)

Create `backend/tests/test_judges.py`:

* Unit tests for `_normalize_text()`
* Integration-style test for `evaluate()` using:

  * seeded `golden-v1` traces in test DB
  * temp `predictions.jsonl` artifact
  * expect `answer_correctness` exact value
* Tests for `JudgeFactory.get_judge()` with override `"answer_correctness"`

**Exit:** `judges.py` coverage rises materially (target ≥70% for that module).

### Phase 3 — Restore coverage gate (HIGH)

* Add CI coverage gate: `pytest --cov --cov-fail-under=75` (or stage it: 70 now, 75 next).
* Ensure coverage is **non-regressing** from M23 onward.

### Phase 4 — Fix empty dataset guardrail mismatch (MED)

* Decide behavior: keep current behavior “empty dataset raises ValueError” (recommended).
* Update the failing/contradicting test to assert `pytest.raises(ValueError, match="Dataset is empty")`. 

### Phase 5 — Strengthen tuning guardrail (MED)

* Replace warning-only “metric not answer_correctness” with **blocking**:

  * `LOCKED_METRICS = {"answer_correctness"}`
  * if metric not locked → `raise ValueError(...)`
    This aligns the code with the training-readiness promise. 

### Phase 6 — Frontend test `act()` warnings (LOW)

* Eliminate warnings by using `waitFor`/`findBy*` and awaiting async user events rather than asserting immediately. (This is the standard RTL fix pattern.) ([Kent C. Dodds][1])

### Phase 7 — Docs quick win (LOW)

* Add README section: how to run `seed_golden_dataset.py` and verify `golden-v1` exists. 

---

## Definition of Done for M23

* [ ] `AnswerCorrectnessJudge.evaluate()` computes real scores (no hardcoded pass/100%). 
* [ ] Judge reads real predictions (via a clear artifact contract like `predictions.jsonl`)
* [ ] New judge tests added; `judges.py` coverage materially improved 
* [ ] Backend coverage ≥75% (or staged gate ≥70 now, ≥75 next) 
* [ ] Empty-dataset guardrail and tests agree 
* [ ] Tuning metric guardrail blocks non-locked metrics 
* [ ] React test `act()` warnings eliminated 
* [ ] CI green

---

## After M23: what comes next (M24)

**M24 = Baseline training run + benchmark harness**:

* one small, controlled training job
* evaluate on `golden-v1`
* promote to registry
* (optional) 1–2 trial “smoke tuning” only once metrics are real

If you want, I can draft the M24 Cursor prompt too—but I’d run **M23 first** so training is measuring something real, not placeholder scores.

[1]: https://kentcdodds.com/blog/fix-the-not-wrapped-in-act-warning?utm_source=chatgpt.com "Fix the \"not wrapped in act(...)\" warning"
