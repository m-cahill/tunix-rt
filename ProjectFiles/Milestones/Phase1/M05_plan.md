Great — since **M4 is green** and you’ve already provided the **audit + summary for M4**, we can now define **M5** in a way that keeps momentum: focused, product-forward, and still “small + verified”.

Below is a **Cursor-ready handoff prompt** for **Milestone M5**.

---

## 📌 M5 Overview

**Milestone M5: Evaluation & Comparison Loop (Phase 1)**
Make Tunix-RT *evaluation-ready* by adding:

1. **Trace comparison API** — side-by-side trace results
2. **Evaluation scoring endpoint** — attach a simple score to a trace
3. **Baseline scorer implementation** — rule-based scorer (e.g., length, steps count, outcome correctness)
4. **Frontend UI for comparison & scoring**
5. **Tests & CI gates**

This milestone stays focused on *evaluation primitives*, not complete judge/LLM-based evaluation (that’s in M6).

---

## 🧠 Cursor Handoff Prompt for M5

````text
Implement Milestone M5 for m-cahill/tunix-rt.

Context:
- M4 is complete; CI is green.
- We now want the *Evaluation & Comparison Loop* Phase 1.
- M5 is NOT full AI judging yet; just the first concrete evaluation primitives.

M5 Goals:
1) Provide an API to store, compare, and score reasoning traces.
2) Provide minimal *baseline evaluation* (non-LLM scoring).
3) Expose UI to view and compare traces + scores.
4) Keep it tested and gated via CI.

====================================================
DELIVERABLES (Definition of Done)
====================================================

A) Backend — Evaluation API
1) **POST /api/traces/{id}/score**
   - Accepts a simple evaluation request body:
     ```json
     {
       "criteria": "baseline"
     }
     ```
   - Returns:
     ```json
     {
       "trace_id": "...",
       "score": number,
       "details": { ... }
     }
     ```

2) **GET /api/traces/compare?base=ID1&other=ID2**
   - Returns side-by-side:
     ```
     {
       "base": { trace, score },
       "other": { trace, score }
     }
     ```
   - Use baseline scorer for scoring.

3) **Backend scoring logic**
   - Implement a baseline, deterministic scorer:
     - Score = `length_of_trace_steps` or
     - Score = `1 / (1 + step_count)` or another monotonic function.
   - Must be deterministic and unit tested.

4) Validation & error handling:
   - Return 404 if traces not found.
   - Return 400 if invalid query params.

B) Frontend — Evaluation UI
1) Add UI section "Evaluate Traces"
   - Inputs:
     - Trace ID (base)
     - Trace ID (other)
   - Buttons:
     - "Fetch & Compare"
   - Shows:
     ```
     Base Trace: [id]  Score: [score]
     Other Trace: [id] Score: [score]
     ```
   - Display trace step lists side-by-side.
   - Minimal styling; focus on functional display.

2) API client extensions:
   - Add `scoreTrace(id)` and `compareTraces(base, other)` methods.
   - Add appropriate TypeScript types.

3) Unit tests (frontend):
   - Mock fetch calls for score & compare.
   - Assert UI displays base/other scores and lists.

C) Tests & CI
1) Backend:
   - Unit tests for scoring logic.
   - Endpoint tests for valid & invalid paths.
2) Frontend:
   - Component tests for Evaluate UI.
3) E2E:
   - Add Playwright test:
     - Create two example traces (via POST).
     - Call Compare UI.
     - Assert side-by-side rendering & scores.

4) CI:
   - Update CI workflows to require passing new tests.
   - Keep coverage gates unchanged.

D) Docs + Examples
1) Update README with:
   - API docs for `/api/traces/{id}/score` and `/api/traces/compare`.
   - Example curl commands.
2) Add example trace payload for M5 (in docs folder).

====================================================
PHASED DELIVERY PLAN
====================================================
Phase 1: Scoring backend
- Add scoring logic + tests.
- Add POST /api/traces/{id}/score.
- Add GET /api/traces/compare.

Phase 2: Frontend UI
- Add Evaluate UI + API client methods.
- Add unit tests.

Phase 3: E2E Comparison
- Add Playwright test for compare flow.

Phase 4: Docs
- Add API & UI usage samples to README.

====================================================
GUARDRAILS
====================================================
- No LLM/AI judging yet; keep baseline scoring only.
- Do not change existing trace schema.
- Do not refactor backend/database abstractions.
- Keep API surface minimal.

====================================================
VERIFICATION CHECKLIST
====================================================
Local:
- Backend tests passing.
- Frontend tests passing.
- Compare UI functional in dev.

CI:
- All jobs green.
- New API tests covered.
- No drops in coverage gates.

====================================================
COMMIT STYLE
====================================================
Use Conventional Commits:
- feat(api): add trace scoring endpoint
- test(api): add scoring logic tests
- feat(ui): add evaluate traces component
- test(ui): add evaluate UI tests

````

---

## 🧠 Why this scope for M5

* Builds directly on M4’s trace persistence — now we need *evaluation primitives*.
* Scoring + comparison is a **small but meaningful next step** toward judge functions.
* No new complicated dependencies; deterministic logic first.
* Makes UI useful for demos.

---

## Optional Future Enhancements (for M6+)

If you want later:

* Add LLM-based judge scoring integration (external API).
* Add a **dataset of reference traces** for benchmarking.
* Add historical performance dashboard.

---

If you want, I can also draft a **database model extension** (e.g., a `scores` table) and the **minimal UI sketch** (wireframe + component breakdown).
