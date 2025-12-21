Thanks — for this particular question, you **do not need any web search** to answer the design decisions for M05. All the clarifying questions are about **internal schema, API design, UI layout, scoring, and test strategy**, which are not dependent on external web data.

So I will proceed with answering the clarifying questions based on standard backend/frontend architecture and best practices, without calling the web tool.

---

Here are my **decisions for each question in your M05 Clarifying Questions**:

---

## ✅ 1. **Database Schema for Scores**

**Decision: Option A — Separate `scores` table**

**Why:**

* Supports multiple scoring criteria (baseline, LLM judge, future metrics)
* Easy to query and index by trace or criteria
* Aligns with extensibility goals for M6+

**Conclusion:** **A**

---

## ✅ 2. **Baseline Scoring Algorithm**

**Decision: Modify the proposed formula slightly**

Use the **proposed structure**, but normalize to **0–100** range for clarity in UI and comparison. Your formula is solid; we’ll scale the result by 100.

Example adjustment:

```python
def baseline_score(trace: ReasoningTrace) -> float:
    step_count = len(trace.steps)
    avg_step_length = sum(len(s.content) for s in trace.steps) / step_count

    step_score   = min(step_count / 10.0, 1.0) * 50
    length_score = min(avg_step_length / 500.0, 1.0) * 50

    return step_score + length_score
```

Return float in **0–100** range.

**Conclusion:** **Modify formula to 0–100 range**

---

## ✅ 3. **Comparison API Response Format**

**Decision: Option A — Full payloads**

**Why:**
Frontend needs full trace data to render step lists, and this reduces roundtrips and post-processing in UI.

**Conclusion:** **A**

---

## ✅ 4. **Frontend UI Layout Preference**

**Decision: Option A — Side-by-side**

**Why:**
Supports easy visual comparison, matches audit plan language, and is more intuitive for reasoning trace analysis.

**Conclusion:** **A**

---

## ✅ 5. **Score Details Structure**

**Decision: Approve proposed structure**

**Why:**
All relevant metadata for scoring interpretation is included, and it’s easy to extend if needed.

**Conclusion:** **Approve**

---

## ✅ 6. **Caching Strategy for Scores**

**Decision: Option B — Store scores in database**

**Why:**
Scores are deterministic and can be reused for comparison, sorting, and historical querying.

**Conclusion:** **B**

---

## ✅ 7. **Error Handling — Missing Traces**

**Decision: Option A — 404 fail-fast**

**Why:**
Clear, canonical REST practice; easier client logic.

**Conclusion:** **A**

---

## ✅ 8. **M05 Scope Boundary**

**Decision:** Defer all extensions to M6

**Why:**
Keeps M05 tightly scoped; avoid feature creep.

**Conclusion:** **Confirm defer to M6**

---

## ✅ 9. **Testing — E2E Trace Setup**

**Decision: Option B — Two distinctly different traces**

**Why:**
Better surface differences in comparison UI and helps test scoring logic clearly.

**Conclusion:** **B**

---

## ✅ 10. **Documentation — API Examples Location**

**Decision: Option C — Both README.md and tunix-rt.md**

**Why:**
README is highest visibility; tunix-rt.md is canonical technical reference.

**Conclusion:** **C**

---

## ✅ 11. **Migration Naming**

**Decision:** Use `add_scores_table`

**Why:**
Clear and self-descriptive for the first scores table.

**Conclusion:** **Confirm**

---

## ✅ 12. **Frontend API Client Organization**

**Decision:** Keep in `client.ts`

**Why:**
Only two new methods in M05; avoid premature file splitting.

**Conclusion:** **Keep in client.ts**

---

## ✅ 13. **Score Endpoint Return Code**

**Decision:** `201 Created`

**Why:**
Since scores *are stored* in DB, creation semantics make sense.

**Conclusion:** **B (201 Created)**

---

## 📌 Final Summary of Decisions

| #  | Topic               | Decision                         |
| -- | ------------------- | -------------------------------- |
| 1  | Database schema     | **A: Separate `scores` table**   |
| 2  | Baseline algorithm  | **Modified to 0–100 range**      |
| 3  | Comparison response | **A: Full payloads**             |
| 4  | UI layout           | **A: Side-by-side**              |
| 5  | Score details       | **Approve proposed structure**   |
| 6  | Caching strategy    | **B: Store in DB**               |
| 7  | Missing trace error | **A: 404 fail-fast**             |
| 8  | M05 scope boundary  | **Defer to M6**                  |
| 9  | E2E test traces     | **B: two distinct**              |
| 10 | Docs location       | **C: Both README + tunix-rt.md** |
| 11 | Migration name      | **Confirm `add_scores_table`**   |
| 12 | API client file     | **Keep in client.ts**            |
| 13 | Score endpoint code | **201 Created**                  |

---

If you like, I can now generate the **M05 Cursor handoff prompt** including these decisions baked in (with schema examples, API contract, and test specs). Just say “Generate M05 Cursor prompt.”
