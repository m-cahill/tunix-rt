# M05 Clarifying Questions

**Date:** 2025-12-21  
**Milestone:** M05 - Evaluation & Comparison Loop (Phase 1)  
**Status:** Awaiting Responses

---

## 1. Database Schema for Scores

**Question:** Should we store scores in a separate table or include them in the trace record?

**Option A: Separate `scores` table**
- Pros: Can have multiple scores per trace (different criteria), clean separation of concerns, easier to add new score types later
- Schema:
  ```sql
  scores:
    - id: UUID (PK)
    - trace_id: UUID (FK to traces)
    - criteria: VARCHAR(64) (e.g., "baseline", "llm_judge")
    - score: FLOAT
    - details: JSON (additional scoring metadata)
    - created_at: TIMESTAMPTZ
  ```

**Option B: Add score columns to `traces` table**
- Pros: Simpler queries, fewer joins
- Cons: Less flexible for multiple scoring criteria

**Option C: Store scores in payload JSON only (no schema change)**
- Pros: Zero migrations, fastest to implement
- Cons: No indexing, harder to query by score

**Recommendation:** Option A for future extensibility (M6 will add LLM judges).

**Your Decision:** _[Please specify A, B, or C, or suggest alternative]_

---

## 2. Baseline Scoring Algorithm

**Question:** What specific scoring formula should we use for the baseline scorer?

The M05 plan suggests:
- `length_of_trace_steps` or
- `1 / (1 + step_count)`

**Additional considerations:**
- Should we penalize/reward based on final_answer length?
- Should we consider prompt complexity?
- Should the score be normalized to a 0-1 or 0-100 range?

**Proposed Baseline Scorer (for discussion):**
```python
def baseline_score(trace: ReasoningTrace) -> float:
    """
    Simple deterministic scorer based on trace structure.
    Higher scores = more detailed reasoning.
    
    Score range: 0.0 - 1.0
    """
    step_count = len(trace.steps)
    avg_step_length = sum(len(step.content) for step in trace.steps) / step_count
    
    # Normalize step count (1-10 steps ideal range)
    step_score = min(step_count / 10.0, 1.0) * 0.5
    
    # Normalize avg step length (100-500 chars ideal)
    length_score = min(avg_step_length / 500.0, 1.0) * 0.5
    
    return step_score + length_score
```

**Your Decision:** _[Use as-is, modify formula, or provide alternative]_

---

## 3. Comparison API Response Format

**Question:** Should the comparison endpoint return both traces with their full payloads, or just metadata + scores?

**Option A: Full payloads (verbose)**
```json
{
  "base": {
    "id": "...",
    "created_at": "...",
    "score": 0.75,
    "trace_version": "1.0",
    "payload": { /* full ReasoningTrace */ }
  },
  "other": { /* same structure */ }
}
```

**Option B: Metadata only (lean)**
```json
{
  "base": {
    "id": "...",
    "score": 0.75,
    "step_count": 5,
    "prompt_preview": "What is 27 × 19?"
  },
  "other": { /* same */ },
  "comparison": {
    "score_delta": -0.15,
    "winner": "base"
  }
}
```

**Recommendation:** Option A for M5 (frontend will need full data), Option B optimization in M6 if needed.

**Your Decision:** _[A or B]_

---

## 4. Frontend UI Layout Preference

**Question:** How should the comparison UI be structured?

**Option A: Single-page side-by-side**
- Two columns: Base (left) | Other (right)
- Scores displayed at top of each column
- Trace steps scrollable within columns
- Good for detailed comparison

**Option B: Tabbed view**
- Tab 1: Base trace
- Tab 2: Other trace
- Tab 3: Comparison summary
- Good for focused review

**Option C: Accordion/expandable**
- Compact comparison summary visible by default
- Click to expand full trace details
- Good for mobile/smaller screens

**Recommendation:** Option A (aligns with M05 plan "side-by-side" language).

**Your Decision:** _[A, B, or C]_

---

## 5. Score Details Structure

**Question:** What should be included in the `details` field of the score response?

**Proposed structure:**
```json
{
  "trace_id": "550e8400-...",
  "score": 0.75,
  "details": {
    "step_count": 5,
    "avg_step_length": 342,
    "total_chars": 1710,
    "step_score": 0.5,
    "length_score": 0.25,
    "criteria": "baseline",
    "scored_at": "2025-12-21T10:30:00Z"
  }
}
```

**Your Decision:** _[Approve, modify, or provide alternative]_

---

## 6. Caching Strategy for Scores

**Question:** Should we cache computed scores?

**Option A: No caching**
- Recompute on every request
- Pros: Always accurate, simple
- Cons: Wasteful for deterministic baseline scorer

**Option B: Store scores in database**
- Cache in `scores` table (if we create it)
- Pros: Fast retrieval, queryable
- Cons: Extra write on score endpoint

**Option C: In-memory TTL cache (like RediAI health)**
- Pros: Fast, no DB changes
- Cons: Cache invalidation complexity

**Recommendation:** Option B (aligns with separate scores table from Q1).

**Your Decision:** _[A, B, or C]_

---

## 7. Error Handling - Missing Traces

**Question:** For `GET /api/traces/compare?base=ID1&other=ID2`, what should we return if only one trace exists?

**Option A: 404 with specific message**
```json
{
  "detail": "Base trace abc123... not found"
}
```

**Option B: Partial response**
```json
{
  "base": null,
  "other": { /* trace data */ },
  "error": "Base trace not found"
}
```

**Recommendation:** Option A (fail fast, clearer contract).

**Your Decision:** _[A or B]_

---

## 8. M05 Scope Boundary

**Question:** Should we include the following in M05, or defer to M6?

- [ ] List all scores for a trace: `GET /api/traces/{id}/scores`
- [ ] Delete a score: `DELETE /api/scores/{score_id}`
- [ ] Comparison history/favorites
- [ ] Bulk scoring: `POST /api/traces/score` (score multiple traces)

**Recommendation:** Defer ALL to M6. M05 scope is:
1. Single trace scoring endpoint
2. Two-trace comparison endpoint
3. Baseline scorer only
4. Minimal UI

**Your Decision:** _[Confirm defer to M6, or specify which to include]_

---

## 9. Testing - E2E Trace Setup

**Question:** For the E2E test "Create two example traces (via POST) → Call Compare UI → Assert side-by-side rendering & scores", should we:

**Option A:** Use the existing `EXAMPLE_TRACE` from `exampleTrace.ts` with minor variations
**Option B:** Create two distinctly different traces (e.g., simple math vs complex reasoning)

**Recommendation:** Option B for better comparison visibility in tests.

**Your Decision:** _[A or B, or provide specific trace examples]_

---

## 10. Documentation - API Examples Location

**Question:** Where should we add the new API curl examples?

**Option A:** Update `tunix-rt.md` (main technical doc)
**Option B:** Update `README.md` (user-facing quick start)
**Option C:** Both
**Option D:** New file `docs/api-examples.md`

**Recommendation:** Option C (tunix-rt.md for completeness, README.md for discoverability).

**Your Decision:** _[A, B, C, or D]_

---

## 11. Migration Naming

**Question:** What should the migration be named?

**Suggested:** `add_scores_table` (if we go with Option A from Q1)

**Your Decision:** _[Confirm or provide alternative]_

---

## 12. Frontend API Client Organization

**Question:** Should we add new methods to `frontend/src/api/client.ts`, or create a separate `frontend/src/api/scoring.ts`?

**Recommendation:** Keep in `client.ts` for M5 (only 2 new methods), split in M6 if scoring grows.

**Your Decision:** _[Keep in client.ts or split now]_

---

## 13. Score Endpoint Return Code

**Question:** Should `POST /api/traces/{id}/score` return:

**Option A:** 200 OK (score computed, not creating new resource semantically)
**Option B:** 201 Created (if we store score in database)

**Recommendation:** 
- If scores are stored (Q6 Option B): **201 Created**
- If scores are not stored (Q6 Option A): **200 OK**

**Your Decision:** _[A or B, pending Q6 answer]_

---

## Summary - Key Decisions Needed

| # | Topic | Recommendation | Your Decision |
|---|-------|----------------|---------------|
| 1 | Database schema | Separate `scores` table (Option A) | ? |
| 2 | Baseline algorithm | Proposed formula (step + length) | ? |
| 3 | Comparison response | Full payloads (Option A) | ? |
| 4 | UI layout | Side-by-side (Option A) | ? |
| 5 | Score details | Proposed structure | ? |
| 6 | Caching strategy | Store in DB (Option B) | ? |
| 7 | Missing trace error | 404 fail-fast (Option A) | ? |
| 8 | M05 scope boundary | Defer extensions to M6 | ? |
| 9 | E2E test traces | Two distinct traces (Option B) | ? |
| 10 | Documentation location | Both tunix-rt.md + README.md | ? |
| 11 | Migration name | `add_scores_table` | ? |
| 12 | API client file | Keep in client.ts | ? |
| 13 | Score endpoint code | 201 Created (if stored) | ? |

---

**Instructions:** Please review and provide decisions for each question. I will wait for your responses before creating the TODO list and beginning implementation.

---

**Prepared by:** AI Assistant (Claude Sonnet 4.5)  
**Status:** READY FOR REVIEW
