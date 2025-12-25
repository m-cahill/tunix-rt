# M28 Clarifying Questions

1.  **UNGAR Integration (Phase 5):** The plan lists Option A (Fix) as preferred over Option B (xfail). Should I time-box the fix attempt (e.g., 1 hour) before falling back to xfail, or should I prioritize fixing it fully?
2.  **Tuning Sweep (Phase 2):** The plan mentions using "existing M19 tuning infrastructure". I see the `TuningService` and endpoints. Does this infrastructure currently support the specific search space (LR, batch size, etc.) mentioned, or will I need to extend the schema/service to support these specific parameters if they aren't already generic?
3.  **Eval Score Definition (Phase 4):** For the "eval score" on the leaderboard, is a simple average of `answer_correctness` across the evaluation set sufficient, or do you have a specific aggregation formula in mind?
4.  **Run Comparison UI (Phase 3):** Do you have a preference for where this "side-by-side" view should live? Should it be a new top-level page (e.g., `/compare?runA=...&runB=...`), or a modal triggered from the Run History list?
