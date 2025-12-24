# M17 Clarifying Questions

1.  **Automatic Triggering:**
    *   Should the evaluation loop run **automatically** immediately after a Tunix run completes successfully?
    *   Should it also be possible to **manually trigger/re-trigger** evaluation for a past run via the API?

2.  **Database & Leaderboard Performance:**
    *   To support the "Leaderboard" (sorting/filtering by metrics), should we create a dedicated `tunix_run_evaluations` table in Postgres with columns for key metrics (e.g., `score`, `accuracy`), linked to `tunix_runs`?
    *   Or should we rely solely on the `evaluation.json` artifact (which would make sorting/filtering slow/hard)?
    *   *Recommendation: Dedicated table for high-level metrics + JSON artifact for full details.*

3.  **Judge Implementation:**
    *   For the initial `gemma-judge-v1` implementation, should I create a **mock judge** that returns deterministic/random scores to validate the infrastructure first?
    *   Or do you have a specific simple logic (e.g., regex matching, length checks) you want implemented immediately?

4.  **API & Frontend:**
    *   Should the evaluation data be included directly in the `GET /api/tunix/runs/{id}` response, or should it be a separate endpoint `GET /api/tunix/runs/{id}/evaluation`?
    *   For the Leaderboard UI, do you prefer a **new top-level page** (e.g., "/leaderboard") or should this be a new tab/view within the existing "Runs" page?
