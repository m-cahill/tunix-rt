# M18 Clarifying Questions

1.  **GemmaJudge & LLM Provider:**
    *   How should `GemmaJudge` connect to the Gemma model?
    *   Is **RediAI** intended to be the inference provider? If so, I need to extend `RediClient` with an inference method (e.g., `chat` or `generate`). Do you have the API spec for that endpoint?
    *   Or should I use a standard library (e.g., `google-generativeai`, `ollama`, `openai` client) to connect to an external/local provider?

2.  **Regression Baseline Storage:**
    *   For comparing against a "named baseline run", how should we store/mark a run as a baseline?
    *   Should I add a `is_baseline` boolean flag to `TunixRun` or `TunixRunEvaluation`?
    *   Or should I create a separate `RegressionBaseline` table that points to specific runs?

3.  **Frontend Scope:**
    *   The plan mentions "Update frontend to consume paginated results".
    *   Should I implement the frontend changes (React/Typescript) as part of this milestone? (I assume yes, but confirming).
