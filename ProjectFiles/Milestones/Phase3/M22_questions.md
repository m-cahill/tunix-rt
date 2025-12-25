
# M22 Questions

1. **Frontend Tests:** `frontend/src/components/Tuning.test.tsx` does not exist. Should I create this file to test the "Promote Best" functionality? I assume I should use `vi.mock` for the API calls similar to `ModelRegistry.test.tsx`.

2. **Evaluator Implementation:** For Phase 2.2, should `answer_correctness` be implemented as a new `Judge` class (e.g., `AnswerCorrectnessJudge`) in `backend/tunix_rt_backend/services/judges.py`? This seems consistent with the existing `MockJudge` and `GemmaJudge`.

3. **Golden Dataset:** For Phase 3.1 ("golden-v1"), do you have specific examples/content in mind, or should I create a minimal valid dataset (e.g., 5-10 simple items) that satisfies the schema?
