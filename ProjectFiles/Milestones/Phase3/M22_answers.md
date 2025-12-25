Here are the decisions for the M22 questions (with the rationale and how I’d implement them). 

---

## 1) Frontend tests for “Promote Best”

**Yes — create `frontend/src/components/Tuning.test.tsx`.** 

* Use the same approach as `ModelRegistry.test.tsx`: **mock the API module** with `vi.mock`, and assert:

  * the button renders when appropriate,
  * clicking calls the expected client method,
  * success triggers refresh/toast,
  * failure shows an error state.

This aligns with Vitest’s recommended mocking workflow (`vi.mock`, `vi.fn`, and clearing/resetting mocks between tests). ([Vitest][1])

**Guardrail:** keep the test **purely component-level** (no network, no Playwright) so it remains fast and stable.

---

## 2) Where to implement `answer_correctness`

**Yes — implement it as a new Judge class** (e.g., `AnswerCorrectnessJudge`) in `backend/tunix_rt_backend/services/judges.py`. 

That matches your existing pattern (`MockJudge`, `GemmaJudge`), keeps evaluation extensible, and makes it easy to later add LLM-as-judge variants while keeping deterministic metrics separate. For “correctness”, start with a **deterministic exact-match style** scorer (normalization + exact match), since match-based metrics are the simplest and most stable baseline. ([Hugging Face][2])

**Guardrail:** version the evaluator (e.g., `answer_correctness@v1`) in stored metrics so future changes don’t silently invalidate historical comparisons.

---

## 3) What should “golden-v1” contain

Create a **minimal valid, curated dataset** (start with **5–10 items**) that satisfies the schema and supports `answer_correctness`. 

You don’t need realistic breadth yet; you need a **stable benchmark** you can run repeatedly to compare runs over time. Golden datasets are explicitly meant to be a consistent reference point for measuring progress and comparing versions under identical conditions. ([DAC.digital][3])

**Recommended contents for the first 5–10 items:**

* 3–5 straightforward QA items (single correct answer)
* 1–2 normalization edge cases (whitespace/case/punctuation)
* 1–2 “format” expectations if your outputs require structure (e.g., “Answer: X”)

**Guardrail:** keep `golden-v1` *small* and *frozen*. Add `golden-v2` later rather than editing `v1`.

---

[1]: https://vitest.dev/guide/mocking?utm_source=chatgpt.com "Mocking | Guide"
[2]: https://huggingface.co/spaces/OpenEvals/evaluation-guidebook?utm_source=chatgpt.com "Evaluation Guidebook - a Hugging Face Space by ..."
[3]: https://dac.digital/what-is-a-golden-dataset/?utm_source=chatgpt.com "What Is a Golden Dataset in AI and Why Does It Matter?"
