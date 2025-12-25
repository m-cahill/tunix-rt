# M23 Questions

1.  **`predictions.jsonl` Contract & Generation**:
    The M23 plan requires the "Tunix run/export pipeline" to write `predictions.jsonl` for the judge to consume. Currently, `training/train_sft_tunix.py` (which seems to be the script behind `tunix train` in local mode) is a placeholder from M09/M10 and does *not* generate this file.
    *   **Question**: Should I update `training/train_sft_tunix.py` to generate a `predictions.jsonl` artifact (even if using dummy data for now) to fulfill the contract?
    *   **Or**: Should I implement a separate "inference/generation" step in `tunix_execution.py` that runs *after* the training command to produce this file?

2.  **`AnswerCorrectnessJudge` Implementation**:
    The plan refers to `AnswerCorrectnessJudge.evaluate()` as a "stub/placeholder". I see an implementation in `backend/tunix_rt_backend/services/judges.py` that loads the manifest and attempts to read `predictions.jsonl`, then compares them.
    *   **Question**: Is the "stub" nature primarily about the missing input (`predictions.jsonl`) and perhaps fragile error handling, or is the comparison logic itself considered a placeholder? (The plan mentions "Normalize prediction and ground truth and compute mean correctness", which `_compare` seems to do simply).

3.  **Frontend `act()` Warnings**:
    The plan mentions removing React test warnings.
    *   **Question**: Are there specific tests that are known to be noisy, or should I run the full frontend suite and fix whatever I find?

4.  **Locked Metrics**:
    The plan mentions `LOCKED_METRICS = {"answer_correctness"}`.
    *   **Question**: Should this restriction be applied in `TuningJobCreate` schema validation, or at runtime in the `TuningService`?
