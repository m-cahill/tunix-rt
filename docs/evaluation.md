# Evaluation Specification

This document defines the **frozen evaluation semantics** for Tunix RT training. These metrics serve as the ground truth for optimization and regression testing.

## Primary Metric: Answer Correctness

*   **Metric Name**: `answer_correctness`
*   **Scale**: `0` or `1` (Binary)
*   **Target**: Maximize (1.0 is perfect)
*   **Aggregation**: Mean across dataset

### Definition

A run output is considered **correct** (`1.0`) if the prediction matches the ground truth answer, subject to normalization rules. Otherwise, it is **incorrect** (`0.0`).

### Normalization Rules (MVP)

1.  **Whitespace**: Leading/trailing whitespace is stripped.
2.  **Case**: Comparison is case-insensitive.
3.  **Format**: If the output is expected to be a structured answer (e.g., "Answer: X"), the system extracts the answer part before comparison. (For `golden-v1`, we assume exact content matching after whitespace/case normalization).

### Failure Handling

*   **Missing Prediction**: Score `0.0`.
*   **Run Failed/Crashed**: Score `0.0` (Verdict: Fail).
*   **Timeout**: Score `0.0` (Verdict: Fail).

## Future Metrics (Planned)

*   **LLM-as-Judge**: For open-ended generation (coherence, helpfulness).
*   **Strict Format Compliance**: For JSON/Structured output tasks.

## Versioning

Evaluators are versioned.
*   Current Version: `answer_correctness@v1`
*   Implementation: `AnswerCorrectnessJudge` (Deterministic)
