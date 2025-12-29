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

---

## Per-Item Predictions (M36)

### Current State

Evaluation computes an aggregate `primary_score` (mean of item-level `answer_correctness` values) but **does not persist individual predictions**.

**What is available today:**
- `primary_score`: Float 0-1, aggregate metric
- `scorecard`: Object with `n_items`, `n_scored`, `n_skipped`, section/category/difficulty breakdowns
- `metrics`: Dict with overall metric values

**What is NOT available:**
- Per-item `{item_id, expected, predicted, correctness}` records
- Full text of model predictions vs ground truth
- Detailed diff between runs at item level

### Run Comparison Limitation

The **Run Comparison** UI (`RunComparison.tsx`) can display:
- Loss curve overlay (from training metrics)
- Aggregate score comparison
- Basic metadata diff (model, dataset, config)

It **cannot** display:
- Per-item prediction diffs (expected vs predicted text)
- Which specific items changed between runs
- Item-level debugging information

When using `MockJudge` (dry-run mode), the per-item diff table shows "unavailable" because no real predictions are generated.

### Planned Improvement (M37)

**M37** will add per-item artifact storage:

1. **Schema**: `{item_id: string, expected: string, predicted: string, correctness: float, eval_time_ms: int}`
2. **Storage**: Append to `tunix_run_predictions` table or JSONL artifact
3. **API**: `GET /api/tunix/runs/{run_id}/predictions` endpoint
4. **UI**: Full diff table in RunComparison with text comparison

This will enable:
- Debugging which items regressed between runs
- Identifying model weaknesses by item category
- Per-item regression tests

---

## Eval Sets

### eval_v1.jsonl (Legacy)

- **Items**: 50
- **Purpose**: Quick validation
- **Composition**: Mixed difficulty

### eval_v2.jsonl (Recommended)

- **Items**: 100
- **Purpose**: Competition-grade evaluation
- **Composition**:
  - Core: 60 items (60%)
  - Trace-sensitive: 25 items (25%)
  - Edge cases: 15 items (15%)
- **Categories**: arithmetic, geometry, unit_conversion, general_knowledge, multi_step_word_problem, algebraic_reasoning, sequence_reasoning, formatting, numeric_precision, edge_case
- **Difficulty**: easy (34%), medium (41%), hard (25%)

### Validation

Validate eval set schema with:

```bash
python backend/tools/validate_evalset.py training/evalsets/eval_v2.jsonl
```

---

## See Also

- [M36 Kaggle Run Guide](M36_KAGGLE_RUN.md) - How to run on Kaggle and capture evidence
- [Training End-to-End](training_end_to_end.md) - Full training workflow
