"""Trace scoring logic for evaluation and comparison.

This module provides:
- baseline_score(): Trace-level structural scoring (0-100 scale)
- compute_primary_score(): Evaluation-level aggregation (0-1 scale)

The primary_score is the canonical metric for optimization and reporting,
computed as mean(answer_correctness) across an evaluation set.
"""

from datetime import datetime, timezone
from typing import Any

from tunix_rt_backend.schemas import ReasoningTrace, ScoreDetails


def baseline_score(trace: ReasoningTrace) -> tuple[float, ScoreDetails]:
    """Compute baseline score for a reasoning trace.

    This is a deterministic scorer that evaluates trace quality based on
    structural properties: step count and average step length.

    Score range: 0-100
    - Step score (0-50): Rewards having 1-10 steps (ideal range)
    - Length score (0-50): Rewards average step length of 100-500 chars (ideal range)

    Args:
        trace: ReasoningTrace to score

    Returns:
        Tuple of (score, details) where:
        - score: float in range 0-100
        - details: ScoreDetails with breakdown of scoring components
    """
    step_count = len(trace.steps)
    avg_step_length = sum(len(step.content) for step in trace.steps) / step_count
    total_chars = sum(len(step.content) for step in trace.steps)

    # Normalize step count (1-10 steps ideal range)
    # Score increases linearly up to 10 steps, then caps at 50
    step_score = min(step_count / 10.0, 1.0) * 50

    # Normalize avg step length (100-500 chars ideal)
    # Score increases linearly up to 500 chars, then caps at 50
    length_score = min(avg_step_length / 500.0, 1.0) * 50

    total_score = step_score + length_score

    # Build details object
    details = ScoreDetails(
        step_count=step_count,
        avg_step_length=round(avg_step_length, 2),
        total_chars=total_chars,
        step_score=round(step_score, 2),
        length_score=round(length_score, 2),
        criteria="baseline",
        scored_at=datetime.now(timezone.utc),
    )

    return round(total_score, 2), details


# ============================================================
# Primary Score (M34) - Evaluation-level Aggregation
# ============================================================


def compute_primary_score(evaluation_rows: list[dict[str, Any]]) -> float | None:
    """Compute the primary score from a list of evaluation results.

    The primary score is the canonical metric for optimization and reporting.
    It aggregates individual evaluation results into a single scalar value
    in the range [0, 1].

    Priority:
    1. Use `metrics["answer_correctness"]` if present (already 0-1 scale)
    2. Fallback to `score / 100.0` if score is 0-100 scale
    3. Skip rows with no valid score

    Args:
        evaluation_rows: List of evaluation result dictionaries, each containing
            at least one of:
            - metrics["answer_correctness"]: float in [0, 1]
            - score: float (typically 0-100 scale)

    Returns:
        Mean of valid scores as float in [0, 1], or None if no valid rows.

    Example:
        >>> rows = [
        ...     {"metrics": {"answer_correctness": 0.8}},
        ...     {"metrics": {"answer_correctness": 1.0}},
        ...     {"score": 60},  # Fallback: 60/100 = 0.6
        ... ]
        >>> compute_primary_score(rows)
        0.8
    """
    valid_scores: list[float] = []

    for row in evaluation_rows:
        score_value: float | None = None

        # Priority 1: Check for answer_correctness in metrics
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            answer_correctness = metrics.get("answer_correctness")
            if answer_correctness is not None:
                try:
                    score_value = float(answer_correctness)
                except (TypeError, ValueError):
                    pass

        # Priority 2: Fallback to normalized score field
        if score_value is None:
            raw_score = row.get("score")
            if raw_score is not None:
                try:
                    raw_float = float(raw_score)
                    # Normalize 0-100 scale to 0-1 scale
                    # If score is already in 0-1 range, don't double-normalize
                    if raw_float > 1.0:
                        score_value = raw_float / 100.0
                    else:
                        score_value = raw_float
                except (TypeError, ValueError):
                    pass

        # Validate score is in valid range
        if score_value is not None and 0.0 <= score_value <= 1.0:
            valid_scores.append(score_value)

    # Return mean or None
    if not valid_scores:
        return None

    return sum(valid_scores) / len(valid_scores)
