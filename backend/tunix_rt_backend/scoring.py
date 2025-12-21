"""Trace scoring logic for evaluation and comparison."""

from datetime import datetime, timezone

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
