"""Trace scoring logic for evaluation and comparison (M35).

This module provides:
- baseline_score(): Trace-level structural scoring (0-100 scale)
- compute_primary_score(): Evaluation-level aggregation (0-1 scale)
- compute_scorecard(): Detailed evaluation statistics (n_items, stddev, per-section)

The primary_score is the canonical metric for optimization and reporting,
computed as mean(answer_correctness) across an evaluation set.

The scorecard provides additional diagnostics for understanding score distribution
and performance across different categories/sections of the eval set.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tunix_rt_backend.schemas import ReasoningTrace, ScoreDetails

# ============================================================
# Scorecard Data Class (M35)
# ============================================================


@dataclass
class Scorecard:
    """Detailed evaluation statistics for a set of evaluation results.

    Provides a structured summary of evaluation outcomes including counts,
    statistical measures, and per-category breakdowns. This is the "decision-grade"
    output that helps answer "is run B better than run A?" with confidence.

    Attributes:
        n_items: Total number of items in the evaluation set
        n_scored: Number of items that received a valid score
        n_skipped: Number of items skipped (missing predictions, errors, etc.)
        primary_score: Mean score across scored items (0-1 scale), or None
        stddev: Standard deviation of scores, or None if < 2 scored items
        min_score: Minimum score observed, or None if no scored items
        max_score: Maximum score observed, or None if no scored items
        section_scores: Per-section mean scores (e.g., {"core": 0.85, "edge_case": 0.70})
        category_scores: Per-category mean scores (e.g., {"arithmetic": 0.90})
        difficulty_scores: Per-difficulty mean scores (e.g., {"easy": 0.95, "hard": 0.60})
    """

    n_items: int = 0
    n_scored: int = 0
    n_skipped: int = 0
    primary_score: float | None = None
    stddev: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    section_scores: dict[str, float] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
    difficulty_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert scorecard to dictionary for JSON serialization."""
        return {
            "n_items": self.n_items,
            "n_scored": self.n_scored,
            "n_skipped": self.n_skipped,
            "primary_score": self.primary_score,
            "stddev": self.stddev,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "section_scores": self.section_scores,
            "category_scores": self.category_scores,
            "difficulty_scores": self.difficulty_scores,
        }


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


# ============================================================
# Scorecard Aggregation (M35)
# ============================================================


def compute_scorecard(
    evaluation_rows: list[dict[str, Any]],
    eval_items: list[dict[str, Any]] | None = None,
) -> Scorecard:
    """Compute a detailed scorecard from evaluation results.

    The scorecard provides comprehensive statistics for understanding evaluation
    outcomes, including per-section and per-category breakdowns. This enables
    decision-grade comparisons between runs.

    Args:
        evaluation_rows: List of evaluation result dictionaries. Each should contain:
            - item_id: Identifier matching eval_items (optional)
            - score: Float score value (0-1 preferred, 0-100 normalized)
            - metrics: Dict with answer_correctness or other metrics
            - section: Optional section label (core/trace_sensitive/edge_case)
            - category: Optional category label
            - difficulty: Optional difficulty label

        eval_items: Optional list of eval set items to provide metadata (section,
            category, difficulty) if not present in evaluation_rows. Each item
            should have an 'id' field matching evaluation_rows.

    Returns:
        Scorecard with aggregated statistics.

    Note:
        This function is deterministic: same inputs always produce same outputs.
        Item ordering does not affect results (internally sorted by item_id).

    Example:
        >>> rows = [
        ...     {"item_id": "001", "score": 1.0, "section": "core"},
        ...     {"item_id": "002", "score": 0.5, "section": "core"},
        ...     {"item_id": "003", "score": 0.0, "section": "edge_case"},
        ... ]
        >>> card = compute_scorecard(rows)
        >>> card.n_items
        3
        >>> card.primary_score
        0.5
        >>> card.section_scores
        {'core': 0.75, 'edge_case': 0.0}
    """
    # Build item metadata lookup from eval_items if provided
    item_metadata: dict[str, dict[str, Any]] = {}
    if eval_items:
        for item in eval_items:
            item_id = item.get("id")
            if item_id:
                item_metadata[str(item_id)] = {
                    "section": item.get("section"),
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                }

    # Track scores and breakdowns
    all_scores: list[float] = []
    section_scores: dict[str, list[float]] = {}
    category_scores: dict[str, list[float]] = {}
    difficulty_scores: dict[str, list[float]] = {}
    n_skipped = 0

    # Sort rows by item_id for deterministic processing
    sorted_rows = sorted(
        evaluation_rows,
        key=lambda r: str(r.get("item_id") or r.get("id") or ""),
    )

    for row in sorted_rows:
        # Extract score value (same priority as compute_primary_score)
        score_value = _extract_score(row)

        if score_value is None:
            n_skipped += 1
            continue

        all_scores.append(score_value)

        # Get metadata (from row or from eval_items lookup)
        item_id = str(row.get("item_id") or row.get("id") or "")
        metadata = item_metadata.get(item_id, {})

        section = row.get("section") or metadata.get("section")
        category = row.get("category") or metadata.get("category")
        difficulty = row.get("difficulty") or metadata.get("difficulty")

        # Aggregate by section
        if section:
            if section not in section_scores:
                section_scores[section] = []
            section_scores[section].append(score_value)

        # Aggregate by category
        if category:
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score_value)

        # Aggregate by difficulty
        if difficulty:
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            difficulty_scores[difficulty].append(score_value)

    # Compute statistics
    n_items = len(sorted_rows)
    n_scored = len(all_scores)

    primary_score = sum(all_scores) / n_scored if all_scores else None
    stddev = _compute_stddev(all_scores) if len(all_scores) >= 2 else None
    min_score = min(all_scores) if all_scores else None
    max_score = max(all_scores) if all_scores else None

    # Compute per-group means
    section_means = {k: sum(v) / len(v) for k, v in section_scores.items()}
    category_means = {k: sum(v) / len(v) for k, v in category_scores.items()}
    difficulty_means = {k: sum(v) / len(v) for k, v in difficulty_scores.items()}

    return Scorecard(
        n_items=n_items,
        n_scored=n_scored,
        n_skipped=n_skipped,
        primary_score=primary_score,
        stddev=stddev,
        min_score=min_score,
        max_score=max_score,
        section_scores=section_means,
        category_scores=category_means,
        difficulty_scores=difficulty_means,
    )


def _extract_score(row: dict[str, Any]) -> float | None:
    """Extract a normalized 0-1 score from an evaluation row.

    Same logic as compute_primary_score for consistency.

    Args:
        row: Single evaluation result dictionary

    Returns:
        Score as float in [0, 1], or None if no valid score found
    """
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

    # Priority 2: Check for correctness directly on row
    if score_value is None:
        correctness = row.get("correctness")
        if correctness is not None:
            try:
                score_value = float(correctness)
            except (TypeError, ValueError):
                pass

    # Priority 3: Fallback to normalized score field
    if score_value is None:
        raw_score = row.get("score")
        if raw_score is not None:
            try:
                raw_float = float(raw_score)
                # Normalize 0-100 scale to 0-1 scale
                if raw_float > 1.0:
                    score_value = raw_float / 100.0
                else:
                    score_value = raw_float
            except (TypeError, ValueError):
                pass

    # Validate score is in valid range
    if score_value is not None and 0.0 <= score_value <= 1.0:
        return score_value

    return None


def _compute_stddev(values: list[float]) -> float:
    """Compute population standard deviation.

    Args:
        values: List of numeric values (must have at least 2 elements)

    Returns:
        Standard deviation as float
    """
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)
