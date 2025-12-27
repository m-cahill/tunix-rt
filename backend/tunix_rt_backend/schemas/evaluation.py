"""Evaluation schemas (M17, M35).

Defines the structure for run evaluations, metrics, and judge outputs.
M35 additions: LeaderboardFilters, ScorecardSummary for decision-grade leaderboard.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from tunix_rt_backend.schemas import PaginationInfo

# ============================================================
# Scorecard Summary (M35)
# ============================================================


class ScorecardSummary(BaseModel):
    """Compact scorecard summary for leaderboard display (M35).

    Provides quick visibility into evaluation quality without
    requiring a separate API call.
    """

    n_items: int = Field(..., description="Total items in evaluation")
    n_scored: int = Field(..., description="Items that received valid scores")
    n_skipped: int = Field(..., description="Items skipped (missing predictions, etc.)")
    primary_score: float | None = Field(None, description="Mean score (0-1)")
    stddev: float | None = Field(None, description="Standard deviation of scores")


# ============================================================
# Leaderboard Filters (M35)
# ============================================================


class LeaderboardFilters(BaseModel):
    """Query parameters for filtering leaderboard results (M35).

    All filters use AND logic (additive). Empty/None values are ignored.
    """

    dataset_key: str | None = Field(None, description="Filter by dataset key (exact match)")
    model_id: str | None = Field(None, description="Filter by model ID (contains match)")
    config_path: str | None = Field(None, description="Filter by config path (contains match)")
    date_from: datetime | None = Field(None, description="Filter by evaluation date (>=)")
    date_to: datetime | None = Field(None, description="Filter by evaluation date (<=)")


class EvaluationMetric(BaseModel):
    """Single evaluation metric."""

    name: str = Field(..., description="Metric name (e.g. 'accuracy', 'length')")
    score: float = Field(..., description="Numeric score value")
    max_score: float = Field(default=1.0, description="Maximum possible score")
    details: dict[str, Any] | None = Field(None, description="Additional metric details")


class EvaluationJudgeInfo(BaseModel):
    """Information about the judge used."""

    name: str = Field(..., description="Judge name (e.g. 'mock-judge', 'gemma-judge')")
    version: str = Field(..., description="Judge version")


class EvaluationResult(BaseModel):
    """Full evaluation result payload (stored as artifact)."""

    run_id: UUID = Field(..., description="ID of the run being evaluated")
    score: float = Field(..., description="Primary aggregate score (0-100)")
    verdict: Literal["pass", "fail", "uncertain"] = Field(..., description="Final verdict")
    metrics: dict[str, float] = Field(..., description="Key metrics for quick access")
    detailed_metrics: list[EvaluationMetric] = Field(..., description="Detailed metric breakdown")
    judge: EvaluationJudgeInfo = Field(..., description="Judge information")
    evaluated_at: str = Field(..., description="Evaluation timestamp (ISO-8601)")
    # M34: Canonical 0-1 primary score for optimization/reporting
    primary_score: float | None = Field(
        None,
        description="Canonical primary score in [0,1] range (mean of answer_correctness)",
    )


class EvaluationRequest(BaseModel):
    """Request to trigger evaluation."""

    judge_override: str | None = Field(None, description="Optional judge configuration override")


class EvaluationResponse(EvaluationResult):
    """Response for evaluation endpoints."""

    evaluation_id: UUID = Field(..., description="Unique ID of the evaluation record")


class LeaderboardItem(BaseModel):
    """Leaderboard entry."""

    run_id: str = Field(..., description="Run ID")
    model_id: str = Field(..., description="Model ID")
    dataset_key: str = Field(..., description="Dataset Key")
    config_path: str | None = Field(None, description="Config file path (M35)")
    score: float = Field(..., description="Primary score (0-100)")
    verdict: str = Field(..., description="Verdict")
    metrics: dict[str, float] = Field(..., description="Key metrics (e.g. accuracy)")
    evaluated_at: str = Field(..., description="Evaluation timestamp")
    # M34: Canonical 0-1 primary score
    primary_score: float | None = Field(
        None,
        description="Canonical primary score in [0,1] range",
    )
    # M35: Inline scorecard summary
    scorecard: ScorecardSummary | None = Field(
        None,
        description="Quick scorecard summary (optional)",
    )


class LeaderboardResponse(BaseModel):
    """Leaderboard response with pagination and filters (M18, M35)."""

    data: list[LeaderboardItem] = Field(..., description="List of leaderboard entries")
    pagination: PaginationInfo | None = Field(None, description="Pagination metadata")
    # M35: Applied filters for reference
    filters: LeaderboardFilters | None = Field(None, description="Applied filter criteria")
