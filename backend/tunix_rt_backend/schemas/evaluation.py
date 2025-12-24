"""Evaluation schemas (M17).

Defines the structure for run evaluations, metrics, and judge outputs.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from tunix_rt_backend.schemas import PaginationInfo


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
    score: float = Field(..., description="Primary score")
    verdict: str = Field(..., description="Verdict")
    metrics: dict[str, float] = Field(..., description="Key metrics (e.g. accuracy)")
    evaluated_at: str = Field(..., description="Evaluation timestamp")


class LeaderboardResponse(BaseModel):
    """Leaderboard response with pagination (M18)."""

    data: list[LeaderboardItem] = Field(..., description="List of leaderboard entries")
    pagination: PaginationInfo | None = Field(None, description="Pagination metadata")
