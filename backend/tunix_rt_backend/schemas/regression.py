"""Regression schemas (M18, M35).

M35 additions: eval_set field, primary_score default, promote best run.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RegressionBaselineCreate(BaseModel):
    """Request to create a regression baseline."""

    name: str = Field(..., description="Unique name for the baseline")
    run_id: UUID = Field(..., description="Run ID to use as baseline")
    metric: str = Field(
        default="primary_score",
        description="Metric name (e.g., 'primary_score', 'score', 'answer_correctness')",
    )
    lower_is_better: bool | None = Field(None, description="Whether lower values are better")
    # M35: Additional keying fields for multiple baselines
    eval_set: str | None = Field(None, description="Eval set used (e.g., 'eval_v2.jsonl')")
    dataset_key: str | None = Field(None, description="Dataset key for scoping")


class RegressionBaselineResponse(BaseModel):
    """Response for baseline creation/retrieval."""

    id: UUID
    name: str
    run_id: UUID
    metric: str
    lower_is_better: bool | None = None
    eval_set: str | None = None
    dataset_key: str | None = None
    created_at: datetime


class PromoteBestRunRequest(BaseModel):
    """Request to promote the best run from a tuning job as baseline (M35)."""

    tuning_job_id: UUID | None = Field(None, description="Tuning job to select best from")
    run_id: UUID | None = Field(None, description="Specific run to promote (alternative)")
    baseline_name: str = Field(..., description="Name for the new baseline")
    metric: str = Field(default="primary_score", description="Metric to use")


class RegressionCheckRequest(BaseModel):
    """Request to check for regression."""

    run_id: UUID = Field(..., description="Run to check")
    baseline_name: str = Field(..., description="Baseline name to compare against")


class RegressionCheckResult(BaseModel):
    """Result of a regression check."""

    run_id: UUID
    baseline_name: str
    baseline_run_id: UUID
    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_percent: float
    verdict: Literal["pass", "fail"]
    details: str | None = None
