"""Regression schemas (M18)."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RegressionBaselineCreate(BaseModel):
    """Request to create a regression baseline."""

    name: str = Field(..., description="Unique name for the baseline")
    run_id: UUID = Field(..., description="Run ID to use as baseline")
    metric: str = Field(..., description="Metric name (e.g., 'score', 'accuracy')")
    lower_is_better: bool | None = Field(None, description="Whether lower values are better")


class RegressionBaselineResponse(BaseModel):
    """Response for baseline creation/retrieval."""

    id: UUID
    name: str
    run_id: UUID
    metric: str
    lower_is_better: bool | None = None
    created_at: datetime


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
