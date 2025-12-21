"""Pydantic schemas for trace scoring and evaluation."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from tunix_rt_backend.schemas.trace import ReasoningTrace


class ScoreRequest(BaseModel):
    """Request to score a trace.

    Attributes:
        criteria: Scoring criteria to use (currently only 'baseline')
    """

    criteria: Literal["baseline"] = Field(
        default="baseline",
        description="Scoring criteria (only 'baseline' supported in M5)",
    )


class ScoreDetails(BaseModel):
    """Detailed scoring breakdown.

    Attributes:
        step_count: Number of steps in the trace
        avg_step_length: Average character length of steps
        total_chars: Total characters across all steps
        step_score: Score contribution from step count (0-50)
        length_score: Score contribution from average length (0-50)
        criteria: Scoring criteria used
        scored_at: Timestamp when score was computed
    """

    step_count: int
    avg_step_length: float
    total_chars: int
    step_score: float
    length_score: float
    criteria: str
    scored_at: datetime


class ScoreResponse(BaseModel):
    """Response from scoring a trace.

    Attributes:
        trace_id: UUID of the scored trace
        score: Numeric score (0-100 for baseline)
        details: Detailed scoring breakdown
    """

    trace_id: uuid.UUID
    score: float = Field(..., ge=0, le=100, description="Score value (0-100)")
    details: ScoreDetails


class TraceWithScore(BaseModel):
    """Trace metadata with associated score.

    Attributes:
        id: Trace UUID
        created_at: Trace creation timestamp
        score: Score value
        trace_version: Trace format version
        payload: Full trace data
    """

    id: uuid.UUID
    created_at: datetime
    score: float
    trace_version: str
    payload: ReasoningTrace


class CompareResponse(BaseModel):
    """Response from comparing two traces.

    Attributes:
        base: Base trace with score
        other: Other trace with score
    """

    base: TraceWithScore
    other: TraceWithScore
