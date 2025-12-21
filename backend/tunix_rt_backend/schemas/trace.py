"""Pydantic schemas for reasoning traces."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TraceStep(BaseModel):
    """A single step in a reasoning trace.

    Attributes:
        i: Step index (unique, non-negative)
        type: Step type (free-form string)
        content: Step content
    """

    i: int = Field(..., ge=0, description="Step index (must be non-negative)")
    type: str = Field(..., min_length=1, max_length=64, description="Step type")
    content: str = Field(..., min_length=1, max_length=20000, description="Step content")

    @field_validator("i")
    @classmethod
    def validate_step_index(cls, v: int) -> int:
        """Ensure step index is non-negative."""
        if v < 0:
            raise ValueError("Step index must be non-negative")
        # v is valid - continue
        return v


class ReasoningTrace(BaseModel):
    """A complete reasoning trace.

    Attributes:
        trace_version: Version of the trace format
        prompt: Original prompt/question
        final_answer: Final answer/conclusion
        steps: List of reasoning steps
        meta: Optional metadata
    """

    trace_version: str = Field(..., min_length=1, max_length=64, description="Trace format version")
    prompt: str = Field(..., min_length=1, max_length=50000, description="Original prompt")
    final_answer: str = Field(..., min_length=1, max_length=50000, description="Final answer")
    steps: list[TraceStep] = Field(
        ..., min_length=1, max_length=1000, description="Reasoning steps"
    )
    meta: dict[str, Any] | None = Field(default=None, description="Optional metadata")

    @field_validator("steps")
    @classmethod
    def validate_steps_unique_indices(cls, v: list[TraceStep]) -> list[TraceStep]:
        """Ensure all step indices are unique."""
        indices = [step.i for step in v]
        if len(indices) != len(set(indices)):
            raise ValueError("Step indices must be unique")
        return v


class TraceCreateResponse(BaseModel):
    """Response from creating a trace.

    Attributes:
        id: UUID of the created trace
        created_at: Timestamp when trace was created
        trace_version: Version string from the trace
    """

    id: uuid.UUID
    created_at: datetime
    trace_version: str


class TraceDetail(BaseModel):
    """Full trace details including payload.

    Attributes:
        id: UUID of the trace
        created_at: Timestamp when trace was created
        trace_version: Version string
        payload: Full trace data
    """

    id: uuid.UUID
    created_at: datetime
    trace_version: str
    payload: ReasoningTrace


class TraceListItem(BaseModel):
    """Trace list item (without full payload).

    Attributes:
        id: UUID of the trace
        created_at: Timestamp when trace was created
        trace_version: Version string
    """

    id: uuid.UUID
    created_at: datetime
    trace_version: str


class PaginationInfo(BaseModel):
    """Pagination metadata.

    Attributes:
        limit: Maximum items per page
        offset: Current offset
        next_offset: Offset for next page (or None if last page)
    """

    limit: int
    offset: int
    next_offset: int | None


class TraceListResponse(BaseModel):
    """Response from listing traces.

    Attributes:
        data: List of trace items
        pagination: Pagination metadata
    """

    data: list[TraceListItem]
    pagination: PaginationInfo
