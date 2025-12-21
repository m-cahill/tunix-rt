"""Pydantic schemas for UNGAR integration endpoints."""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class UngarStatusResponse(BaseModel):
    """UNGAR availability status.

    Attributes:
        available: True if UNGAR is installed and importable
        version: UNGAR version string (if available)
    """

    available: bool
    version: str | None = None


class UngarGenerateRequest(BaseModel):
    """Request to generate UNGAR traces.

    Attributes:
        count: Number of episodes to generate (1-100)
        seed: Random seed for reproducibility (optional)
        persist: Whether to persist traces to database (default: True)
    """

    count: int = Field(..., ge=1, le=100, description="Number of episodes to generate")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    persist: bool = Field(default=True, description="Whether to persist traces to database")


class UngarGenerateResponse(BaseModel):
    """Response from generating UNGAR traces.

    Attributes:
        trace_ids: List of created trace IDs (if persisted)
        preview: Preview of first few traces (metadata only)
    """

    trace_ids: list[uuid.UUID]
    preview: list[dict[str, Any]] = Field(
        ..., description="Preview of first 3 traces (metadata only)"
    )
