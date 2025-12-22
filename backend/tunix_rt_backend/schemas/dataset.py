"""Dataset schemas for training-ready exports.

This module defines schemas for dataset manifests and build requests.
Datasets are collections of traces selected and exported for training purposes.
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# Export format types for dataset export endpoint
ExportFormat = Literal["trace", "tunix_sft", "training_example"]


class DatasetBuildRequest(BaseModel):
    """Request to build a new dataset from traces.

    Attributes:
        dataset_name: Human-readable name (e.g., 'ungar_hcd_baseline')
        dataset_version: Version string (e.g., 'v1', '1.0.0')
        filters: Filter criteria for trace selection
        limit: Maximum number of traces to include
        selection_strategy: How to select traces ('latest' or 'random')
        seed: Random seed for reproducible selection (required for 'random' strategy)
        session_id: Optional session identifier for multi-session workflows
        parent_dataset_id: Optional parent dataset for incremental datasets
        training_run_id: Optional training run identifier
    """

    dataset_name: str = Field(..., min_length=1, max_length=128, description="Dataset name")
    dataset_version: str = Field(..., min_length=1, max_length=64, description="Version string")
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Filter criteria (e.g., {'source': 'ungar'})"
    )
    limit: int = Field(default=100, ge=1, le=10000, description="Max traces to include")
    selection_strategy: Literal["latest", "random"] = Field(
        default="latest", description="Selection strategy"
    )
    seed: int | None = Field(default=None, description="Random seed (required for 'random')")
    session_id: str | None = Field(default=None, max_length=128, description="Session ID")
    parent_dataset_id: str | None = Field(
        default=None, max_length=256, description="Parent dataset key"
    )
    training_run_id: str | None = Field(default=None, max_length=128, description="Training run ID")


class DatasetManifest(BaseModel):
    """Dataset manifest stored on disk.

    Attributes:
        dataset_key: Unique identifier ({dataset_name}-{dataset_version})
        build_id: Unique build identifier (UUID)
        dataset_name: Human-readable name
        dataset_version: Version string
        dataset_schema_version: Schema version for dataset format
        created_at: Build timestamp
        filters: Filter criteria used
        selection_strategy: Selection strategy used
        seed: Random seed used (if applicable)
        trace_ids: List of trace UUIDs included
        trace_count: Number of traces
        stats: Statistical summary
        session_id: Optional session identifier
        parent_dataset_id: Optional parent dataset key
        training_run_id: Optional training run identifier
    """

    dataset_key: str = Field(..., description="Unique dataset key (name-version)")
    build_id: uuid.UUID = Field(..., description="Unique build ID")
    dataset_name: str = Field(..., description="Dataset name")
    dataset_version: str = Field(..., description="Version string")
    dataset_schema_version: str = Field(default="1.0", description="Dataset schema version")
    created_at: datetime = Field(..., description="Build timestamp")
    filters: dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
    selection_strategy: Literal["latest", "random"] = Field(..., description="Selection strategy")
    seed: int | None = Field(default=None, description="Random seed (if used)")
    trace_ids: list[uuid.UUID] = Field(..., description="Included trace IDs")
    trace_count: int = Field(..., ge=0, description="Number of traces")
    stats: dict[str, Any] = Field(default_factory=dict, description="Statistical summary")
    session_id: str | None = Field(default=None, description="Session ID")
    parent_dataset_id: str | None = Field(default=None, description="Parent dataset key")
    training_run_id: str | None = Field(default=None, description="Training run ID")


class DatasetBuildResponse(BaseModel):
    """Response from dataset build request.

    Attributes:
        dataset_key: Dataset identifier
        build_id: Build identifier
        trace_count: Number of traces included
        manifest_path: Path to manifest file
    """

    dataset_key: str
    build_id: uuid.UUID
    trace_count: int
    manifest_path: str
