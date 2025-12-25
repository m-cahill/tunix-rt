"""Tunix integration schemas.

This module defines request/response schemas for Tunix integration endpoints.
M12: Artifact generation (JSONL export + YAML manifests) without runtime
M13: Optional runtime execution (dry-run + local modes)
"""

from typing import Literal

from pydantic import BaseModel, Field


class TunixStatusResponse(BaseModel):
    """Response from Tunix availability check.

    Attributes:
        available: Whether Tunix runtime is installed (M12: always False)
        version: Tunix version string if available (M12: always None)
        runtime_required: Whether runtime is needed for operations (M12: always False)
        message: Human-readable status message
    """

    available: bool = Field(..., description="Tunix runtime availability")
    version: str | None = Field(None, description="Tunix version (if installed)")
    runtime_required: bool = Field(
        ..., description="Whether Tunix runtime is required for operations"
    )
    message: str = Field(..., description="Status message")


class TunixExportRequest(BaseModel):
    """Request to export traces in Tunix-compatible format.

    Attributes:
        dataset_key: Dataset identifier (format: name-version, e.g., 'ungar_hcd-v1')
        trace_ids: Optional list of specific trace IDs to export
        limit: Maximum number of traces to export (if trace_ids not provided)
    """

    dataset_key: str | None = Field(
        None,
        min_length=1,
        max_length=256,
        description="Dataset key (name-version)",
    )
    trace_ids: list[str] | None = Field(
        None,
        description="Specific trace IDs to export (comma-separated UUIDs)",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Max traces to export",
    )


class TunixManifestRequest(BaseModel):
    """Request to generate a Tunix run manifest.

    Attributes:
        dataset_key: Dataset identifier for training data
        model_id: Model identifier (e.g., 'google/gemma-2b-it')
        output_dir: Output directory for training artifacts
        learning_rate: Learning rate (default: 2e-5)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size (default: 8)
        max_seq_length: Maximum sequence length (default: 2048)
    """

    dataset_key: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Dataset key (name-version)",
    )
    model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model identifier (e.g., 'google/gemma-2b-it')",
    )
    output_dir: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Output directory path",
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        le=1.0,
        description="Learning rate",
    )
    num_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of epochs",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=512,
        description="Batch size",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=32768,
        description="Maximum sequence length",
    )


class TunixManifestResponse(BaseModel):
    """Response from manifest generation.

    Attributes:
        manifest_yaml: Generated YAML manifest content
        dataset_key: Dataset identifier used
        model_id: Model identifier used
        format: Dataset format (always 'tunix_sft' in M12)
        message: Human-readable confirmation message
    """

    manifest_yaml: str = Field(..., description="YAML manifest content")
    dataset_key: str = Field(..., description="Dataset identifier")
    model_id: str = Field(..., description="Model identifier")
    format: str = Field(default="tunix_sft", description="Dataset export format")
    message: str = Field(..., description="Confirmation message")


# M13: Execution schemas

ExecutionMode = Literal["dry-run", "local"]
ExecutionStatus = Literal["pending", "running", "completed", "failed", "timeout"]


class TunixRunRequest(BaseModel):
    """Request to execute a Tunix training run (M13).

    Attributes:
        dataset_key: Dataset identifier for training data
        model_id: Model identifier (e.g., 'google/gemma-2b-it')
        output_dir: Output directory for training artifacts
            (optional, auto-generated if not provided)
        dry_run: If True, validate but don't execute (default: True)
        learning_rate: Learning rate (default: 2e-5)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size (default: 8)
        max_seq_length: Maximum sequence length (default: 2048)
    """

    dataset_key: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Dataset key (name-version)",
    )
    model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model identifier (e.g., 'google/gemma-2b-it')",
    )
    output_dir: str | None = Field(
        None,
        min_length=1,
        max_length=512,
        description="Output directory path (auto-generated if not provided)",
    )
    dry_run: bool = Field(
        default=True,
        description="Dry-run mode: validate without executing",
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        le=1.0,
        description="Learning rate",
    )
    num_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of epochs",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=512,
        description="Batch size",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=32768,
        description="Maximum sequence length",
    )


class TunixRunResponse(BaseModel):
    """Response from Tunix run execution (M13).

    Attributes:
        run_id: Unique identifier for this run
        status: Execution status
        mode: Execution mode used (dry-run or local)
        dataset_key: Dataset identifier
        model_id: Model identifier
        output_dir: Output directory path
        exit_code: Process exit code (None for dry-run)
        stdout: Standard output (truncated to 10KB)
        stderr: Standard error (truncated to 10KB)
        duration_seconds: Execution duration in seconds
        started_at: ISO-8601 timestamp when execution started
        completed_at: ISO-8601 timestamp when execution completed (None if still running)
        message: Human-readable status message
    """

    run_id: str = Field(..., description="Unique run identifier (UUID)")
    status: ExecutionStatus = Field(..., description="Execution status")
    mode: ExecutionMode = Field(..., description="Execution mode (dry-run or local)")
    dataset_key: str = Field(..., description="Dataset identifier")
    model_id: str = Field(..., description="Model identifier")
    output_dir: str = Field(..., description="Output directory path")
    exit_code: int | None = Field(None, description="Process exit code")
    stdout: str = Field(default="", description="Standard output (truncated)")
    stderr: str = Field(default="", description="Standard error (truncated)")
    duration_seconds: float | None = Field(None, description="Execution duration")
    started_at: str = Field(..., description="Start timestamp (ISO-8601)")
    completed_at: str | None = Field(None, description="Completion timestamp (ISO-8601)")
    message: str = Field(..., description="Status message")


class TunixRunStatusResponse(BaseModel):
    """Response from checking Tunix run status (M15).

    Attributes:
        run_id: Unique identifier for this run
        status: Execution status
        queued_at: ISO-8601 timestamp when run was queued (created_at)
        started_at: ISO-8601 timestamp when execution started
        completed_at: ISO-8601 timestamp when execution completed (None if still running)
        exit_code: Process exit code (None if still running or dry-run)
    """

    run_id: str = Field(..., description="Unique run identifier (UUID)")
    status: ExecutionStatus = Field(..., description="Execution status")
    queued_at: str = Field(..., description="Queued timestamp (ISO-8601)")
    started_at: str = Field(..., description="Start timestamp (ISO-8601)")
    completed_at: str | None = Field(None, description="Completion timestamp (ISO-8601)")
    exit_code: int | None = Field(None, description="Process exit code")


# M14: Run registry schemas


class TunixRunListItem(BaseModel):
    """List item for Tunix run history (M14).

    Attributes:
        run_id: Unique identifier for this run
        dataset_key: Dataset identifier
        model_id: Model identifier
        mode: Execution mode (dry-run or local)
        status: Execution status
        started_at: ISO-8601 timestamp when execution started
        duration_seconds: Execution duration in seconds (None if not completed)
        metrics: Run metrics (e.g. evaluation scores) (M24)
    """

    run_id: str = Field(..., description="Unique run identifier (UUID)")
    dataset_key: str = Field(..., description="Dataset identifier")
    model_id: str = Field(..., description="Model identifier")
    mode: ExecutionMode = Field(..., description="Execution mode")
    status: ExecutionStatus = Field(..., description="Execution status")
    started_at: str = Field(..., description="Start timestamp (ISO-8601)")
    duration_seconds: float | None = Field(None, description="Execution duration")
    metrics: dict[str, float | int | str | None] | None = Field(None, description="Run metrics")


class TunixRunListResponse(BaseModel):
    """Response from listing Tunix runs (M14).

    Attributes:
        data: List of run summary items
        pagination: Pagination metadata
    """

    data: list[TunixRunListItem] = Field(..., description="List of runs")
    pagination: dict[str, int | None] = Field(..., description="Pagination info")
