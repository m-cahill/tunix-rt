"""Tunix integration schemas.

This module defines request/response schemas for Tunix integration endpoints.
M12 provides artifact generation (JSONL export + YAML manifests) without
requiring Tunix runtime to be installed.
"""

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
