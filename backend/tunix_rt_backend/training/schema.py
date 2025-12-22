"""Training-specific schemas for Tunix SFT workflows.

This module defines TrainingExample, a derived abstraction from reasoning traces
that represents a single prompt/response training pair. TrainingExamples are
produced from traces/datasets at training time and never stored in the database.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """A single training example (prompt/response pair).

    This is a training-time abstraction derived from reasoning traces.
    It represents the fundamental unit of supervised fine-tuning data.

    Attributes:
        id: Unique identifier for this training example
        prompt: The input prompt (may include instructions + question)
        response: The expected model response (reasoning + answer)
        metadata: Optional metadata (source trace ID, recipe version, etc.)
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique training example ID")
    prompt: str = Field(..., min_length=1, description="Training prompt (input)")
    response: str = Field(..., min_length=1, description="Expected response (output)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata (source_trace_id, recipe, created_at, etc.)",
    )


class TrainingExampleCreate(BaseModel):
    """Schema for creating a training example (without auto-generated ID).

    Used when converting traces to training examples in batch.

    Attributes:
        prompt: The input prompt
        response: The expected model response
        metadata: Optional metadata
    """

    prompt: str = Field(..., min_length=1, description="Training prompt")
    response: str = Field(..., min_length=1, description="Expected response")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class TrainingManifest(BaseModel):
    """Manifest for a training run.

    Tracks all configuration and provenance for a training job,
    enabling reproducibility and auditing.

    Attributes:
        run_id: Unique identifier for this training run
        created_at: When the training run was created
        dataset_key: Which dataset was used for training
        training_config: Training configuration (model, hyperparameters, etc.)
        recipe: Training recipe identifier (e.g., "trace_sft_v1")
        seed: Random seed for reproducibility
        git_sha: Git commit SHA at training time
        artifacts_path: Path to training artifacts directory
    """

    run_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Training run ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Run creation time"
    )
    dataset_key: str = Field(..., description="Dataset used for training")
    training_config: dict[str, Any] = Field(
        ..., description="Training configuration (model, hyperparameters)"
    )
    recipe: str = Field(..., description="Training recipe identifier")
    seed: int = Field(..., description="Random seed for reproducibility")
    git_sha: str | None = Field(None, description="Git commit SHA at training time")
    artifacts_path: str = Field(..., description="Path to artifacts directory")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EvaluationResult(BaseModel):
    """Result from evaluating a model on a single example.

    Attributes:
        example_id: ID of the evaluation example (from eval set)
        prompt: The prompt that was evaluated
        generated_response: What the model actually generated
        expected_response: What the model should have generated (optional)
        score: Numeric score (if scored via baseline scorer)
        metadata: Additional metrics (latency, tokens, etc.)
    """

    example_id: str = Field(..., description="Evaluation example ID")
    prompt: str = Field(..., description="Prompt used for evaluation")
    generated_response: str = Field(..., description="Model's generated response")
    expected_response: str | None = Field(None, description="Expected response (if available)")
    score: float | None = Field(None, description="Numeric score (0-100)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")


class EvaluationManifest(BaseModel):
    """Manifest for an evaluation run.

    Tracks configuration and results for a model evaluation session.

    Attributes:
        eval_id: Unique identifier for this eval run
        run_id: Associated training run ID (if evaluating a trained model)
        created_at: When the evaluation was performed
        model_checkpoint: Path to model checkpoint being evaluated
        eval_set_path: Path to evaluation set file
        results_count: Number of examples evaluated
        avg_score: Average score across all examples (if scored)
        artifacts_path: Path to evaluation artifacts
    """

    eval_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Evaluation run ID")
    run_id: uuid.UUID | None = Field(None, description="Associated training run ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Eval creation time"
    )
    model_checkpoint: str = Field(..., description="Model checkpoint path")
    eval_set_path: str = Field(..., description="Evaluation set file path")
    results_count: int = Field(..., ge=0, description="Number of examples evaluated")
    avg_score: float | None = Field(None, description="Average score (if applicable)")
    artifacts_path: str = Field(..., description="Path to eval artifacts directory")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
