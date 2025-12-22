"""Tests for training schema module."""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from tunix_rt_backend.training.schema import (
    EvaluationManifest,
    EvaluationResult,
    TrainingExample,
    TrainingExampleCreate,
    TrainingManifest,
)


class TestTrainingExample:
    """Tests for TrainingExample schema."""

    def test_training_example_valid(self):
        """Test creating a valid training example."""
        example = TrainingExample(
            prompt="What is 2+2?",
            response="The answer is 4",
            metadata={"source_trace_id": str(uuid.uuid4())},
        )

        assert isinstance(example.id, uuid.UUID)
        assert example.prompt == "What is 2+2?"
        assert example.response == "The answer is 4"
        assert "source_trace_id" in example.metadata

    def test_training_example_auto_id(self):
        """Test that ID is auto-generated if not provided."""
        example1 = TrainingExample(prompt="Test", response="Response")
        example2 = TrainingExample(prompt="Test", response="Response")

        assert isinstance(example1.id, uuid.UUID)
        assert isinstance(example2.id, uuid.UUID)
        assert example1.id != example2.id  # Should be unique

    def test_training_example_empty_prompt(self):
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingExample(prompt="", response="Valid response")

        errors = exc_info.value.errors()
        assert any(err["loc"] == ("prompt",) for err in errors)

    def test_training_example_empty_response(self):
        """Test that empty response is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingExample(prompt="Valid prompt", response="")

        errors = exc_info.value.errors()
        assert any(err["loc"] == ("response",) for err in errors)

    def test_training_example_metadata_optional(self):
        """Test that metadata is optional and defaults to empty dict."""
        example = TrainingExample(prompt="Test", response="Response")

        assert example.metadata == {}

    def test_training_example_serialization(self):
        """Test that training example can be serialized to dict."""
        example = TrainingExample(
            prompt="Test prompt",
            response="Test response",
            metadata={"key": "value"},
        )

        data = example.model_dump()

        assert data["prompt"] == "Test prompt"
        assert data["response"] == "Test response"
        assert data["metadata"] == {"key": "value"}
        assert "id" in data


class TestTrainingExampleCreate:
    """Tests for TrainingExampleCreate schema."""

    def test_create_valid(self):
        """Test creating a valid TrainingExampleCreate."""
        create = TrainingExampleCreate(
            prompt="What is the capital of France?",
            response="Paris",
            metadata={"difficulty": "easy"},
        )

        assert create.prompt == "What is the capital of France?"
        assert create.response == "Paris"
        assert create.metadata == {"difficulty": "easy"}

    def test_create_no_auto_id(self):
        """Test that TrainingExampleCreate does not auto-generate ID."""
        create = TrainingExampleCreate(prompt="Test", response="Response")

        data = create.model_dump()
        assert "id" not in data


class TestTrainingManifest:
    """Tests for TrainingManifest schema."""

    def test_manifest_valid(self):
        """Test creating a valid training manifest."""
        manifest = TrainingManifest(
            dataset_key="test-dataset-v1",
            training_config={"model": "gemma-2b", "lr": 1e-5},
            recipe="trace_sft_v1",
            seed=42,
            git_sha="abc123",
            artifacts_path="/path/to/artifacts",
        )

        assert isinstance(manifest.run_id, uuid.UUID)
        assert isinstance(manifest.created_at, datetime)
        assert manifest.dataset_key == "test-dataset-v1"
        assert manifest.recipe == "trace_sft_v1"
        assert manifest.seed == 42
        assert manifest.git_sha == "abc123"

    def test_manifest_auto_fields(self):
        """Test that run_id and created_at are auto-generated."""
        manifest1 = TrainingManifest(
            dataset_key="test-v1",
            training_config={},
            recipe="sft_v1",
            seed=42,
            artifacts_path="/tmp",
        )

        manifest2 = TrainingManifest(
            dataset_key="test-v1",
            training_config={},
            recipe="sft_v1",
            seed=42,
            artifacts_path="/tmp",
        )

        assert isinstance(manifest1.run_id, uuid.UUID)
        assert isinstance(manifest1.created_at, datetime)
        assert manifest1.run_id != manifest2.run_id  # Unique IDs

    def test_manifest_git_sha_optional(self):
        """Test that git_sha is optional."""
        manifest = TrainingManifest(
            dataset_key="test-v1",
            training_config={},
            recipe="sft_v1",
            seed=42,
            artifacts_path="/tmp",
        )

        assert manifest.git_sha is None

    def test_manifest_metadata_optional(self):
        """Test that metadata defaults to empty dict."""
        manifest = TrainingManifest(
            dataset_key="test-v1",
            training_config={},
            recipe="sft_v1",
            seed=42,
            artifacts_path="/tmp",
        )

        assert manifest.metadata == {}


class TestEvaluationResult:
    """Tests for EvaluationResult schema."""

    def test_result_valid(self):
        """Test creating a valid evaluation result."""
        result = EvaluationResult(
            example_id="eval-001",
            prompt="What is 5*5?",
            generated_response="25",
            expected_response="25",
            score=100.0,
            metadata={"latency_ms": 150},
        )

        assert result.example_id == "eval-001"
        assert result.prompt == "What is 5*5?"
        assert result.generated_response == "25"
        assert result.expected_response == "25"
        assert result.score == 100.0
        assert result.metadata["latency_ms"] == 150

    def test_result_optional_fields(self):
        """Test that expected_response and score are optional."""
        result = EvaluationResult(
            example_id="eval-002",
            prompt="Test",
            generated_response="Generated",
        )

        assert result.expected_response is None
        assert result.score is None
        assert result.metadata == {}


class TestEvaluationManifest:
    """Tests for EvaluationManifest schema."""

    def test_eval_manifest_valid(self):
        """Test creating a valid evaluation manifest."""
        run_id = uuid.uuid4()
        manifest = EvaluationManifest(
            run_id=run_id,
            model_checkpoint="/models/checkpoint-100",
            eval_set_path="/data/eval_v1.jsonl",
            results_count=25,
            avg_score=75.5,
            artifacts_path="/artifacts/eval",
        )

        assert isinstance(manifest.eval_id, uuid.UUID)
        assert manifest.run_id == run_id
        assert manifest.model_checkpoint == "/models/checkpoint-100"
        assert manifest.results_count == 25
        assert manifest.avg_score == 75.5

    def test_eval_manifest_auto_fields(self):
        """Test that eval_id and created_at are auto-generated."""
        manifest = EvaluationManifest(
            model_checkpoint="/models/base",
            eval_set_path="/data/eval.jsonl",
            results_count=10,
            artifacts_path="/artifacts",
        )

        assert isinstance(manifest.eval_id, uuid.UUID)
        assert isinstance(manifest.created_at, datetime)

    def test_eval_manifest_optional_fields(self):
        """Test that run_id and avg_score are optional."""
        manifest = EvaluationManifest(
            model_checkpoint="/models/base",
            eval_set_path="/data/eval.jsonl",
            results_count=10,
            artifacts_path="/artifacts",
        )

        assert manifest.run_id is None
        assert manifest.avg_score is None

    def test_eval_manifest_results_count_non_negative(self):
        """Test that results_count must be non-negative."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationManifest(
                model_checkpoint="/models/base",
                eval_set_path="/data/eval.jsonl",
                results_count=-5,  # Invalid: negative
                artifacts_path="/artifacts",
            )

        errors = exc_info.value.errors()
        assert any(err["loc"] == ("results_count",) for err in errors)
