"""Schema guardrail test: validate dev-reasoning-v2 dataset against ReasoningTrace schema.

This test ensures the dev-reasoning-v2 dataset (500+ traces) matches the strict
ReasoningTrace schema with steps: [{i, type, content}, ...].

M32 requirement: Schema validation for scaled dataset.
"""

import json
import random
from pathlib import Path

import pytest

from tunix_rt_backend.schemas import ReasoningTrace

# Path to dev-reasoning-v2 dataset
DATASET_DIR = Path(__file__).parent.parent / "datasets" / "dev-reasoning-v2"


class TestDevReasoningV2SchemaValidation:
    """Validate dev-reasoning-v2 dataset against ReasoningTrace schema."""

    def test_dataset_files_exist(self) -> None:
        """Ensure dataset and manifest files exist."""
        dataset_path = DATASET_DIR / "dataset.jsonl"
        manifest_path = DATASET_DIR / "manifest.json"

        assert dataset_path.exists(), f"Dataset not found: {dataset_path}"
        assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    def test_manifest_has_required_fields(self) -> None:
        """Validate manifest contains required metadata fields."""
        manifest_path = DATASET_DIR / "manifest.json"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Required fields
        assert manifest.get("dataset_key") == "dev-reasoning-v2"
        assert manifest.get("dataset_version") == "v2"
        assert manifest.get("trace_count") is not None
        assert manifest.get("trace_count") >= 500, "M32 requires 500+ traces"
        assert manifest.get("seed") == 42, "Dataset should be deterministic with seed=42"
        assert manifest.get("provenance") is not None

    def test_all_traces_validate_against_reasoning_trace_schema(self) -> None:
        """Every trace in dataset.jsonl must be a valid ReasoningTrace."""
        dataset_path = DATASET_DIR / "dataset.jsonl"

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) >= 500, f"Expected 500+ traces, found {len(lines)}"

        errors = []
        for i, line in enumerate(lines, start=1):
            try:
                data = json.loads(line)
                # Validate against ReasoningTrace schema
                trace = ReasoningTrace.model_validate(data)

                # Structural assertions
                assert trace.trace_version == "1.0", f"Line {i}: unexpected trace_version"
                assert trace.prompt, f"Line {i}: prompt must not be empty"
                assert trace.final_answer, f"Line {i}: final_answer must not be empty"
                assert len(trace.steps) >= 1, f"Line {i}: must have at least 1 step"

                # Validate step structure (strict schema: i, type, content)
                for step_idx, step in enumerate(trace.steps):
                    assert step.i >= 0, f"Line {i}, step {step_idx}: step.i must be >= 0"
                    assert step.type, f"Line {i}, step {step_idx}: step.type required"
                    assert step.content, f"Line {i}, step {step_idx}: step.content required"

            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"Line {i}: Validation failed - {e}")

        if errors:
            pytest.fail(f"{len(errors)} validation errors:\n" + "\n".join(errors[:10]))

    def test_sampled_traces_have_correct_metadata(self) -> None:
        """Sample 50 traces and verify metadata structure."""
        dataset_path = DATASET_DIR / "dataset.jsonl"

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Sample 50 traces deterministically
        random.seed(42)
        sample_indices = random.sample(range(len(lines)), min(50, len(lines)))

        valid_categories = {"reasoning", "synthetic", "golden_style", "edge_case"}

        for idx in sample_indices:
            data = json.loads(lines[idx])

            # Verify meta fields
            assert "meta" in data, f"Index {idx}: missing meta field"
            meta = data["meta"]

            assert meta.get("dataset") == "dev-reasoning-v2", f"Index {idx}: wrong dataset tag"
            assert meta.get("generator") == "seed_dev_reasoning_v2", f"Index {idx}: wrong generator"
            assert meta.get("seed") == 42, f"Index {idx}: wrong seed"
            assert meta.get("category") in valid_categories, f"Index {idx}: invalid category"

    def test_dataset_composition_matches_manifest(self) -> None:
        """Verify dataset composition matches manifest stats."""
        manifest_path = DATASET_DIR / "manifest.json"
        dataset_path = DATASET_DIR / "dataset.jsonl"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Count by category
        category_counts: dict[str, int] = {}
        for line in lines:
            data = json.loads(line)
            category = data.get("meta", {}).get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

        # Verify against manifest
        manifest_composition = manifest.get("stats", {}).get("composition", {})

        for category, expected_count in manifest_composition.items():
            actual_count = category_counts.get(category, 0)
            assert actual_count == expected_count, (
                f"Category '{category}': expected {expected_count}, got {actual_count}"
            )

    def test_step_indices_are_sequential(self) -> None:
        """Verify step indices are sequential starting from 0."""
        dataset_path = DATASET_DIR / "dataset.jsonl"

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Sample first 100 traces
        for i, line in enumerate(lines[:100]):
            data = json.loads(line)
            trace = ReasoningTrace.model_validate(data)

            step_indices = [step.i for step in trace.steps]
            expected_indices = list(range(len(trace.steps)))

            assert step_indices == expected_indices, (
                f"Line {i}: Step indices not sequential. "
                f"Got {step_indices}, expected {expected_indices}"
            )

    def test_no_duplicate_step_indices(self) -> None:
        """Verify no duplicate step indices within a trace."""
        dataset_path = DATASET_DIR / "dataset.jsonl"

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            data = json.loads(line)
            steps = data.get("steps", [])
            indices = [s.get("i") for s in steps]

            assert len(indices) == len(set(indices)), (
                f"Line {i}: Duplicate step indices found: {indices}"
            )

    def test_edge_cases_exist(self) -> None:
        """Verify edge case traces are present in the dataset."""
        dataset_path = DATASET_DIR / "dataset.jsonl"

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        edge_case_count = 0
        for line in lines:
            data = json.loads(line)
            if data.get("meta", {}).get("category") == "edge_case":
                edge_case_count += 1

        assert edge_case_count >= 10, f"Expected at least 10 edge cases, found {edge_case_count}"
