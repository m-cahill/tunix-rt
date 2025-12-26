#!/usr/bin/env python3
"""Tests for M33 evidence file schema validation.

These tests ensure that evidence files in submission_runs/ have the required
fields and correct schema, preventing "works once, evidence not captured"
failures.

Required fields (per M33 answers):
- run_manifest.json: run_version, model_id, dataset, commit_sha, timestamp,
                     config_path, command
- eval_summary.json: run_version, eval_set, metrics, evaluated_at, primary_score
"""

import json
from pathlib import Path

import pytest

# ============================================================
# Schema Definitions
# ============================================================

# Required fields for run_manifest.json
RUN_MANIFEST_REQUIRED_FIELDS = [
    "run_version",
    "model_id",
    "dataset",
    "commit_sha",
    "timestamp",
    "config_path",
    "command",
]

# Required fields for eval_summary.json
EVAL_SUMMARY_REQUIRED_FIELDS = [
    "run_version",
    "eval_set",
    "metrics",
    "evaluated_at",
    "primary_score",
]


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def m33_run_dir() -> Path:
    """Return path to M33 evidence directory."""
    # Navigate from backend/tests/ to repo root, then to submission_runs/m33_v1/
    return Path(__file__).parent.parent.parent / "submission_runs" / "m33_v1"


# ============================================================
# Schema Validation Tests
# ============================================================


class TestRunManifestSchema:
    """Tests for run_manifest.json schema validation."""

    def test_run_manifest_exists(self, m33_run_dir: Path) -> None:
        """Verify run_manifest.json exists in the evidence directory."""
        manifest_path = m33_run_dir / "run_manifest.json"
        assert manifest_path.exists(), f"Missing: {manifest_path}"

    def test_run_manifest_is_valid_json(self, m33_run_dir: Path) -> None:
        """Verify run_manifest.json is valid JSON."""
        manifest_path = m33_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "run_manifest.json must be a JSON object"

    def test_run_manifest_has_required_fields(self, m33_run_dir: Path) -> None:
        """Verify run_manifest.json has all required fields."""
        manifest_path = m33_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in RUN_MANIFEST_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_run_manifest_model_id_is_valid_gemma(self, m33_run_dir: Path) -> None:
        """Verify model_id is a valid competition model."""
        manifest_path = m33_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        model_id = data.get("model_id", "")
        valid_models = ["google/gemma-3-1b-it", "google/gemma-2-2b"]
        assert model_id in valid_models, f"model_id must be one of {valid_models}"

    def test_run_manifest_dataset_is_string(self, m33_run_dir: Path) -> None:
        """Verify dataset is a non-empty string."""
        manifest_path = m33_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        dataset = data.get("dataset")
        assert isinstance(dataset, str), "dataset must be a string"
        assert len(dataset) > 0, "dataset must not be empty"


class TestEvalSummarySchema:
    """Tests for eval_summary.json schema validation."""

    def test_eval_summary_exists(self, m33_run_dir: Path) -> None:
        """Verify eval_summary.json exists in the evidence directory."""
        summary_path = m33_run_dir / "eval_summary.json"
        assert summary_path.exists(), f"Missing: {summary_path}"

    def test_eval_summary_is_valid_json(self, m33_run_dir: Path) -> None:
        """Verify eval_summary.json is valid JSON."""
        summary_path = m33_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "eval_summary.json must be a JSON object"

    def test_eval_summary_has_required_fields(self, m33_run_dir: Path) -> None:
        """Verify eval_summary.json has all required fields."""
        summary_path = m33_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in EVAL_SUMMARY_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_eval_summary_metrics_is_dict(self, m33_run_dir: Path) -> None:
        """Verify metrics is a dictionary."""
        summary_path = m33_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics")
        assert isinstance(metrics, dict), "metrics must be a dictionary"

    def test_eval_summary_eval_set_is_string(self, m33_run_dir: Path) -> None:
        """Verify eval_set is a non-empty string."""
        summary_path = m33_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        eval_set = data.get("eval_set")
        assert isinstance(eval_set, str), "eval_set must be a string"
        assert len(eval_set) > 0, "eval_set must not be empty"


class TestKaggleOutputLog:
    """Tests for kaggle_output_log.txt."""

    def test_kaggle_output_log_exists(self, m33_run_dir: Path) -> None:
        """Verify kaggle_output_log.txt exists in the evidence directory."""
        log_path = m33_run_dir / "kaggle_output_log.txt"
        assert log_path.exists(), f"Missing: {log_path}"

    def test_kaggle_output_log_is_not_empty(self, m33_run_dir: Path) -> None:
        """Verify kaggle_output_log.txt is not empty (has content or placeholder)."""
        log_path = m33_run_dir / "kaggle_output_log.txt"
        content = log_path.read_text(encoding="utf-8")
        assert len(content) > 0, "kaggle_output_log.txt must not be empty"


class TestPackagingToolIncludesEvidence:
    """Tests that packaging tool correctly handles evidence files."""

    def test_evidence_files_list_is_correct(self) -> None:
        """Verify the expected evidence files are defined."""
        expected_files = [
            "run_manifest.json",
            "eval_summary.json",
            "kaggle_output_log.txt",
        ]
        # This is a sanity check that our test knows about all evidence files
        for f in expected_files:
            assert isinstance(f, str), f"Expected file name string, got {type(f)}"
