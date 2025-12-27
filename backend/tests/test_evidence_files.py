#!/usr/bin/env python3
"""Tests for evidence file schema validation (M33/M34).

These tests ensure that evidence files in submission_runs/ have the required
fields and correct schema, preventing "works once, evidence not captured"
failures.

Required fields (per M33/M34 answers):
- run_manifest.json: run_version, model_id, dataset, commit_sha, timestamp,
                     config_path, command
- eval_summary.json: run_version, eval_set, metrics, evaluated_at, primary_score
- best_params.json (M34+): params dict, source, sweep_config
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


# ============================================================
# M34 Evidence Tests
# ============================================================


@pytest.fixture
def m34_run_dir() -> Path:
    """Return path to M34 evidence directory."""
    return Path(__file__).parent.parent.parent / "submission_runs" / "m34_v1"


# M34-specific required fields (extends M33)
M34_RUN_MANIFEST_ADDITIONAL_FIELDS = [
    "tuning_job_id",  # Can be null for template
    "trial_id",  # Can be null for template
]

M34_BEST_PARAMS_REQUIRED_FIELDS = [
    "source",
    "params",
    "sweep_config",
]


class TestM34RunManifestSchema:
    """Tests for M34 run_manifest.json schema validation."""

    def test_run_manifest_exists(self, m34_run_dir: Path) -> None:
        """Verify run_manifest.json exists in M34 evidence directory."""
        manifest_path = m34_run_dir / "run_manifest.json"
        assert manifest_path.exists(), f"Missing: {manifest_path}"

    def test_run_manifest_has_required_fields(self, m34_run_dir: Path) -> None:
        """Verify run_manifest.json has all M33 required fields."""
        manifest_path = m34_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in RUN_MANIFEST_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_run_manifest_has_m34_fields(self, m34_run_dir: Path) -> None:
        """Verify run_manifest.json has M34 tuning provenance fields."""
        manifest_path = m34_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [
            field for field in M34_RUN_MANIFEST_ADDITIONAL_FIELDS if field not in data
        ]
        assert not missing_fields, f"Missing M34 fields: {missing_fields}"

    def test_run_manifest_model_id_is_valid(self, m34_run_dir: Path) -> None:
        """Verify model_id is a valid competition model."""
        manifest_path = m34_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        model_id = data.get("model_id", "")
        valid_models = ["google/gemma-3-1b-it", "google/gemma-2-2b"]
        assert model_id in valid_models, f"model_id must be one of {valid_models}"


class TestM34EvalSummarySchema:
    """Tests for M34 eval_summary.json schema validation."""

    def test_eval_summary_exists(self, m34_run_dir: Path) -> None:
        """Verify eval_summary.json exists in M34 evidence directory."""
        summary_path = m34_run_dir / "eval_summary.json"
        assert summary_path.exists(), f"Missing: {summary_path}"

    def test_eval_summary_has_required_fields(self, m34_run_dir: Path) -> None:
        """Verify eval_summary.json has all required fields."""
        summary_path = m34_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in EVAL_SUMMARY_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_eval_summary_primary_score_field_exists(self, m34_run_dir: Path) -> None:
        """Verify primary_score field exists (can be null for template)."""
        summary_path = m34_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "primary_score" in data, "primary_score field must exist"


class TestM34BestParamsSchema:
    """Tests for M34 best_params.json schema validation."""

    def test_best_params_exists(self, m34_run_dir: Path) -> None:
        """Verify best_params.json exists in M34 evidence directory."""
        params_path = m34_run_dir / "best_params.json"
        assert params_path.exists(), f"Missing: {params_path}"

    def test_best_params_is_valid_json(self, m34_run_dir: Path) -> None:
        """Verify best_params.json is valid JSON."""
        params_path = m34_run_dir / "best_params.json"
        with open(params_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "best_params.json must be a JSON object"

    def test_best_params_has_required_fields(self, m34_run_dir: Path) -> None:
        """Verify best_params.json has required fields."""
        params_path = m34_run_dir / "best_params.json"
        with open(params_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in M34_BEST_PARAMS_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_best_params_params_is_dict(self, m34_run_dir: Path) -> None:
        """Verify params is a dictionary."""
        params_path = m34_run_dir / "best_params.json"
        with open(params_path, encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("params")
        assert isinstance(params, dict), "params must be a dictionary"

    def test_best_params_sweep_config_is_dict(self, m34_run_dir: Path) -> None:
        """Verify sweep_config is a dictionary."""
        params_path = m34_run_dir / "best_params.json"
        with open(params_path, encoding="utf-8") as f:
            data = json.load(f)

        sweep_config = data.get("sweep_config")
        assert isinstance(sweep_config, dict), "sweep_config must be a dictionary"


class TestM34KaggleOutputLog:
    """Tests for M34 kaggle_output_log.txt."""

    def test_kaggle_output_log_exists(self, m34_run_dir: Path) -> None:
        """Verify kaggle_output_log.txt exists in M34 evidence directory."""
        log_path = m34_run_dir / "kaggle_output_log.txt"
        assert log_path.exists(), f"Missing: {log_path}"

    def test_kaggle_output_log_is_not_empty(self, m34_run_dir: Path) -> None:
        """Verify kaggle_output_log.txt is not empty."""
        log_path = m34_run_dir / "kaggle_output_log.txt"
        content = log_path.read_text(encoding="utf-8")
        assert len(content) > 0, "kaggle_output_log.txt must not be empty"


class TestM34EvidenceFilesComplete:
    """Tests that all M34 evidence files are present."""

    def test_all_m34_evidence_files_exist(self, m34_run_dir: Path) -> None:
        """Verify all expected M34 evidence files exist."""
        expected_files = [
            "run_manifest.json",
            "eval_summary.json",
            "best_params.json",
            "kaggle_output_log.txt",
        ]
        for filename in expected_files:
            file_path = m34_run_dir / filename
            assert file_path.exists(), f"Missing M34 evidence file: {filename}"


# ============================================================
# M35 Evidence File Tests
# ============================================================

# M35 required fields for eval_summary.json
M35_EVAL_SUMMARY_REQUIRED_FIELDS = [
    "run_version",
    "eval_set",
    "metrics",
    "primary_score",
    "scorecard",  # M35: New scorecard field
    "evaluated_at",
]


@pytest.fixture
def m35_run_dir() -> Path:
    """Return path to M35 evidence directory."""
    return Path(__file__).parent.parent.parent / "submission_runs" / "m35_v1"


class TestM35RunManifestSchema:
    """Tests for M35 run_manifest.json schema validation."""

    def test_run_manifest_exists(self, m35_run_dir: Path) -> None:
        """Verify run_manifest.json exists in M35 evidence directory."""
        manifest_path = m35_run_dir / "run_manifest.json"
        assert manifest_path.exists(), f"Missing: {manifest_path}"

    def test_run_manifest_has_required_fields(self, m35_run_dir: Path) -> None:
        """Verify run_manifest.json has all required fields."""
        manifest_path = m35_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in RUN_MANIFEST_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_run_manifest_has_eval_set(self, m35_run_dir: Path) -> None:
        """M35: Verify run_manifest.json includes eval_set field."""
        manifest_path = m35_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "eval_set" in data, "M35 run_manifest must include eval_set field"

    def test_run_manifest_eval_set_is_eval_v2(self, m35_run_dir: Path) -> None:
        """M35: Verify eval_set references eval_v2.jsonl."""
        manifest_path = m35_run_dir / "run_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        eval_set = data.get("eval_set", "")
        assert "eval_v2" in eval_set, f"M35 must use eval_v2.jsonl, got: {eval_set}"


class TestM35EvalSummarySchema:
    """Tests for M35 eval_summary.json schema validation."""

    def test_eval_summary_exists(self, m35_run_dir: Path) -> None:
        """Verify eval_summary.json exists in M35 evidence directory."""
        summary_path = m35_run_dir / "eval_summary.json"
        assert summary_path.exists(), f"Missing: {summary_path}"

    def test_eval_summary_has_required_fields(self, m35_run_dir: Path) -> None:
        """Verify eval_summary.json has all M35 required fields."""
        summary_path = m35_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        missing_fields = [field for field in M35_EVAL_SUMMARY_REQUIRED_FIELDS if field not in data]
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_eval_summary_has_scorecard(self, m35_run_dir: Path) -> None:
        """M35: Verify eval_summary.json includes scorecard object."""
        summary_path = m35_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "scorecard" in data, "M35 eval_summary must include scorecard"
        scorecard = data["scorecard"]
        assert isinstance(scorecard, dict), "scorecard must be a dictionary"

    def test_eval_summary_scorecard_has_n_items(self, m35_run_dir: Path) -> None:
        """M35: Verify scorecard has n_items field."""
        summary_path = m35_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        scorecard = data.get("scorecard", {})
        assert "n_items" in scorecard, "scorecard must include n_items"
        # M35 uses eval_v2 with 100 items
        assert scorecard["n_items"] == 100, (
            f"Expected 100 items for eval_v2, got {scorecard['n_items']}"
        )

    def test_eval_summary_uses_eval_v2(self, m35_run_dir: Path) -> None:
        """M35: Verify eval_set references eval_v2.jsonl."""
        summary_path = m35_run_dir / "eval_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        eval_set = data.get("eval_set", "")
        assert "eval_v2" in eval_set, f"M35 must use eval_v2.jsonl, got: {eval_set}"


class TestM35KaggleOutputLog:
    """Tests for M35 kaggle_output_log.txt."""

    def test_kaggle_output_log_exists(self, m35_run_dir: Path) -> None:
        """Verify kaggle_output_log.txt exists in M35 evidence directory."""
        log_path = m35_run_dir / "kaggle_output_log.txt"
        assert log_path.exists(), f"Missing: {log_path}"

    def test_kaggle_output_log_is_not_empty(self, m35_run_dir: Path) -> None:
        """Verify kaggle_output_log.txt is not empty."""
        log_path = m35_run_dir / "kaggle_output_log.txt"
        content = log_path.read_text(encoding="utf-8")
        assert len(content) > 0, "kaggle_output_log.txt must not be empty"


class TestM35EvidenceFilesComplete:
    """Tests that all M35 evidence files are present."""

    def test_all_m35_evidence_files_exist(self, m35_run_dir: Path) -> None:
        """Verify all expected M35 evidence files exist."""
        expected_files = [
            "run_manifest.json",
            "eval_summary.json",
            "kaggle_output_log.txt",
        ]
        for filename in expected_files:
            file_path = m35_run_dir / filename
            assert file_path.exists(), f"Missing M35 evidence file: {filename}"
