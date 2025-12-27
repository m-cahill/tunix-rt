"""Tests for eval set validator (M35).

Tests the validate_evalset tool and eval_v2.jsonl schema compliance.
"""

import json

# Import from tools - add parent to path for tool imports
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from validate_evalset import (
    REQUIRED_FIELDS,
    VALID_DIFFICULTIES,
    VALID_SECTIONS,
    ValidationResult,
    validate_evalset,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def valid_evalset_content() -> str:
    """Valid eval set JSONL content."""
    items = [
        {"id": f"test-{i:03d}", "prompt": f"What is {i}?", "expected_answer": str(i)}
        for i in range(50)
    ]
    return "\n".join(json.dumps(item) for item in items)


@pytest.fixture
def valid_evalset_file(valid_evalset_content: str, tmp_path: Path) -> Path:
    """Create a temporary valid eval set file."""
    filepath = tmp_path / "test_eval.jsonl"
    filepath.write_text(valid_evalset_content)
    return filepath


# ============================================================
# ValidationResult Tests
# ============================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_state_is_valid(self) -> None:
        """New ValidationResult should be valid by default."""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_marks_invalid(self) -> None:
        """Adding an error should mark result as invalid."""
        result = ValidationResult()
        result.add_error("Test error")
        assert result.valid is False
        assert "Test error" in result.errors

    def test_add_warning_keeps_valid(self) -> None:
        """Adding a warning should not affect validity."""
        result = ValidationResult()
        result.add_warning("Test warning")
        assert result.valid is True
        assert "Test warning" in result.warnings


# ============================================================
# validate_evalset Function Tests
# ============================================================


class TestValidateEvalset:
    """Tests for validate_evalset function."""

    def test_valid_file_passes(self, valid_evalset_file: Path) -> None:
        """Valid eval set should pass validation."""
        result = validate_evalset(valid_evalset_file)
        assert result.valid is True
        assert result.stats["total_items"] == 50

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Missing file should fail validation."""
        result = validate_evalset(tmp_path / "nonexistent.jsonl")
        assert result.valid is False
        assert any("not found" in err for err in result.errors)

    def test_empty_file_fails(self, tmp_path: Path) -> None:
        """Empty file should fail validation."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")
        result = validate_evalset(filepath)
        assert result.valid is False
        assert any("No valid items" in err for err in result.errors)

    def test_invalid_json_fails(self, tmp_path: Path) -> None:
        """Invalid JSON line should be reported as error."""
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text('{"id": "1", prompt": "broken}\nnot json at all\n')
        result = validate_evalset(filepath)
        assert result.valid is False
        assert any("Invalid JSON" in err for err in result.errors)

    def test_missing_required_field_fails(self, tmp_path: Path) -> None:
        """Missing required field should fail."""
        items = [{"id": "test-001", "prompt": "Question?"}]  # Missing expected_answer
        filepath = tmp_path / "missing_field.jsonl"
        filepath.write_text(json.dumps(items[0]))
        result = validate_evalset(filepath, min_items=1)
        assert result.valid is False
        assert any("expected_answer" in err for err in result.errors)

    def test_duplicate_id_fails(self, tmp_path: Path) -> None:
        """Duplicate IDs should be reported as error."""
        items = [
            {"id": "dupe", "prompt": "Q1?", "expected_answer": "A1"},
            {"id": "dupe", "prompt": "Q2?", "expected_answer": "A2"},
        ]
        filepath = tmp_path / "dupe.jsonl"
        filepath.write_text("\n".join(json.dumps(item) for item in items))
        result = validate_evalset(filepath, min_items=1)
        assert result.valid is False
        assert any("Duplicate ID" in err for err in result.errors)

    def test_min_items_check(self, tmp_path: Path) -> None:
        """Insufficient items should fail."""
        items = [{"id": f"t{i}", "prompt": "Q?", "expected_answer": "A"} for i in range(10)]
        filepath = tmp_path / "few.jsonl"
        filepath.write_text("\n".join(json.dumps(item) for item in items))
        result = validate_evalset(filepath, min_items=50)
        assert result.valid is False
        assert any("Insufficient items" in err for err in result.errors)

    def test_unknown_section_warning(self, tmp_path: Path) -> None:
        """Unknown section value should generate warning (not error)."""
        item = {"id": "t1", "prompt": "Q?", "expected_answer": "A", "section": "invalid_section"}
        filepath = tmp_path / "bad_section.jsonl"
        filepath.write_text(json.dumps(item))
        result = validate_evalset(filepath, min_items=1)
        assert result.valid is True  # Warning, not error
        assert any("Unknown section" in warn for warn in result.warnings)

    def test_unknown_difficulty_warning(self, tmp_path: Path) -> None:
        """Unknown difficulty value should generate warning."""
        item = {"id": "t1", "prompt": "Q?", "expected_answer": "A", "difficulty": "impossible"}
        filepath = tmp_path / "bad_diff.jsonl"
        filepath.write_text(json.dumps(item))
        result = validate_evalset(filepath, min_items=1)
        assert result.valid is True  # Warning, not error
        assert any("Unknown difficulty" in warn for warn in result.warnings)

    def test_stats_include_counts(self, tmp_path: Path) -> None:
        """Statistics should include section/category/difficulty counts."""
        items = [
            {
                "id": "t1",
                "prompt": "Q1?",
                "expected_answer": "A1",
                "section": "core",
                "category": "math",
                "difficulty": "easy",
            },
            {
                "id": "t2",
                "prompt": "Q2?",
                "expected_answer": "A2",
                "section": "core",
                "category": "math",
                "difficulty": "medium",
            },
            {
                "id": "t3",
                "prompt": "Q3?",
                "expected_answer": "A3",
                "section": "edge_case",
                "category": "format",
                "difficulty": "easy",
            },
        ]
        filepath = tmp_path / "stats.jsonl"
        filepath.write_text("\n".join(json.dumps(item) for item in items))
        result = validate_evalset(filepath, min_items=1)

        assert result.stats["total_items"] == 3
        assert result.stats["sections"] == {"core": 2, "edge_case": 1}
        assert result.stats["categories"] == {"math": 2, "format": 1}
        assert result.stats["difficulties"] == {"easy": 2, "medium": 1}


# ============================================================
# eval_v2.jsonl Integration Tests
# ============================================================


class TestEvalV2Schema:
    """Integration tests for eval_v2.jsonl schema compliance."""

    @pytest.fixture
    def eval_v2_path(self) -> Path:
        """Path to the actual eval_v2.jsonl file."""
        # backend/tests/test_evalset_validator.py -> backend -> tunix-rt -> training/evalsets
        return Path(__file__).parent.parent.parent / "training" / "evalsets" / "eval_v2.jsonl"

    def test_eval_v2_exists(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should exist."""
        assert eval_v2_path.exists(), f"Expected eval_v2.jsonl at {eval_v2_path}"

    def test_eval_v2_is_valid(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should pass validation."""
        if not eval_v2_path.exists():
            pytest.skip("eval_v2.jsonl not found")
        result = validate_evalset(eval_v2_path, min_items=75)
        assert result.valid, f"Validation errors: {result.errors}"

    def test_eval_v2_has_minimum_items(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should have at least 75 items."""
        if not eval_v2_path.exists():
            pytest.skip("eval_v2.jsonl not found")
        result = validate_evalset(eval_v2_path, min_items=75)
        assert result.stats["total_items"] >= 75

    def test_eval_v2_has_all_sections(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should have items in all three sections."""
        if not eval_v2_path.exists():
            pytest.skip("eval_v2.jsonl not found")
        result = validate_evalset(eval_v2_path, min_items=1)
        sections = result.stats.get("sections", {})
        for expected_section in ["core", "trace_sensitive", "edge_case"]:
            assert expected_section in sections, f"Missing section: {expected_section}"
            assert sections[expected_section] > 0, f"Empty section: {expected_section}"

    def test_eval_v2_section_proportions(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should have roughly 60/25/15 section split."""
        if not eval_v2_path.exists():
            pytest.skip("eval_v2.jsonl not found")
        result = validate_evalset(eval_v2_path, min_items=1)
        percentages = result.stats.get("section_percentages", {})

        # Allow 5% tolerance
        assert 55 <= percentages.get("core", 0) <= 65, (
            f"core section out of range: {percentages.get('core')}%"
        )
        assert 20 <= percentages.get("trace_sensitive", 0) <= 30, "trace_sensitive out of range"
        assert 10 <= percentages.get("edge_case", 0) <= 20, "edge_case out of range"

    def test_eval_v2_has_difficulties(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should have a mix of difficulties."""
        if not eval_v2_path.exists():
            pytest.skip("eval_v2.jsonl not found")
        result = validate_evalset(eval_v2_path, min_items=1)
        difficulties = result.stats.get("difficulties", {})

        # Should have at least easy and medium
        assert "easy" in difficulties, "Missing easy difficulty items"
        assert "medium" in difficulties, "Missing medium difficulty items"

    def test_eval_v2_unique_ids(self, eval_v2_path: Path) -> None:
        """eval_v2.jsonl should have all unique IDs."""
        if not eval_v2_path.exists():
            pytest.skip("eval_v2.jsonl not found")
        result = validate_evalset(eval_v2_path, min_items=1)
        assert result.stats["total_items"] == result.stats["unique_ids"]


# ============================================================
# Schema Constants Tests
# ============================================================


class TestSchemaConstants:
    """Tests for schema constant definitions."""

    def test_required_fields_defined(self) -> None:
        """Required fields should be defined."""
        assert "id" in REQUIRED_FIELDS
        assert "prompt" in REQUIRED_FIELDS
        assert "expected_answer" in REQUIRED_FIELDS

    def test_valid_sections_defined(self) -> None:
        """Valid sections should include expected values."""
        assert "core" in VALID_SECTIONS
        assert "trace_sensitive" in VALID_SECTIONS
        assert "edge_case" in VALID_SECTIONS

    def test_valid_difficulties_defined(self) -> None:
        """Valid difficulties should include expected values."""
        assert "easy" in VALID_DIFFICULTIES
        assert "medium" in VALID_DIFFICULTIES
        assert "hard" in VALID_DIFFICULTIES
