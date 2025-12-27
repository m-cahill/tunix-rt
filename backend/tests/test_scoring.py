#!/usr/bin/env python3
"""Tests for scoring module (M34).

These tests verify the compute_primary_score() function which aggregates
evaluation results into a single primary score for optimization and reporting.

Test scenarios:
- All rows have answer_correctness (ideal case)
- Mixed rows (some answer_correctness, some score fallback)
- Empty input
- Null/missing values handling
- Score normalization (0-100 to 0-1)
- Edge cases (already 0-1 scores, invalid values)
"""

import pytest

from tunix_rt_backend.scoring import compute_primary_score


class TestComputePrimaryScore:
    """Tests for compute_primary_score() function."""

    # ================================================================
    # Happy Path Tests
    # ================================================================

    def test_all_correct_returns_one(self) -> None:
        """All evaluations with answer_correctness=1.0 returns 1.0."""
        rows = [
            {"metrics": {"answer_correctness": 1.0}},
            {"metrics": {"answer_correctness": 1.0}},
            {"metrics": {"answer_correctness": 1.0}},
        ]
        result = compute_primary_score(rows)
        assert result == 1.0

    def test_all_incorrect_returns_zero(self) -> None:
        """All evaluations with answer_correctness=0.0 returns 0.0."""
        rows = [
            {"metrics": {"answer_correctness": 0.0}},
            {"metrics": {"answer_correctness": 0.0}},
        ]
        result = compute_primary_score(rows)
        assert result == 0.0

    def test_mixed_correctness_returns_mean(self) -> None:
        """Mixed answer_correctness values returns correct mean."""
        rows = [
            {"metrics": {"answer_correctness": 1.0}},  # Correct
            {"metrics": {"answer_correctness": 0.0}},  # Incorrect
            {"metrics": {"answer_correctness": 1.0}},  # Correct
        ]
        result = compute_primary_score(rows)
        # Mean of [1.0, 0.0, 1.0] = 2/3 ≈ 0.6667
        assert result is not None
        assert abs(result - (2 / 3)) < 0.0001

    def test_fractional_correctness(self) -> None:
        """Handles fractional answer_correctness values."""
        rows = [
            {"metrics": {"answer_correctness": 0.5}},
            {"metrics": {"answer_correctness": 0.8}},
        ]
        result = compute_primary_score(rows)
        # Mean of [0.5, 0.8] = 0.65
        assert result == pytest.approx(0.65)

    # ================================================================
    # Fallback to Score Tests
    # ================================================================

    def test_fallback_to_score_normalized(self) -> None:
        """Falls back to score/100 when answer_correctness missing."""
        rows = [
            {"score": 80},  # 80/100 = 0.8
            {"score": 60},  # 60/100 = 0.6
        ]
        result = compute_primary_score(rows)
        # Mean of [0.8, 0.6] = 0.7
        assert result == pytest.approx(0.7)

    def test_score_already_normalized(self) -> None:
        """Score already in 0-1 range is not double-normalized."""
        rows = [
            {"score": 0.9},  # Already 0-1
            {"score": 0.7},  # Already 0-1
        ]
        result = compute_primary_score(rows)
        # Mean of [0.9, 0.7] = 0.8
        assert result == pytest.approx(0.8)

    def test_mixed_answer_correctness_and_score(self) -> None:
        """Prefers answer_correctness when present, falls back to score."""
        rows = [
            {"metrics": {"answer_correctness": 1.0}},  # Uses answer_correctness
            {"score": 50},  # Fallback: 50/100 = 0.5
            {"metrics": {"answer_correctness": 0.8}},  # Uses answer_correctness
        ]
        result = compute_primary_score(rows)
        # Mean of [1.0, 0.5, 0.8] = 2.3/3 ≈ 0.7667
        assert result is not None
        assert abs(result - (2.3 / 3)) < 0.0001

    # ================================================================
    # Empty/Null Handling Tests
    # ================================================================

    def test_empty_list_returns_none(self) -> None:
        """Empty evaluation list returns None."""
        result = compute_primary_score([])
        assert result is None

    def test_all_null_correctness_returns_none(self) -> None:
        """All rows with null answer_correctness and no score returns None."""
        rows = [
            {"metrics": {"answer_correctness": None}},
            {"metrics": {}},
            {},
        ]
        result = compute_primary_score(rows)
        assert result is None

    def test_skips_null_values(self) -> None:
        """Null values are skipped, valid ones are averaged."""
        rows = [
            {"metrics": {"answer_correctness": 1.0}},
            {"metrics": {"answer_correctness": None}},  # Skipped
            {"metrics": {"answer_correctness": 0.5}},
        ]
        result = compute_primary_score(rows)
        # Mean of [1.0, 0.5] = 0.75
        assert result == pytest.approx(0.75)

    def test_skips_invalid_string_values(self) -> None:
        """Invalid string values are skipped."""
        rows = [
            {"metrics": {"answer_correctness": "invalid"}},
            {"metrics": {"answer_correctness": 0.8}},
        ]
        result = compute_primary_score(rows)
        assert result == pytest.approx(0.8)

    # ================================================================
    # Edge Case Tests
    # ================================================================

    def test_single_row(self) -> None:
        """Single row returns that row's score."""
        rows = [{"metrics": {"answer_correctness": 0.75}}]
        result = compute_primary_score(rows)
        assert result == 0.75

    def test_score_at_boundaries(self) -> None:
        """Handles boundary values correctly."""
        rows = [
            {"score": 0},  # 0/100 = 0.0
            {"score": 100},  # 100/100 = 1.0
        ]
        result = compute_primary_score(rows)
        # Mean of [0.0, 1.0] = 0.5
        assert result == pytest.approx(0.5)

    def test_out_of_range_scores_excluded(self) -> None:
        """Scores outside 0-1 range (after normalization) are excluded."""
        rows = [
            {"score": 150},  # 150/100 = 1.5 (out of range, excluded)
            {"score": -10},  # -10/100 = -0.1 (out of range, excluded)
            {"metrics": {"answer_correctness": 0.8}},  # Valid
        ]
        result = compute_primary_score(rows)
        # Only valid row is 0.8
        assert result == pytest.approx(0.8)

    def test_metrics_not_dict_handled(self) -> None:
        """Non-dict metrics field handled gracefully."""
        rows = [
            {"metrics": "not a dict"},
            {"metrics": {"answer_correctness": 0.6}},
        ]
        result = compute_primary_score(rows)
        assert result == pytest.approx(0.6)

    def test_stability_of_rounding(self) -> None:
        """Result is a plain float, not over-rounded."""
        rows = [
            {"metrics": {"answer_correctness": 0.333333}},
            {"metrics": {"answer_correctness": 0.666666}},
        ]
        result = compute_primary_score(rows)
        # Mean should preserve precision
        assert result is not None
        assert abs(result - 0.4999995) < 0.0001

    def test_answer_correctness_takes_priority_over_score(self) -> None:
        """When both answer_correctness and score present, prefer answer_correctness."""
        rows = [
            {"metrics": {"answer_correctness": 0.9}, "score": 50},  # Uses 0.9, not 0.5
        ]
        result = compute_primary_score(rows)
        assert result == pytest.approx(0.9)


class TestComputePrimaryScoreTypeSafety:
    """Type safety tests for compute_primary_score()."""

    def test_integer_answer_correctness(self) -> None:
        """Integer answer_correctness values are converted to float."""
        rows = [
            {"metrics": {"answer_correctness": 1}},  # int, not float
            {"metrics": {"answer_correctness": 0}},
        ]
        result = compute_primary_score(rows)
        assert result == pytest.approx(0.5)

    def test_integer_score(self) -> None:
        """Integer score values are converted to float."""
        rows = [
            {"score": 80},  # int
            {"score": 40},  # int
        ]
        result = compute_primary_score(rows)
        # Mean of [0.8, 0.4] = 0.6
        assert result == pytest.approx(0.6)
