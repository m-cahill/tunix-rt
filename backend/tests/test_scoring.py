#!/usr/bin/env python3
"""Tests for scoring module (M34/M35).

These tests verify:
- compute_primary_score(): Aggregates evaluation results into a single primary score
- compute_scorecard(): Generates detailed evaluation statistics (M35)

Test scenarios:
- All rows have answer_correctness (ideal case)
- Mixed rows (some answer_correctness, some score fallback)
- Empty input
- Null/missing values handling
- Score normalization (0-100 to 0-1)
- Edge cases (already 0-1 scores, invalid values)
- Scorecard statistics (stddev, per-section breakdowns)
"""

import pytest

from tunix_rt_backend.scoring import Scorecard, compute_primary_score, compute_scorecard


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


# ============================================================
# Scorecard Tests (M35)
# ============================================================


class TestScorecard:
    """Tests for Scorecard dataclass."""

    def test_default_values(self) -> None:
        """Scorecard has sensible defaults."""
        card = Scorecard()
        assert card.n_items == 0
        assert card.n_scored == 0
        assert card.n_skipped == 0
        assert card.primary_score is None
        assert card.stddev is None
        assert card.section_scores == {}

    def test_to_dict(self) -> None:
        """to_dict() returns all fields."""
        card = Scorecard(
            n_items=10,
            n_scored=8,
            n_skipped=2,
            primary_score=0.75,
            stddev=0.15,
            min_score=0.5,
            max_score=1.0,
            section_scores={"core": 0.8},
        )
        d = card.to_dict()
        assert d["n_items"] == 10
        assert d["n_scored"] == 8
        assert d["primary_score"] == 0.75
        assert d["section_scores"] == {"core": 0.8}


class TestComputeScorecard:
    """Tests for compute_scorecard() function."""

    # ================================================================
    # Basic Aggregation Tests
    # ================================================================

    def test_empty_list(self) -> None:
        """Empty evaluation list returns zero-count scorecard."""
        card = compute_scorecard([])
        assert card.n_items == 0
        assert card.n_scored == 0
        assert card.primary_score is None

    def test_single_row(self) -> None:
        """Single row scorecard."""
        rows = [{"item_id": "001", "metrics": {"answer_correctness": 0.8}}]
        card = compute_scorecard(rows)
        assert card.n_items == 1
        assert card.n_scored == 1
        assert card.n_skipped == 0
        assert card.primary_score == 0.8
        assert card.stddev is None  # Need 2+ items for stddev

    def test_multiple_rows(self) -> None:
        """Multiple rows with mean and stddev."""
        rows = [
            {"item_id": "001", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "metrics": {"answer_correctness": 0.5}},
            {"item_id": "003", "metrics": {"answer_correctness": 0.5}},
        ]
        card = compute_scorecard(rows)
        assert card.n_items == 3
        assert card.n_scored == 3
        assert card.primary_score == pytest.approx(2.0 / 3)
        assert card.stddev is not None
        assert card.min_score == 0.5
        assert card.max_score == 1.0

    def test_skipped_items(self) -> None:
        """Items without valid scores are skipped."""
        rows = [
            {"item_id": "001", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "metrics": {}},  # No score
            {"item_id": "003"},  # No metrics or score
        ]
        card = compute_scorecard(rows)
        assert card.n_items == 3
        assert card.n_scored == 1
        assert card.n_skipped == 2

    # ================================================================
    # Section Breakdown Tests
    # ================================================================

    def test_section_scores(self) -> None:
        """Computes per-section mean scores."""
        rows = [
            {"item_id": "001", "section": "core", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "section": "core", "metrics": {"answer_correctness": 0.8}},
            {"item_id": "003", "section": "edge_case", "metrics": {"answer_correctness": 0.5}},
        ]
        card = compute_scorecard(rows)
        assert card.section_scores["core"] == pytest.approx(0.9)
        assert card.section_scores["edge_case"] == pytest.approx(0.5)

    def test_category_scores(self) -> None:
        """Computes per-category mean scores."""
        rows = [
            {"item_id": "001", "category": "arithmetic", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "category": "arithmetic", "metrics": {"answer_correctness": 0.6}},
            {"item_id": "003", "category": "geometry", "metrics": {"answer_correctness": 0.8}},
        ]
        card = compute_scorecard(rows)
        assert card.category_scores["arithmetic"] == pytest.approx(0.8)
        assert card.category_scores["geometry"] == pytest.approx(0.8)

    def test_difficulty_scores(self) -> None:
        """Computes per-difficulty mean scores."""
        rows = [
            {"item_id": "001", "difficulty": "easy", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "difficulty": "easy", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "003", "difficulty": "hard", "metrics": {"answer_correctness": 0.5}},
        ]
        card = compute_scorecard(rows)
        assert card.difficulty_scores["easy"] == pytest.approx(1.0)
        assert card.difficulty_scores["hard"] == pytest.approx(0.5)

    # ================================================================
    # Metadata Lookup Tests
    # ================================================================

    def test_metadata_from_eval_items(self) -> None:
        """Looks up section/category from eval_items when missing in rows."""
        eval_items = [
            {"id": "001", "section": "core", "category": "math"},
            {"id": "002", "section": "edge_case", "category": "format"},
        ]
        rows = [
            {"item_id": "001", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "metrics": {"answer_correctness": 0.5}},
        ]
        card = compute_scorecard(rows, eval_items=eval_items)
        assert "core" in card.section_scores
        assert "edge_case" in card.section_scores
        assert card.category_scores.get("math") == pytest.approx(1.0)

    def test_row_metadata_takes_priority(self) -> None:
        """Row-level metadata takes priority over eval_items."""
        eval_items = [{"id": "001", "section": "core"}]
        rows = [{"item_id": "001", "section": "edge_case", "metrics": {"answer_correctness": 1.0}}]
        card = compute_scorecard(rows, eval_items=eval_items)
        # Row says edge_case, eval_items says core - row wins
        assert "edge_case" in card.section_scores
        assert "core" not in card.section_scores

    # ================================================================
    # Determinism Tests
    # ================================================================

    def test_deterministic_ordering(self) -> None:
        """Same inputs in different order produce same output."""
        rows_a = [
            {"item_id": "001", "metrics": {"answer_correctness": 1.0}},
            {"item_id": "002", "metrics": {"answer_correctness": 0.5}},
        ]
        rows_b = [
            {"item_id": "002", "metrics": {"answer_correctness": 0.5}},
            {"item_id": "001", "metrics": {"answer_correctness": 1.0}},
        ]
        card_a = compute_scorecard(rows_a)
        card_b = compute_scorecard(rows_b)
        assert card_a.primary_score == card_b.primary_score
        assert card_a.stddev == card_b.stddev

    def test_reproducible_stddev(self) -> None:
        """Standard deviation calculation is deterministic."""
        rows = [
            {"item_id": f"{i:03d}", "metrics": {"answer_correctness": i / 10}}
            for i in range(11)  # 0.0 to 1.0
        ]
        card1 = compute_scorecard(rows)
        card2 = compute_scorecard(rows)
        assert card1.stddev == card2.stddev

    # ================================================================
    # Score Extraction Tests
    # ================================================================

    def test_uses_correctness_field(self) -> None:
        """Supports 'correctness' field directly on row."""
        rows = [
            {"item_id": "001", "correctness": 1.0},
            {"item_id": "002", "correctness": 0.0},
        ]
        card = compute_scorecard(rows)
        assert card.n_scored == 2
        assert card.primary_score == pytest.approx(0.5)

    def test_score_fallback(self) -> None:
        """Falls back to score field with normalization."""
        rows = [
            {"item_id": "001", "score": 80},  # 0.8
            {"item_id": "002", "score": 60},  # 0.6
        ]
        card = compute_scorecard(rows)
        assert card.primary_score == pytest.approx(0.7)
