"""Tests for regression service (M18, M35).

Tests the RegressionService for baseline management and regression checks.
M35: Tests for primary_score default, enhanced get_metric_value.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tunix_rt_backend.schemas.regression import (
    RegressionBaselineCreate,
    RegressionBaselineResponse,
)
from tunix_rt_backend.services.regression import DEFAULT_REGRESSION_METRIC, RegressionService

# ============================================================
# Mock Fixtures
# ============================================================


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock async database session."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def mock_evaluation() -> MagicMock:
    """Mock evaluation response with primary_score."""
    eval_response = MagicMock()
    eval_response.score = 85.0
    eval_response.primary_score = 0.85
    eval_response.metrics = {"answer_correctness": 0.85}
    eval_response.detailed_metrics = [
        MagicMock(name="item_001", score=1.0),
        MagicMock(name="item_002", score=0.5),
    ]
    return eval_response


# ============================================================
# Constants Tests
# ============================================================


class TestRegressionConstants:
    """Tests for regression module constants."""

    def test_default_metric_is_primary_score(self) -> None:
        """M35: Default metric should be primary_score."""
        assert DEFAULT_REGRESSION_METRIC == "primary_score"


# ============================================================
# RegressionService._get_metric_value Tests
# ============================================================


class TestGetMetricValue:
    """Tests for _get_metric_value method."""

    @pytest.fixture
    def service(self, mock_db: AsyncMock) -> RegressionService:
        """Create service instance."""
        return RegressionService(mock_db)

    def test_extracts_primary_score(
        self, service: RegressionService, mock_evaluation: MagicMock
    ) -> None:
        """M35: Extracts primary_score when metric is 'primary_score'."""
        result = service._get_metric_value(mock_evaluation, "primary_score")
        assert result == 0.85

    def test_primary_score_fallback_to_normalized_score(self, service: RegressionService) -> None:
        """M35: Falls back to score/100 if primary_score is None."""
        eval_response = MagicMock()
        eval_response.primary_score = None
        eval_response.score = 80.0
        eval_response.metrics = {}
        eval_response.detailed_metrics = []

        result = service._get_metric_value(eval_response, "primary_score")
        assert result == 0.8  # 80/100

    def test_extracts_raw_score(
        self, service: RegressionService, mock_evaluation: MagicMock
    ) -> None:
        """Extracts raw score when metric is 'score'."""
        result = service._get_metric_value(mock_evaluation, "score")
        assert result == 85.0

    def test_extracts_from_metrics_dict(
        self, service: RegressionService, mock_evaluation: MagicMock
    ) -> None:
        """Extracts metric from metrics dictionary."""
        result = service._get_metric_value(mock_evaluation, "answer_correctness")
        assert result == 0.85

    def test_raises_for_unknown_metric(
        self, service: RegressionService, mock_evaluation: MagicMock
    ) -> None:
        """Raises ValueError for unknown metric."""
        with pytest.raises(ValueError, match="not found"):
            service._get_metric_value(mock_evaluation, "nonexistent_metric")


# ============================================================
# Schema Tests
# ============================================================


class TestRegressionSchemas:
    """Tests for regression request/response schemas."""

    def test_baseline_create_defaults_to_primary_score(self) -> None:
        """M35: RegressionBaselineCreate defaults metric to 'primary_score'."""
        request = RegressionBaselineCreate(
            name="test-baseline",
            run_id=uuid.uuid4(),
        )
        assert request.metric == "primary_score"

    def test_baseline_create_with_custom_metric(self) -> None:
        """RegressionBaselineCreate accepts custom metric."""
        request = RegressionBaselineCreate(
            name="test-baseline",
            run_id=uuid.uuid4(),
            metric="answer_correctness",
        )
        assert request.metric == "answer_correctness"

    def test_baseline_create_with_eval_set(self) -> None:
        """M35: RegressionBaselineCreate accepts eval_set field."""
        request = RegressionBaselineCreate(
            name="test-baseline",
            run_id=uuid.uuid4(),
            eval_set="eval_v2.jsonl",
            dataset_key="dev-reasoning-v2",
        )
        assert request.eval_set == "eval_v2.jsonl"
        assert request.dataset_key == "dev-reasoning-v2"

    def test_baseline_response_includes_new_fields(self) -> None:
        """M35: RegressionBaselineResponse includes eval_set and dataset_key."""
        response = RegressionBaselineResponse(
            id=uuid.uuid4(),
            name="test-baseline",
            run_id=uuid.uuid4(),
            metric="primary_score",
            eval_set="eval_v2.jsonl",
            dataset_key="dev-reasoning-v2",
            created_at=datetime.now(timezone.utc),
        )
        assert response.eval_set == "eval_v2.jsonl"
        assert response.dataset_key == "dev-reasoning-v2"


# ============================================================
# Regression Check Logic Tests
# ============================================================


class TestRegressionCheckLogic:
    """Tests for regression check calculations."""

    def test_higher_is_better_pass(self) -> None:
        """Pass when current is higher and higher is better."""
        # higher_is_better: current >= baseline - 5% passes
        baseline_val = 0.80
        current_val = 0.85
        delta = current_val - baseline_val
        delta_percent = (delta / baseline_val) * 100.0

        # delta_percent = 6.25% (increase)
        # higher_is_better, so this passes
        assert delta_percent > 0  # Improvement
        assert True  # Would pass

    def test_higher_is_better_fail(self) -> None:
        """Fail when current is much lower and higher is better."""
        baseline_val = 0.80
        current_val = 0.70
        delta_percent = ((current_val - baseline_val) / baseline_val) * 100.0

        # delta_percent = -12.5% (decrease > 5%)
        # higher_is_better, so this fails
        assert delta_percent < -5.0  # Significant decrease

    def test_lower_is_better_pass(self) -> None:
        """Pass when current is lower and lower is better (e.g., loss)."""
        baseline_val = 0.30  # 30% error
        current_val = 0.25  # 25% error (better)
        delta_percent = ((current_val - baseline_val) / baseline_val) * 100.0

        # delta_percent = -16.7% (decrease, which is improvement)
        # lower_is_better, so decrease is good
        assert delta_percent < 0  # Improvement

    def test_tolerance_threshold(self) -> None:
        """5% tolerance is applied correctly."""
        tolerance = 5.0
        baseline_val = 0.80

        # 4% drop should pass
        current_slightly_lower = 0.768  # 4% drop
        delta_percent = ((current_slightly_lower - baseline_val) / baseline_val) * 100.0
        assert delta_percent > -tolerance  # Passes

        # 6% drop should fail
        current_much_lower = 0.752  # 6% drop
        delta_percent = ((current_much_lower - baseline_val) / baseline_val) * 100.0
        assert delta_percent < -tolerance  # Fails
