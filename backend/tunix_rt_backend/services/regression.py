"""Regression testing service (M18, M35).

Handles baselines and regression checks.
M35 additions: primary_score default, eval_set/dataset_key scoping, promote best run.
"""

import logging
import uuid
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import RegressionBaseline
from tunix_rt_backend.schemas.regression import (
    RegressionBaselineCreate,
    RegressionBaselineResponse,
    RegressionCheckResult,
)
from tunix_rt_backend.services.evaluation import EvaluationService

logger = logging.getLogger(__name__)


# M35: Default metric for regression checks
DEFAULT_REGRESSION_METRIC = "primary_score"


class RegressionService:
    """Service for managing regression baselines and checks."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_baseline(
        self, request: RegressionBaselineCreate
    ) -> RegressionBaselineResponse:
        """Create or update a named baseline."""
        # Check if run exists and has evaluation
        # We need evaluation to ensure the metric exists?
        # The prompt says "Baselines are policy... One named baseline per metric".
        # But our schema is (name -> run_id, metric). And name is unique.
        # So a baseline name implies a specific metric.

        # Verify run has evaluation
        service = EvaluationService(self.db)
        evaluation = await service.get_evaluation(request.run_id)
        if not evaluation:
            raise ValueError(f"Run {request.run_id} has no evaluation")

        # Check if metric exists in evaluation
        if request.metric == "score":
            pass  # Always exists
        elif request.metric in evaluation.metrics:
            pass
        else:
            # Check detailed metrics
            found = False
            for m in evaluation.detailed_metrics:
                if m.name == request.metric:
                    found = True
                    break
            if not found:
                raise ValueError(f"Metric '{request.metric}' not found in run evaluation")

        # Upsert baseline (delete existing with same name if any, then insert)
        # Or just try insert and handle duplicate.
        # Since we want to support updating baselines ("gemma-v1-initial" might be updated?),
        # let's check if it exists.
        stmt = select(RegressionBaseline).where(RegressionBaseline.name == request.name)
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.run_id = request.run_id
            existing.metric = request.metric
            existing.lower_is_better = request.lower_is_better
            # existing.created_at = datetime.now(...) # Optional: update timestamp
            await self.db.commit()
            await self.db.refresh(existing)
            db_baseline = existing
        else:
            db_baseline = RegressionBaseline(
                name=request.name,
                run_id=request.run_id,
                metric=request.metric,
                lower_is_better=request.lower_is_better,
            )
            self.db.add(db_baseline)
            await self.db.commit()
            await self.db.refresh(db_baseline)

        return RegressionBaselineResponse(
            id=db_baseline.id,
            name=db_baseline.name,
            run_id=db_baseline.run_id,
            metric=db_baseline.metric,
            lower_is_better=db_baseline.lower_is_better,
            created_at=db_baseline.created_at,
        )

    async def check_regression(
        self, run_id: uuid.UUID, baseline_name: str
    ) -> RegressionCheckResult:
        """Check if a run has regressed against a baseline."""
        # 1. Fetch Baseline
        stmt = select(RegressionBaseline).where(RegressionBaseline.name == baseline_name)
        result = await self.db.execute(stmt)
        baseline = result.scalar_one_or_none()
        if not baseline:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        # 2. Fetch Run Evaluation
        service = EvaluationService(self.db)
        current_eval = await service.get_evaluation(run_id)
        if not current_eval:
            raise ValueError(f"Run {run_id} not evaluated")

        # 3. Fetch Baseline Evaluation
        baseline_eval = await service.get_evaluation(baseline.run_id)
        if not baseline_eval:
            # Should not happen if data integrity is maintained
            raise ValueError(f"Baseline run {baseline.run_id} evaluation not found")

        # 4. Extract Metric Values
        current_val = self._get_metric_value(current_eval, baseline.metric)
        baseline_val = self._get_metric_value(baseline_eval, baseline.metric)

        # 5. Compare
        # Use configured direction, fallback to heuristic
        if baseline.lower_is_better is not None:
            lower_is_better = baseline.lower_is_better
        else:
            # Heuristic for direction (fallback if not configured)
            lower_is_better = any(
                x in baseline.metric.lower() for x in ["loss", "latency", "time", "error"]
            )

        delta = current_val - baseline_val

        # Calculate percent change
        if baseline_val != 0:
            delta_percent = (delta / baseline_val) * 100.0
        else:
            delta_percent = 0.0 if delta == 0 else (100.0 if delta > 0 else -100.0)

        # Threshold logic
        # 5% degradation tolerance
        tolerance = 5.0

        verdict: Literal["pass", "fail"] = "pass"
        threshold_desc = ""

        if lower_is_better:
            # If lower is better, current > baseline + 5% is bad
            # delta_percent > 5.0 is bad
            if delta_percent > tolerance:
                verdict = "fail"
                threshold_desc = f"increase > {tolerance}%"
            else:
                threshold_desc = f"increase <= {tolerance}%"
        else:
            # If higher is better, current < baseline - 5% is bad
            # delta_percent < -5.0 is bad
            if delta_percent < -tolerance:
                verdict = "fail"
                threshold_desc = f"drop > {tolerance}%"
            else:
                threshold_desc = f"drop <= {tolerance}%"

        return RegressionCheckResult(
            run_id=run_id,
            baseline_name=baseline_name,
            baseline_run_id=baseline.run_id,
            metric_name=baseline.metric,
            baseline_value=baseline_val,
            current_value=current_val,
            delta=delta,
            delta_percent=round(delta_percent, 2),
            verdict=verdict,
            details=f"Threshold: {threshold_desc}. Actual change: {delta_percent:.2f}%",
        )

    def _get_metric_value(self, evaluation: Any, metric_name: str) -> float:
        """Extract metric value from evaluation response.

        M35: Added support for 'primary_score' as a first-class metric.

        Args:
            evaluation: EvaluationResponse object
            metric_name: Name of metric to extract

        Returns:
            Metric value as float

        Raises:
            ValueError: If metric not found
        """
        # M35: Primary score is the canonical metric
        if metric_name == "primary_score":
            if evaluation.primary_score is not None:
                return float(evaluation.primary_score)
            # Fallback to normalized score if primary_score is None
            return float(evaluation.score / 100.0)

        if metric_name == "score":
            return float(evaluation.score)

        # Check simplified metrics dict
        if metric_name in evaluation.metrics:
            return float(evaluation.metrics[metric_name])

        # Check detailed metrics
        for m in evaluation.detailed_metrics:
            if m.name == metric_name:
                return float(m.score)

        raise ValueError(f"Metric '{metric_name}' not found in evaluation")
