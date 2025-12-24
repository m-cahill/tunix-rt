"""Evaluation service (M17).

Handles running evaluations on Tunix runs.
"""

import hashlib
import logging
import uuid
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import TunixRun, TunixRunEvaluation
from tunix_rt_backend.schemas.evaluation import (
    EvaluationJudgeInfo,
    EvaluationMetric,
    EvaluationResponse,
    LeaderboardItem,
    LeaderboardResponse,
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating Tunix runs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_evaluation(self, run_id: uuid.UUID) -> EvaluationResponse | None:
        """Get existing evaluation for a run."""
        stmt = (
            select(TunixRunEvaluation)
            .where(TunixRunEvaluation.run_id == run_id)
            .order_by(TunixRunEvaluation.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        evaluation = result.scalar_one_or_none()

        if not evaluation:
            return None

        # Reconstruct response from DB model
        # We assume database integrity for 'verdict'
        verdict: Literal["pass", "fail", "uncertain"] = evaluation.verdict  # type: ignore[assignment]

        return EvaluationResponse(
            evaluation_id=evaluation.id,
            run_id=evaluation.run_id,
            score=evaluation.score,
            verdict=verdict,
            judge=EvaluationJudgeInfo(name=evaluation.judge_name, version=evaluation.judge_version),
            metrics=evaluation.details.get("metrics", {}),
            detailed_metrics=[
                EvaluationMetric(**m) for m in evaluation.details.get("detailed_metrics", [])
            ],
            evaluated_at=evaluation.created_at.isoformat(),
        )

    async def get_leaderboard(self) -> LeaderboardResponse:
        """Get leaderboard data (M17)."""
        # Join evaluations with runs to get model_id/dataset_key
        # Get latest evaluation for each run?
        # For M17, assuming simple case: filter by latest created_at per run or just all
        # evaluations?
        # A run usually has one evaluation. If multiple, we might see duplicates.
        # Let's fetch all evaluations for now.

        stmt = (
            select(TunixRunEvaluation, TunixRun)
            .join(TunixRun, TunixRunEvaluation.run_id == TunixRun.run_id)
            .order_by(TunixRunEvaluation.score.desc())
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        items = []
        for evaluation, run in rows:
            metrics = evaluation.details.get("metrics", {})
            items.append(
                LeaderboardItem(
                    run_id=str(run.run_id),
                    model_id=run.model_id,
                    dataset_key=run.dataset_key,
                    score=evaluation.score,
                    verdict=evaluation.verdict,
                    metrics=metrics,
                    evaluated_at=evaluation.created_at.isoformat(),
                )
            )

        return LeaderboardResponse(data=items)

    async def evaluate_run(
        self, run_id: uuid.UUID, judge_override: str | None = None
    ) -> EvaluationResponse:
        """Run evaluation for a specific run (M17 Mock Judge)."""
        # 1. Fetch Run
        run = await self.db.get(TunixRun, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        if run.mode == "dry-run":
            raise ValueError(f"Cannot evaluate dry-run {run_id}")

        if run.status != "completed":
            # If failed/timeout, cannot pass evaluation
            if run.status in ["failed", "timeout", "cancelled"]:
                # We will score it 0 for record keeping?
                # Or raise error? The prompt implies "Was this run any good?"
                # A failed run is 0.0 goodness.
                pass
            else:
                # If pending/running, cannot evaluate
                raise ValueError(f"Run {run_id} is in {run.status} state, cannot evaluate")

        # 2. Mock Judge Logic
        judge_name = "mock-judge"
        judge_version = "v1"

        if run.status != "completed":
            score = 0.0
            verdict: Literal["pass", "fail", "uncertain"] = "fail"
            metrics = {"accuracy": 0.0, "compliance": 0.0}
            detailed_metrics = [
                EvaluationMetric(
                    name="accuracy",
                    score=0.0,
                    max_score=1.0,
                    details={"reason": f"Run status: {run.status}"},
                ),
                EvaluationMetric(
                    name="compliance",
                    score=0.0,
                    max_score=1.0,
                    details={"reason": "Run did not complete"},
                ),
            ]
        else:
            # Deterministic pseudo-random score based on run_id
            run_hash = int(hashlib.sha256(str(run_id).encode()).hexdigest(), 16)
            base_score = 50.0 + (run_hash % 50)  # 50-99 range

            # Adjust based on duration
            if run.duration_seconds and run.duration_seconds < 1.0:
                base_score -= 10

            score = min(max(base_score, 0.0), 100.0)

            verdict = "pass" if score >= 70 else "fail"

            metrics = {"accuracy": round(score / 100.0, 2), "compliance": 1.0, "coherence": 0.85}

            detailed_metrics = [
                EvaluationMetric(
                    name="accuracy", score=metrics["accuracy"], max_score=1.0, details=None
                ),
                EvaluationMetric(
                    name="compliance", score=metrics["compliance"], max_score=1.0, details=None
                ),
                EvaluationMetric(
                    name="coherence", score=metrics["coherence"], max_score=1.0, details=None
                ),
                EvaluationMetric(
                    name="output_length",
                    score=len(run.stdout) if run.stdout else 0,
                    max_score=10000,
                    details={"unit": "chars"},
                ),
            ]

        # 3. Persist
        details = {
            "metrics": metrics,
            "detailed_metrics": [m.model_dump() for m in detailed_metrics],
            "raw_judge_output": "Mock judge execution successful.",
        }

        evaluation = TunixRunEvaluation(
            run_id=run_id,
            score=score,
            verdict=verdict,
            judge_name=judge_name,
            judge_version=judge_version,
            details=details,
        )

        self.db.add(evaluation)
        await self.db.commit()
        await self.db.refresh(evaluation)

        return EvaluationResponse(
            evaluation_id=evaluation.id,
            run_id=evaluation.run_id,
            score=evaluation.score,
            verdict=verdict,
            judge=EvaluationJudgeInfo(name=judge_name, version=judge_version),
            metrics=metrics,
            detailed_metrics=detailed_metrics,
            evaluated_at=evaluation.created_at.isoformat(),
        )
