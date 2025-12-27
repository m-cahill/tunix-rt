"""Evaluation service (M17, M35).

Handles running evaluations on Tunix runs.
M35 additions: Leaderboard filtering, scorecard summary.
"""

import logging
import uuid
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import TunixRun, TunixRunEvaluation
from tunix_rt_backend.redi_client import MockRediClient, RediClientProtocol
from tunix_rt_backend.schemas import PaginationInfo
from tunix_rt_backend.schemas.evaluation import (
    EvaluationJudgeInfo,
    EvaluationMetric,
    EvaluationResponse,
    LeaderboardFilters,
    LeaderboardItem,
    LeaderboardResponse,
    ScorecardSummary,
)
from tunix_rt_backend.scoring import compute_primary_score
from tunix_rt_backend.services.judges import JudgeFactory

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating Tunix runs."""

    def __init__(self, db: AsyncSession, redi_client: RediClientProtocol | None = None):
        self.db = db
        # Default to MockRediClient if not provided (e.g. in tests)
        if redi_client is None:
            redi_client = MockRediClient()
        self.judge_factory = JudgeFactory(redi_client, db)

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

        # M34: Compute primary_score from metrics
        metrics = evaluation.details.get("metrics", {})
        primary_score_value = compute_primary_score(
            [{"metrics": metrics, "score": evaluation.score}]
        )

        return EvaluationResponse(
            evaluation_id=evaluation.id,
            run_id=evaluation.run_id,
            score=evaluation.score,
            verdict=verdict,
            judge=EvaluationJudgeInfo(name=evaluation.judge_name, version=evaluation.judge_version),
            metrics=metrics,
            detailed_metrics=[
                EvaluationMetric(**m) for m in evaluation.details.get("detailed_metrics", [])
            ],
            evaluated_at=evaluation.created_at.isoformat(),
            primary_score=primary_score_value,
        )

    async def get_leaderboard(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: LeaderboardFilters | None = None,
    ) -> LeaderboardResponse:
        """Get leaderboard data with pagination and filtering (M18, M35).

        Args:
            limit: Maximum results to return (1-100)
            offset: Results offset for pagination
            filters: Optional LeaderboardFilters with AND logic

        Returns:
            LeaderboardResponse with items, pagination, and applied filters
        """
        # Build base query: join evaluations with runs
        stmt = select(TunixRunEvaluation, TunixRun).join(
            TunixRun, TunixRunEvaluation.run_id == TunixRun.run_id
        )

        # M35: Apply filters (AND logic)
        if filters:
            # Dataset filter (exact match)
            if filters.dataset_key:
                stmt = stmt.where(TunixRun.dataset_key == filters.dataset_key)

            # Model filter (contains match)
            if filters.model_id:
                stmt = stmt.where(TunixRun.model_id.contains(filters.model_id))

            # Config filter (contains match on config JSON)
            if filters.config_path:
                # Config is stored as JSON, check if config_path is in the config dict
                # We filter on the string representation for simplicity
                stmt = stmt.where(TunixRun.config.cast(str).contains(filters.config_path))

            # Date range filter (on evaluation date)
            if filters.date_from:
                stmt = stmt.where(TunixRunEvaluation.created_at >= filters.date_from)
            if filters.date_to:
                stmt = stmt.where(TunixRunEvaluation.created_at <= filters.date_to)

        # Order by score (descending) and apply pagination
        stmt = stmt.order_by(TunixRunEvaluation.score.desc()).limit(limit + 1).offset(offset)

        result = await self.db.execute(stmt)
        rows = result.all()

        # Check for more results
        has_more = len(rows) > limit
        rows_to_return = rows[:limit]

        items = []
        for evaluation, run in rows_to_return:
            metrics = evaluation.details.get("metrics", {})
            detailed_metrics = evaluation.details.get("detailed_metrics", [])

            # M34: Compute primary_score for each leaderboard entry
            primary_score_value = compute_primary_score(
                [{"metrics": metrics, "score": evaluation.score}]
            )

            # M35: Build inline scorecard summary
            n_scored = len([m for m in detailed_metrics if m.get("score") is not None])
            scorecard = ScorecardSummary(
                n_items=len(detailed_metrics) if detailed_metrics else 1,
                n_scored=n_scored if n_scored > 0 else 1,
                n_skipped=0,
                primary_score=primary_score_value,
                stddev=None,  # Would require storing per-item scores
            )

            # M35: Extract config_path from run.config
            config_path = None
            if run.config and isinstance(run.config, dict):
                config_path = run.config.get("config_path") or run.config.get("config")

            items.append(
                LeaderboardItem(
                    run_id=str(run.run_id),
                    model_id=run.model_id,
                    dataset_key=run.dataset_key,
                    config_path=config_path,
                    score=evaluation.score,
                    verdict=evaluation.verdict,
                    metrics=metrics,
                    evaluated_at=evaluation.created_at.isoformat(),
                    primary_score=primary_score_value,
                    scorecard=scorecard,
                )
            )

        next_offset = offset + limit if has_more else None

        return LeaderboardResponse(
            data=items,
            pagination=PaginationInfo(limit=limit, offset=offset, next_offset=next_offset),
            filters=filters,
        )

    async def evaluate_run(
        self, run_id: uuid.UUID, judge_override: str | None = None
    ) -> EvaluationResponse:
        """Run evaluation for a specific run."""
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

        # 2. Judge Logic
        judge = self.judge_factory.get_judge(judge_override)
        result = await judge.evaluate(run)

        # 3. Persist
        details = {
            "metrics": result.metrics,
            "detailed_metrics": [m.model_dump() for m in result.detailed_metrics],
            "raw_judge_output": result.raw_output,
        }

        evaluation = TunixRunEvaluation(
            run_id=run_id,
            score=result.score,
            verdict=result.verdict,
            judge_name=result.judge_info.name,
            judge_version=result.judge_info.version,
            details=details,
        )

        self.db.add(evaluation)
        await self.db.commit()
        await self.db.refresh(evaluation)

        # M34: Compute primary_score from metrics
        primary_score_value = compute_primary_score(
            [{"metrics": result.metrics, "score": evaluation.score}]
        )

        return EvaluationResponse(
            evaluation_id=evaluation.id,
            run_id=evaluation.run_id,
            score=evaluation.score,
            verdict=result.verdict,
            judge=result.judge_info,
            metrics=result.metrics,
            detailed_metrics=result.detailed_metrics,
            evaluated_at=evaluation.created_at.isoformat(),
            primary_score=primary_score_value,
        )
