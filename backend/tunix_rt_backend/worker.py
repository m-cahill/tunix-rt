"""Tunix background worker (M15).

This worker processes enqueued Tunix runs (status='pending').
It claims runs atomically using SELECT ... FOR UPDATE SKIP LOCKED.

Guardrail:
> “Worker requires Postgres due to SKIP LOCKED semantics.”
"""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import async_session_maker
from tunix_rt_backend.db.models import TunixRun

# M15: Observability
from tunix_rt_backend.metrics import TUNIX_RUN_DURATION_SECONDS, TUNIX_RUNS_TOTAL
from tunix_rt_backend.schemas import TunixRunRequest
from tunix_rt_backend.services.tunix_execution import (
    process_tunix_run,
    update_tunix_run_record,
)

logger = logging.getLogger(__name__)


async def claim_pending_run(db: AsyncSession) -> TunixRun | None:
    """Claim a pending run atomically using SKIP LOCKED (Postgres only)."""
    # Subquery to find next pending run
    subquery = (
        select(TunixRun.run_id)
        .where(TunixRun.status == "pending")
        .order_by(TunixRun.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    ).scalar_subquery()

    # Update status to running
    stmt = (
        update(TunixRun)
        .where(TunixRun.run_id == subquery)
        .values(status="running", started_at=datetime.now(timezone.utc))
        .returning(TunixRun)
    )

    result = await db.execute(stmt)
    await db.commit()
    return result.scalar_one_or_none()


async def process_run_safely(run: TunixRun, db: AsyncSession) -> None:
    """Process a single run with error handling."""
    try:
        if not run.config:
            raise ValueError("Run has no config")

        request = TunixRunRequest(**run.config)

        # Generate output_dir if not provided (deterministic based on run_id)
        output_dir = request.output_dir or f"./output/tunix_run_{str(run.run_id)[:8]}"

        # Process run
        response = await process_tunix_run(
            run_id=str(run.run_id),
            request=request,
            output_dir=output_dir,
            started_at=run.started_at.isoformat(),
            db=db,
        )

        # Update DB
        await update_tunix_run_record(db, run, response)
        logger.info(f"Completed run {run.run_id} with status {response.status}")

        # Metrics
        TUNIX_RUNS_TOTAL.labels(
            status=response.status, mode=request.dry_run and "dry-run" or "local"
        ).inc()
        if response.duration_seconds is not None:
            TUNIX_RUN_DURATION_SECONDS.labels(
                mode=request.dry_run and "dry-run" or "local", status=response.status
            ).observe(response.duration_seconds)

        # M17: Auto-evaluate if completed (skip dry-run)
        if response.status == "completed" and not request.dry_run:
            try:
                from tunix_rt_backend.services.evaluation import EvaluationService

                await EvaluationService(db).evaluate_run(run.run_id)
                logger.info(f"Auto-evaluation completed for {run.run_id}")
            except Exception as e:
                logger.error(f"Auto-evaluation failed for {run.run_id}: {e}")

    except Exception as e:
        logger.error(f"Error processing run {run.run_id}: {e}")
        # Mark failed manually if execution logic crashed
        run.status = "failed"
        run.stderr = f"Worker internal error: {str(e)}"
        run.completed_at = datetime.now(timezone.utc)
        await db.commit()

        TUNIX_RUNS_TOTAL.labels(status="failed", mode="worker_error").inc()


async def worker_loop() -> None:
    """Main worker loop."""
    logger.info("Tunix worker started")
    while True:
        try:
            async with async_session_maker() as db:
                run = await claim_pending_run(db)
                if run:
                    logger.info(f"Claimed run {run.run_id}")
                    await process_run_safely(run, db)
                else:
                    # No pending runs, wait before polling
                    await asyncio.sleep(4)  # 4s polling interval
        except Exception as e:
            logger.error(f"Worker loop fatal error: {e}")
            await asyncio.sleep(4)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker stopped")
