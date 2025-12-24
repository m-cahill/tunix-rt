"""Tests for evaluation service and endpoints (M17)."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import TunixRun
from tunix_rt_backend.services.evaluation import EvaluationService


@pytest.mark.asyncio
async def test_evaluate_run_completed_success(test_db: AsyncSession):
    """Test evaluating a completed run."""
    run_id = uuid.uuid4()
    run = TunixRun(
        run_id=run_id,
        dataset_key="test-ds",
        model_id="test-model",
        mode="local",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_seconds=10.0,
        stdout="Test output",
        exit_code=0,
    )
    test_db.add(run)
    await test_db.commit()

    service = EvaluationService(test_db)
    response = await service.evaluate_run(run_id)

    assert response.run_id == run_id
    assert response.verdict in ["pass", "fail"]
    assert response.judge.name == "mock-judge"
    assert "accuracy" in response.metrics
    assert "output_length" in [m.name for m in response.detailed_metrics]


@pytest.mark.asyncio
async def test_evaluate_run_pending_raises(test_db: AsyncSession):
    """Test that evaluating a pending run raises ValueError."""
    run_id = uuid.uuid4()
    run = TunixRun(
        run_id=run_id,
        dataset_key="test-ds",
        model_id="test-model",
        mode="local",
        status="pending",
        started_at=datetime.now(timezone.utc),
    )
    test_db.add(run)
    await test_db.commit()

    service = EvaluationService(test_db)
    with pytest.raises(ValueError, match="cannot evaluate"):
        await service.evaluate_run(run_id)


@pytest.mark.asyncio
async def test_evaluate_run_dry_run_raises(test_db: AsyncSession):
    """Test that evaluating a dry-run raises ValueError."""
    run_id = uuid.uuid4()
    run = TunixRun(
        run_id=run_id,
        dataset_key="test-ds",
        model_id="test-model",
        mode="dry-run",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        exit_code=0,
    )
    test_db.add(run)
    await test_db.commit()

    service = EvaluationService(test_db)
    with pytest.raises(ValueError, match="dry-run"):
        await service.evaluate_run(run_id)


@pytest.mark.asyncio
async def test_get_evaluation(test_db: AsyncSession):
    """Test retrieving an existing evaluation."""
    run_id = uuid.uuid4()
    run = TunixRun(
        run_id=run_id,
        dataset_key="test-ds",
        model_id="test-model",
        mode="local",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        exit_code=0,
    )
    test_db.add(run)
    await test_db.commit()

    service = EvaluationService(test_db)
    eval1 = await service.evaluate_run(run_id)

    # Retrieve
    eval2 = await service.get_evaluation(run_id)
    assert eval2 is not None
    assert eval2.evaluation_id == eval1.evaluation_id
    assert eval2.score == eval1.score


@pytest.mark.asyncio
async def test_get_leaderboard(test_db: AsyncSession):
    """Test leaderboard retrieval and sorting."""
    # Create 2 runs with deterministic IDs to control mock score
    # Run 1
    run1_id = uuid.uuid4()
    run1 = TunixRun(
        run_id=run1_id,
        dataset_key="ds1",
        model_id="m1",
        mode="local",
        status="completed",
        started_at=datetime.now(timezone.utc),
        exit_code=0,
        stdout="output1",
    )
    # Run 2
    run2_id = uuid.uuid4()
    run2 = TunixRun(
        run_id=run2_id,
        dataset_key="ds1",
        model_id="m2",
        mode="local",
        status="completed",
        started_at=datetime.now(timezone.utc),
        exit_code=0,
        stdout="output2",
    )
    test_db.add_all([run1, run2])
    await test_db.commit()

    service = EvaluationService(test_db)
    await service.evaluate_run(run1_id)
    await service.evaluate_run(run2_id)

    leaderboard = await service.get_leaderboard()
    assert len(leaderboard.data) == 2

    # Verify sorting (descending score)
    scores = [item.score for item in leaderboard.data]
    assert scores == sorted(scores, reverse=True)

    assert leaderboard.data[0].metrics is not None
