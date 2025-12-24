"""Tests for RegressionService."""

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.db.base import Base
from tunix_rt_backend.db.models import RegressionBaseline, TunixRun, TunixRunEvaluation
from tunix_rt_backend.schemas.regression import RegressionBaselineCreate
from tunix_rt_backend.services.regression import RegressionService

# Test database URL (SQLite in-memory for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    test_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with test_session_maker() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def cleanup_baselines(db_session):
    yield
    await db_session.execute(delete(RegressionBaseline))
    await db_session.commit()


@pytest.mark.asyncio
async def test_create_baseline(db_session):
    """Test creating a regression baseline."""
    # 1. Setup Run + Evaluation
    run_id = uuid.uuid4()
    run = TunixRun(
        run_id=run_id,
        dataset_key="test",
        model_id="test",
        mode="local",
        status="completed",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run)

    evaluation = TunixRunEvaluation(
        run_id=run_id,
        score=85.0,
        verdict="pass",
        judge_name="mock",
        judge_version="v1",
        details={"metrics": {"accuracy": 0.85}, "detailed_metrics": []},
    )
    db_session.add(evaluation)
    await db_session.commit()

    # 2. Create Baseline
    service = RegressionService(db_session)
    request = RegressionBaselineCreate(name="test-baseline-v1", run_id=run_id, metric="score")

    response = await service.create_baseline(request)

    assert response.name == "test-baseline-v1"
    assert response.metric == "score"
    assert response.run_id == run_id


@pytest.mark.asyncio
async def test_check_regression_pass(db_session):
    """Test passing regression check."""
    # 1. Setup Baseline Run
    base_run_id = uuid.uuid4()
    db_session.add(
        TunixRun(
            run_id=base_run_id,
            dataset_key="k",
            model_id="m",
            mode="l",
            status="completed",
            started_at=datetime.now(timezone.utc),
        )
    )
    db_session.add(
        TunixRunEvaluation(
            run_id=base_run_id,
            score=80.0,
            verdict="pass",
            judge_name="mock",
            judge_version="v1",
            details={"metrics": {}, "detailed_metrics": []},
        )
    )

    # 2. Create Baseline
    db_session.add(RegressionBaseline(name="baseline-pass", run_id=base_run_id, metric="score"))

    # 3. Setup Current Run (Higher Score)
    curr_run_id = uuid.uuid4()
    db_session.add(
        TunixRun(
            run_id=curr_run_id,
            dataset_key="k",
            model_id="m",
            mode="l",
            status="completed",
            started_at=datetime.now(timezone.utc),
        )
    )
    db_session.add(
        TunixRunEvaluation(
            run_id=curr_run_id,
            score=85.0,
            verdict="pass",
            judge_name="mock",
            judge_version="v1",
            details={"metrics": {}, "detailed_metrics": []},
        )
    )
    await db_session.commit()

    # 4. Check
    service = RegressionService(db_session)
    result = await service.check_regression(curr_run_id, "baseline-pass")

    assert result.verdict == "pass"
    assert result.delta == 5.0
    assert result.baseline_value == 80.0
    assert result.current_value == 85.0


@pytest.mark.asyncio
async def test_check_regression_fail(db_session):
    """Test failing regression check."""
    # 1. Setup Baseline Run
    base_run_id = uuid.uuid4()
    db_session.add(
        TunixRun(
            run_id=base_run_id,
            dataset_key="k",
            model_id="m",
            mode="l",
            status="completed",
            started_at=datetime.now(timezone.utc),
        )
    )
    db_session.add(
        TunixRunEvaluation(
            run_id=base_run_id,
            score=90.0,
            verdict="pass",
            judge_name="mock",
            judge_version="v1",
            details={"metrics": {}, "detailed_metrics": []},
        )
    )

    # 2. Create Baseline
    db_session.add(RegressionBaseline(name="baseline-fail", run_id=base_run_id, metric="score"))

    # 3. Setup Current Run (Much Lower Score)
    curr_run_id = uuid.uuid4()
    db_session.add(
        TunixRun(
            run_id=curr_run_id,
            dataset_key="k",
            model_id="m",
            mode="l",
            status="completed",
            started_at=datetime.now(timezone.utc),
        )
    )
    db_session.add(
        TunixRunEvaluation(
            run_id=curr_run_id,
            score=70.0,
            verdict="pass",
            judge_name="mock",
            judge_version="v1",
            details={"metrics": {}, "detailed_metrics": []},
        )
    )
    await db_session.commit()

    # 4. Check
    service = RegressionService(db_session)
    result = await service.check_regression(curr_run_id, "baseline-fail")

    assert result.verdict == "fail"
    assert result.delta == -20.0
    # 20/90 * 100 = 22.2% drop
    assert result.delta_percent < -5.0
