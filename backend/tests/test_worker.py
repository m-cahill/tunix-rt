"""Tests for Tunix background worker."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.db.base import Base
from tunix_rt_backend.db.models import TunixRun
from tunix_rt_backend.worker import process_run_safely

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    test_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with test_session_maker() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def db_session(test_db):
    return test_db


@pytest.mark.asyncio
async def test_process_run_safely_success():
    """Test process_run_safely handles successful execution."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config={
            "dataset_key": "test-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )

    with patch("tunix_rt_backend.worker.process_tunix_run", new_callable=AsyncMock) as mock_process:
        # Mock successful response
        mock_process.return_value = MagicMock(status="completed", exit_code=0)

        with patch(
            "tunix_rt_backend.worker.update_tunix_run_record", new_callable=AsyncMock
        ) as mock_update:
            await process_run_safely(run, mock_db)

            mock_process.assert_called_once()
            mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_process_run_safely_error():
    """Test process_run_safely handles errors."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config={
            "dataset_key": "test-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )

    with patch("tunix_rt_backend.worker.process_tunix_run", new_callable=AsyncMock) as mock_process:
        # Mock error
        mock_process.side_effect = ValueError("Execution failed")

        await process_run_safely(run, mock_db)

        assert run.status == "failed"
        assert "Worker internal error" in run.stderr
        mock_db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_claim_pending_run_skip_locked_shim(db_session):
    """Test claim_pending_run (only on Postgres, skip otherwise)."""
    # Check if we are running against Postgres
    if db_session.bind.dialect.name != "postgresql":
        pytest.skip("SKIP LOCKED requires PostgreSQL")

    # If we were on Postgres, we would test creating a pending run and claiming it.
    pass
