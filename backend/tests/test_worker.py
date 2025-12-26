"""Tests for Tunix background worker.

Tests cover:
- process_run_safely: success, error, missing config, auto-evaluation paths
- claim_pending_run: Postgres-only (documented skip)

M32 update: Added edge case tests for process_run_safely.

Note on claim_pending_run:
    This function uses PostgreSQL-specific SKIP LOCKED semantics for atomic
    job claiming. It cannot be meaningfully unit tested on SQLite. Integration
    testing is covered by E2E tests running against actual Postgres.
"""

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
    """Test claim_pending_run (only on Postgres, skip otherwise).

    Note: claim_pending_run uses PostgreSQL's SKIP LOCKED for atomic job claiming.
    This feature is not available in SQLite, so we skip this test unless running
    against a real PostgreSQL instance.

    See: docs/M32_GUARDRAILS.md for rationale on Postgres-specific behavior.
    """
    # Check if we are running against Postgres
    if db_session.bind.dialect.name != "postgresql":
        pytest.skip("SKIP LOCKED requires PostgreSQL")

    # If we were on Postgres, we would test creating a pending run and claiming it.
    pass


# ============================================================
# Edge Case Tests for process_run_safely (M32)
# ============================================================


@pytest.mark.asyncio
async def test_process_run_safely_missing_config():
    """Test process_run_safely handles runs without config."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config=None,  # No config!
    )

    await process_run_safely(run, mock_db)

    # Should mark as failed
    assert run.status == "failed"
    assert "Worker internal error" in run.stderr
    assert run.completed_at is not None
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_process_run_safely_empty_config():
    """Test process_run_safely handles empty config dict."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config={},  # Empty config
    )

    await process_run_safely(run, mock_db)

    # Should fail due to missing required fields in TunixRunRequest
    assert run.status == "failed"
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_process_run_safely_with_duration_metrics():
    """Test process_run_safely records duration metrics."""
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
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.duration_seconds = 10.5
        mock_process.return_value = mock_response

        with patch("tunix_rt_backend.worker.update_tunix_run_record", new_callable=AsyncMock):
            with patch("tunix_rt_backend.worker.TUNIX_RUN_DURATION_SECONDS") as mock_duration:
                await process_run_safely(run, mock_db)

                # Duration metric should be observed
                mock_duration.labels.assert_called()


@pytest.mark.asyncio
async def test_process_run_safely_auto_eval_on_completion():
    """Test process_run_safely triggers auto-evaluation on successful non-dry-run."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config={
            "dataset_key": "test-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": False,  # NOT a dry run
        },
    )

    with patch("tunix_rt_backend.worker.process_tunix_run", new_callable=AsyncMock) as mock_process:
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.duration_seconds = None
        mock_process.return_value = mock_response

        with patch("tunix_rt_backend.worker.update_tunix_run_record", new_callable=AsyncMock):
            with patch("tunix_rt_backend.services.evaluation.EvaluationService") as mock_eval_class:
                mock_eval_instance = MagicMock()
                mock_eval_instance.evaluate_run = AsyncMock()
                mock_eval_class.return_value = mock_eval_instance

                await process_run_safely(run, mock_db)

                # Auto-evaluation should be triggered
                mock_eval_instance.evaluate_run.assert_called_once_with(run.run_id)


@pytest.mark.asyncio
async def test_process_run_safely_skips_eval_on_dry_run():
    """Test process_run_safely skips auto-evaluation on dry runs."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config={
            "dataset_key": "test-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": True,  # Dry run!
        },
    )

    with patch("tunix_rt_backend.worker.process_tunix_run", new_callable=AsyncMock) as mock_process:
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.duration_seconds = None
        mock_process.return_value = mock_response

        with patch("tunix_rt_backend.worker.update_tunix_run_record", new_callable=AsyncMock):
            with patch("tunix_rt_backend.services.evaluation.EvaluationService") as mock_eval_class:
                await process_run_safely(run, mock_db)

                # Auto-evaluation should NOT be called for dry runs
                mock_eval_class.assert_not_called()


@pytest.mark.asyncio
async def test_process_run_safely_eval_error_does_not_crash():
    """Test process_run_safely handles evaluation errors gracefully."""
    mock_db = AsyncMock()
    run = TunixRun(
        run_id=uuid.uuid4(),
        status="running",
        started_at=datetime.now(timezone.utc),
        config={
            "dataset_key": "test-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": False,
        },
    )

    with patch("tunix_rt_backend.worker.process_tunix_run", new_callable=AsyncMock) as mock_process:
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.duration_seconds = None
        mock_process.return_value = mock_response

        with patch("tunix_rt_backend.worker.update_tunix_run_record", new_callable=AsyncMock):
            with patch("tunix_rt_backend.services.evaluation.EvaluationService") as mock_eval_class:
                mock_eval_instance = MagicMock()
                mock_eval_instance.evaluate_run = AsyncMock(
                    side_effect=Exception("Evaluation failed!")
                )
                mock_eval_class.return_value = mock_eval_instance

                # Should not raise - evaluation errors are caught
                await process_run_safely(run, mock_db)

                # The run was still processed (evaluation failed separately)
                mock_eval_instance.evaluate_run.assert_called_once()


@pytest.mark.asyncio
async def test_process_run_safely_sets_completed_at_on_failure():
    """Test that completed_at is set when run fails."""
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
        mock_process.side_effect = RuntimeError("Something went wrong")

        before = datetime.now(timezone.utc)
        await process_run_safely(run, mock_db)
        after = datetime.now(timezone.utc)

        assert run.status == "failed"
        assert run.completed_at is not None
        assert before <= run.completed_at <= after
