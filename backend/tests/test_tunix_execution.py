"""Tests for Tunix execution integration (M13).

This module tests the Tunix runtime execution functionality:
- Default tests: Run without Tunix installed (dry-run, 501 responses)
- Optional tests: Run with Tunix installed (local execution)

All default tests pass without Tunix installed.
Optional tests are marked with @pytest.mark.tunix and skipped if Tunix unavailable.
"""

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db
from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.integrations.tunix.availability import tunix_available

# Test database URL (SQLite in-memory for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session.

    Yields:
        AsyncSession for testing
    """
    # Create async engine for tests
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session factory
    test_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Yield session
    async with test_session_maker() as session:
        yield session

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with database override.

    Args:
        test_db: Test database session

    Yields:
        AsyncClient for testing
    """

    # Override get_db dependency
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    # Clear overrides
    app.dependency_overrides.clear()


@pytest.fixture
def db_session(test_db: AsyncSession) -> AsyncSession:
    """Provide db_session fixture for async tests.

    Args:
        test_db: Test database session

    Returns:
        AsyncSession for testing
    """
    return test_db


# Default tests (no Tunix required)


def test_tunix_availability_checks_import_and_cli():
    """Test tunix_available() checks both import and CLI."""
    # M13: Should check both import and CLI
    # Without Tunix installed, should return False
    available = tunix_available()
    assert isinstance(available, bool)
    # M13 default: False (no Tunix installed in default CI)
    assert available is False


@pytest.mark.asyncio
async def test_tunix_run_endpoint_exists(client: AsyncClient):
    """Test POST /api/tunix/run endpoint exists."""
    # Send dry-run request (should work without Tunix)
    response = await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": "nonexistent-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )

    # Should NOT return 404 (endpoint exists)
    assert response.status_code != status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_dry_run_with_invalid_dataset(db_session):
    """Test dry-run mode with non-existent dataset."""
    from tunix_rt_backend.schemas import TunixRunRequest
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    request = TunixRunRequest(
        dataset_key="nonexistent-v1",
        model_id="google/gemma-2b-it",
        dry_run=True,
    )

    response = await execute_tunix_run(request, db_session)

    # Should fail with dataset not found
    assert response.status == "failed"
    assert response.mode == "dry-run"
    assert "Dataset not found" in response.stderr or "not found" in response.message.lower()
    assert response.exit_code is None


@pytest.mark.asyncio
async def test_dry_run_with_empty_dataset(db_session):
    """Test dry-run mode with empty dataset."""
    import uuid

    from tunix_rt_backend.helpers.datasets import save_manifest
    from tunix_rt_backend.schemas import DatasetManifest, TunixRunRequest
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    # Create empty dataset manifest using proper schema
    dataset_key = "empty_test-v1"
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        build_id=str(uuid.uuid4()),
        dataset_name="empty_test",
        dataset_version="v1",
        dataset_schema_version="1.0",
        created_at="2025-12-22T00:00:00Z",
        filters={},
        selection_strategy="latest",
        seed=None,
        trace_ids=[],
        trace_count=0,
        stats={},
        session_id=None,
        parent_dataset_id=None,
        training_run_id=None,
    )

    manifest_path = save_manifest(manifest)

    try:
        request = TunixRunRequest(
            dataset_key=dataset_key,
            model_id="google/gemma-2b-it",
            dry_run=True,
        )

        response = await execute_tunix_run(request, db_session)

        # Should fail with empty dataset
        assert response.status == "failed"
        assert response.mode == "dry-run"
        assert "empty" in response.stderr.lower() or "empty" in response.message.lower()

    finally:
        # Cleanup
        manifest_path.unlink(missing_ok=True)
        try:
            manifest_path.parent.rmdir()
        except OSError:
            pass


@pytest.mark.asyncio
async def test_dry_run_with_valid_dataset(db_session):
    """Test dry-run mode with valid dataset and traces."""
    import uuid

    from tunix_rt_backend.helpers.datasets import save_manifest
    from tunix_rt_backend.schemas import DatasetManifest, TunixRunRequest
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    # Create a trace in DB
    trace = Trace(
        trace_version="1.0",
        payload={
            "trace_version": "1.0",
            "prompt": "Test prompt",
            "final_answer": "Test answer",
            "steps": [{"i": 0, "type": "test", "content": "Test step"}],
        },
    )
    db_session.add(trace)
    await db_session.commit()
    await db_session.refresh(trace)

    # Create dataset manifest using proper schema
    dataset_key = "test_valid-v1"
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        build_id=str(uuid.uuid4()),
        dataset_name="test_valid",
        dataset_version="v1",
        dataset_schema_version="1.0",
        created_at="2025-12-22T00:00:00Z",
        filters={},
        selection_strategy="latest",
        seed=None,
        trace_ids=[str(trace.id)],
        trace_count=1,
        stats={},
        session_id=None,
        parent_dataset_id=None,
        training_run_id=None,
    )

    manifest_path = save_manifest(manifest)

    try:
        request = TunixRunRequest(
            dataset_key=dataset_key,
            model_id="google/gemma-2b-it",
            dry_run=True,
        )

        response = await execute_tunix_run(request, db_session)

        # Debug: print response if failed
        if response.status != "completed":
            print(f"Status: {response.status}")
            print(f"Message: {response.message}")
            print(f"stderr: {response.stderr}")
            print(f"stdout: {response.stdout}")

        # Should succeed
        assert response.status == "completed"
        assert response.mode == "dry-run"
        assert response.exit_code == 0
        assert "validation completed successfully" in response.stdout.lower()
        assert response.dataset_key == dataset_key
        assert response.model_id == "google/gemma-2b-it"
        assert response.run_id is not None
        assert response.started_at is not None
        assert response.completed_at is not None
        assert response.duration_seconds is not None
        assert response.duration_seconds >= 0

    finally:
        # Cleanup
        manifest_path.unlink(missing_ok=True)
        # Try to remove directory, ignore if not empty
        try:
            manifest_path.parent.rmdir()
        except OSError:
            pass


@pytest.mark.asyncio
async def test_local_execution_without_tunix_returns_501(client: AsyncClient):
    """Test local execution (dry_run=false) returns 501 when Tunix unavailable."""
    response = await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": "test-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": False,  # Request local execution
        },
    )

    # Should return 501 Not Implemented (Tunix not available)
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
    data = response.json()
    assert "detail" in data
    assert "Tunix runtime not available" in data["detail"]


def test_dry_run_request_schema_validation():
    """Test TunixRunRequest schema validation."""
    from pydantic import ValidationError

    from tunix_rt_backend.schemas import TunixRunRequest

    # Valid minimal request
    request = TunixRunRequest(
        dataset_key="test-v1",
        model_id="google/gemma-2b-it",
    )
    assert request.dry_run is True  # Default
    assert request.learning_rate == 2e-5  # Default
    assert request.num_epochs == 3  # Default

    # Invalid dataset_key (too short)
    with pytest.raises(ValidationError):
        TunixRunRequest(
            dataset_key="",
            model_id="google/gemma-2b-it",
        )

    # Invalid learning_rate (too high)
    with pytest.raises(ValidationError):
        TunixRunRequest(
            dataset_key="test-v1",
            model_id="google/gemma-2b-it",
            learning_rate=2.0,  # > 1.0
        )


def test_run_response_schema_structure():
    """Test TunixRunResponse schema structure."""
    from tunix_rt_backend.schemas import TunixRunResponse

    response = TunixRunResponse(
        run_id="test-uuid",
        status="completed",
        mode="dry-run",
        dataset_key="test-v1",
        model_id="google/gemma-2b-it",
        output_dir="./output/test",
        exit_code=0,
        stdout="Test output",
        stderr="",
        duration_seconds=1.5,
        started_at="2025-12-22T00:00:00Z",
        completed_at="2025-12-22T00:00:01.5Z",
        message="Success",
    )

    assert response.run_id == "test-uuid"
    assert response.status == "completed"
    assert response.mode == "dry-run"
    assert response.exit_code == 0
    assert response.duration_seconds == 1.5


# Optional tests (require Tunix installed)


@pytest.mark.tunix
@pytest.mark.asyncio
async def test_local_execution_with_tunix(db_session):
    """Test local execution with Tunix installed (smoke test).

    This test is SKIPPED if Tunix is not installed.
    Run with: pytest -m tunix tests/test_tunix_execution.py
    """
    if not tunix_available():
        pytest.skip("Tunix not installed; use: pip install -e '.[tunix]'")

    import uuid

    from tunix_rt_backend.helpers.datasets import save_manifest
    from tunix_rt_backend.schemas import DatasetManifest, TunixRunRequest
    from tunix_rt_backend.services.tunix_execution import execute_tunix_run

    # Create a minimal trace for smoke test
    trace = Trace(
        trace_version="1.0",
        payload={
            "trace_version": "1.0",
            "prompt": "What is 2 + 2?",
            "final_answer": "4",
            "steps": [{"i": 0, "type": "compute", "content": "Add 2 + 2 = 4"}],
        },
    )
    db_session.add(trace)
    await db_session.commit()
    await db_session.refresh(trace)

    # Create dataset manifest (1 trace only for speed) using proper schema
    dataset_key = "tunix_smoke-v1"
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        build_id=str(uuid.uuid4()),
        dataset_name="tunix_smoke",
        dataset_version="v1",
        dataset_schema_version="1.0",
        created_at="2025-12-22T00:00:00Z",
        filters={},
        selection_strategy="latest",
        seed=None,
        trace_ids=[str(trace.id)],
        trace_count=1,
        stats={},
        session_id=None,
        parent_dataset_id=None,
        training_run_id=None,
    )

    manifest_path = save_manifest(manifest)

    try:
        request = TunixRunRequest(
            dataset_key=dataset_key,
            model_id="google/gemma-2b-it",
            dry_run=False,  # Local execution
            num_epochs=1,
            batch_size=1,
            max_seq_length=128,  # Small for speed
        )

        response = await execute_tunix_run(request, db_session)

        # Should complete (or fail gracefully)
        assert response.status in ["completed", "failed", "timeout"]
        assert response.mode == "local"
        assert response.run_id is not None
        assert response.started_at is not None
        # Note: exit_code may be non-zero if training fails (expected for smoke test)
        # The important part is that execution happened

    finally:
        # Cleanup
        manifest_path.unlink(missing_ok=True)
        try:
            manifest_path.parent.rmdir()
        except OSError:
            pass


@pytest.mark.tunix
def test_tunix_cli_check():
    """Test check_tunix_cli() utility function."""
    if not tunix_available():
        pytest.skip("Tunix not installed; use: pip install -e '.[tunix]'")

    from tunix_rt_backend.integrations.tunix.availability import check_tunix_cli

    cli_status = check_tunix_cli()

    assert cli_status["accessible"] is True
    assert cli_status["version"] is not None
    assert cli_status["error"] is None
