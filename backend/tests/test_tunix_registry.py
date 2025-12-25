"""Tests for Tunix run registry (M14).

This module tests the M14 run persistence and registry functionality:
- Run record creation and persistence
- List runs endpoint with pagination and filtering
- Get run detail endpoint
- Database failure handling

All tests use dry-run mode (no Tunix runtime required).
"""

import uuid
from datetime import UTC, datetime
from typing import AsyncGenerator
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db
from tunix_rt_backend.db.models import Trace, TunixRun
from tunix_rt_backend.helpers.datasets import save_manifest
from tunix_rt_backend.schemas import DatasetManifest

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


@pytest_asyncio.fixture
async def test_dataset(test_db: AsyncSession) -> str:
    """Create a test dataset with one trace.

    Args:
        test_db: Test database session

    Returns:
        Dataset key (name-version)
    """
    # Create trace
    trace = Trace(
        trace_version="1.0",
        payload={
            "trace_version": "1.0",
            "prompt": "Test prompt",
            "final_answer": "Test answer",
            "steps": [{"i": 0, "type": "test", "content": "Test step"}],
        },
    )
    test_db.add(trace)
    await test_db.commit()
    await test_db.refresh(trace)

    # Create dataset manifest
    dataset_key = "test_dataset-v1"
    from datetime import UTC, datetime

    manifest = DatasetManifest(
        dataset_key=dataset_key,
        dataset_name="test_dataset",
        dataset_version="v1",
        trace_ids=[str(trace.id)],
        trace_count=1,
        filters={},
        selection_strategy="latest",
        build_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        stats={},
        session_id=None,
        parent_dataset_id=None,
        training_run_id=None,
    )
    save_manifest(manifest)

    return dataset_key


# ===========================
# M14: Run Persistence Tests
# ===========================


@pytest.mark.asyncio
async def test_run_persists_to_database(
    client: AsyncClient, test_dataset: str, test_db: AsyncSession
) -> None:
    """Test that Tunix runs are persisted to the database."""
    # Execute dry-run
    response = await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": test_dataset,
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    run_id = UUID(data["run_id"])

    # Verify run exists in database
    result = await test_db.execute(select(TunixRun).where(TunixRun.run_id == run_id))
    db_run = result.scalar_one_or_none()

    assert db_run is not None
    assert db_run.dataset_key == test_dataset
    assert db_run.model_id == "google/gemma-2b-it"
    assert db_run.mode == "dry-run"
    assert db_run.status == "completed"
    assert db_run.exit_code == 0
    assert db_run.started_at is not None
    assert db_run.completed_at is not None
    assert db_run.duration_seconds is not None
    assert db_run.stdout != ""
    assert db_run.stderr == ""


@pytest.mark.asyncio
async def test_run_persists_with_failure(client: AsyncClient, test_db: AsyncSession) -> None:
    """Test that failed runs are persisted with failure status."""
    # Execute dry-run with non-existent dataset
    response = await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": "nonexistent-v1",
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "failed"
    run_id = UUID(data["run_id"])

    # Verify run exists in database with failed status
    result = await test_db.execute(select(TunixRun).where(TunixRun.run_id == run_id))
    db_run = result.scalar_one_or_none()

    assert db_run is not None
    assert db_run.status == "failed"
    assert db_run.stderr != ""  # Should have error message


# ===========================
# M14: List Runs Endpoint Tests
# ===========================


@pytest.mark.asyncio
async def test_list_runs_empty(client: AsyncClient) -> None:
    """Test listing runs when database is empty."""
    response = await client.get("/api/tunix/runs")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["data"] == []
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["next_offset"] is None


@pytest.mark.asyncio
async def test_list_runs_with_runs(
    client: AsyncClient, test_dataset: str, test_db: AsyncSession
) -> None:
    """Test listing runs with multiple runs in database."""
    # Create 3 runs
    for i in range(3):
        await client.post(
            "/api/tunix/run",
            json={
                "dataset_key": test_dataset,
                "model_id": f"model-{i}",
                "dry_run": True,
            },
        )

    # List runs
    response = await client.get("/api/tunix/runs")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 3
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["next_offset"] is None

    # Verify runs are sorted by created_at DESC (most recent first)
    run_ids = [r["run_id"] for r in data["data"]]
    assert len(run_ids) == 3
    assert len(set(run_ids)) == 3  # All unique


@pytest.mark.asyncio
async def test_list_runs_pagination(
    client: AsyncClient, test_dataset: str, test_db: AsyncSession
) -> None:
    """Test pagination of runs list."""
    # Create 5 runs
    for i in range(5):
        await client.post(
            "/api/tunix/run",
            json={
                "dataset_key": test_dataset,
                "model_id": f"model-{i}",
                "dry_run": True,
            },
        )

    # Get first page
    response = await client.get("/api/tunix/runs?limit=2&offset=0")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 2
    assert data["pagination"]["next_offset"] == 2

    # Get second page
    response = await client.get("/api/tunix/runs?limit=2&offset=2")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 2
    assert data["pagination"]["next_offset"] == 4

    # Get third page
    response = await client.get("/api/tunix/runs?limit=2&offset=4")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 1
    assert data["pagination"]["next_offset"] is None


@pytest.mark.asyncio
async def test_list_runs_filter_by_status(
    client: AsyncClient, test_dataset: str, test_db: AsyncSession
) -> None:
    """Test filtering runs by status."""
    # Create successful run
    await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": test_dataset,
            "model_id": "model-success",
            "dry_run": True,
        },
    )

    # Create failed run
    failed_response = await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": "nonexistent-v1",
            "model_id": "model-failed",
            "dry_run": True,
        },
    )
    # Verify it actually failed
    assert failed_response.json()["status"] == "failed", (
        f"Expected failed, got: {failed_response.json()}"
    )

    # Filter by completed status
    response = await client.get("/api/tunix/runs?status=completed")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 1, (
        f"Expected 1 completed run, got {len(data['data'])}: {data['data']}"
    )
    assert data["data"][0]["status"] == "completed"

    # Filter by failed status
    response = await client.get("/api/tunix/runs?status=failed")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 1, f"Expected 1 failed run, got {len(data['data'])}: {data['data']}"
    assert data["data"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_list_runs_filter_by_dataset_key(client: AsyncClient, test_db: AsyncSession) -> None:
    """Test filtering runs by dataset_key."""
    # Create dataset 1
    trace1 = Trace(
        trace_version="1.0",
        payload={
            "trace_version": "1.0",
            "prompt": "Test",
            "final_answer": "Answer",
            "steps": [{"i": 0, "type": "test", "content": "Test"}],
        },
    )
    test_db.add(trace1)
    await test_db.commit()
    await test_db.refresh(trace1)

    dataset_key_1 = "dataset_1-v1"
    manifest1 = DatasetManifest(
        dataset_key=dataset_key_1,
        dataset_name="dataset_1",
        dataset_version="v1",
        trace_ids=[str(trace1.id)],
        trace_count=1,
        filters={},
        selection_strategy="latest",
        build_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        stats={},
        session_id=None,
        parent_dataset_id=None,
        training_run_id=None,
    )
    save_manifest(manifest1)

    # Create dataset 2
    trace2 = Trace(
        trace_version="1.0",
        payload={
            "trace_version": "1.0",
            "prompt": "Test2",
            "final_answer": "Answer2",
            "steps": [{"i": 0, "type": "test", "content": "Test2"}],
        },
    )
    test_db.add(trace2)
    await test_db.commit()
    await test_db.refresh(trace2)

    dataset_key_2 = "dataset_2-v1"
    manifest2 = DatasetManifest(
        dataset_key=dataset_key_2,
        dataset_name="dataset_2",
        dataset_version="v1",
        trace_ids=[str(trace2.id)],
        trace_count=1,
        filters={},
        selection_strategy="latest",
        build_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        stats={},
        session_id=None,
        parent_dataset_id=None,
        training_run_id=None,
    )
    save_manifest(manifest2)

    # Create runs for both datasets
    await client.post(
        "/api/tunix/run",
        json={"dataset_key": dataset_key_1, "model_id": "model-1", "dry_run": True},
    )
    await client.post(
        "/api/tunix/run",
        json={"dataset_key": dataset_key_2, "model_id": "model-2", "dry_run": True},
    )

    # Filter by dataset_key_1
    response = await client.get(f"/api/tunix/runs?dataset_key={dataset_key_1}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["dataset_key"] == dataset_key_1


@pytest.mark.asyncio
async def test_list_runs_filter_by_mode(
    client: AsyncClient, test_dataset: str, test_db: AsyncSession
) -> None:
    """Test filtering runs by execution mode."""
    # Create dry-run
    await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": test_dataset,
            "model_id": "model-dry",
            "dry_run": True,
        },
    )

    # Filter by dry-run mode
    response = await client.get("/api/tunix/runs?mode=dry-run")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["mode"] == "dry-run"


@pytest.mark.asyncio
async def test_list_runs_invalid_pagination(client: AsyncClient) -> None:
    """Test validation of pagination parameters."""
    # Invalid limit (> 100)
    response = await client.get("/api/tunix/runs?limit=101")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Invalid limit (< 1)
    response = await client.get("/api/tunix/runs?limit=0")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Invalid offset (< 0)
    response = await client.get("/api/tunix/runs?offset=-1")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ===========================
# M14: Get Run Detail Tests
# ===========================


@pytest.mark.asyncio
async def test_get_run_detail(
    client: AsyncClient, test_dataset: str, test_db: AsyncSession
) -> None:
    """Test getting run details by ID."""
    # Create run
    create_response = await client.post(
        "/api/tunix/run",
        json={
            "dataset_key": test_dataset,
            "model_id": "google/gemma-2b-it",
            "dry_run": True,
        },
    )
    assert create_response.status_code == status.HTTP_200_OK
    run_id = create_response.json()["run_id"]

    # Get run detail
    response = await client.get(f"/api/tunix/runs/{run_id}")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["run_id"] == run_id
    assert data["dataset_key"] == test_dataset
    assert data["model_id"] == "google/gemma-2b-it"
    assert data["mode"] == "dry-run"
    assert data["status"] == "completed"
    assert data["stdout"] != ""  # Should have validation output
    assert data["message"] != ""


@pytest.mark.asyncio
async def test_get_run_detail_not_found(client: AsyncClient) -> None:
    """Test getting non-existent run returns 404."""
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/tunix/runs/{fake_uuid}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_run_detail_invalid_uuid(client: AsyncClient) -> None:
    """Test getting run with invalid UUID returns 422."""
    response = await client.get("/api/tunix/runs/not-a-uuid")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
