"""Tests for dataset build and export functionality."""

import json
import shutil
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db
from tunix_rt_backend.helpers.datasets import (
    compute_dataset_stats,
    create_dataset_key,
    get_datasets_dir,
    load_manifest,
    save_manifest,
)
from tunix_rt_backend.schemas.dataset import DatasetManifest

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
    """Create test client with database override.

    Args:
        test_db: Test database session

    Yields:
        AsyncClient for testing
    """

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def cleanup_test_datasets():
    """Clean up test datasets after each test."""
    yield
    # Clean up test datasets created during tests
    datasets_dir = get_datasets_dir()
    test_dataset_keywords = ["test", "random", "export", "ordered", "session"]
    for item in datasets_dir.iterdir():
        if item.is_dir() and any(keyword in item.name for keyword in test_dataset_keywords):
            shutil.rmtree(item, ignore_errors=True)


class TestDatasetHelpers:
    """Test dataset helper functions."""

    def test_create_dataset_key(self):
        """Test dataset key creation."""
        key = create_dataset_key("ungar_hcd_baseline", "v1")
        assert key == "ungar_hcd_baseline-v1"

        # Test sanitization
        key = create_dataset_key("my dataset", "v 1.0")
        assert key == "my_dataset-v_1.0"

        # Test slash sanitization
        key = create_dataset_key("path/to/data", "v1/beta")
        assert key == "path_to_data-v1_beta"

    def test_get_datasets_dir_creates_directory(self):
        """Test that get_datasets_dir creates directory if it doesn't exist."""
        # Just verify the function runs and creates a path
        datasets_dir = get_datasets_dir()
        assert datasets_dir.exists()
        assert datasets_dir.name == "datasets"

    def test_save_and_load_manifest(self, tmp_path, monkeypatch):
        """Test saving and loading dataset manifests."""
        from datetime import datetime, timezone

        # Monkeypatch datasets directory
        monkeypatch.setattr(
            "tunix_rt_backend.helpers.datasets.get_datasets_dir", lambda: tmp_path
        )

        # Create a test manifest
        manifest = DatasetManifest(
            dataset_key="test-v1",
            build_id=uuid.uuid4(),
            dataset_name="test",
            dataset_version="v1",
            created_at=datetime.now(timezone.utc),
            filters={"source": "ungar"},
            selection_strategy="latest",
            trace_ids=[uuid.uuid4(), uuid.uuid4()],
            trace_count=2,
            stats={"avg_step_count": 3.5},
        )

        # Save manifest
        manifest_path = save_manifest(manifest)
        assert manifest_path.exists()
        assert manifest_path.name == "manifest.json"

        # Load manifest
        loaded = load_manifest("test-v1")
        assert loaded.dataset_key == manifest.dataset_key
        assert loaded.build_id == manifest.build_id
        assert loaded.trace_count == 2
        assert len(loaded.trace_ids) == 2

    def test_load_manifest_not_found(self, tmp_path, monkeypatch):
        """Test load_manifest raises FileNotFoundError for missing dataset."""
        monkeypatch.setattr(
            "tunix_rt_backend.helpers.datasets.get_datasets_dir", lambda: tmp_path
        )

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            load_manifest("nonexistent-v1")

    def test_compute_dataset_stats_empty(self):
        """Test compute_dataset_stats with empty trace list."""
        stats = compute_dataset_stats([])
        assert stats["trace_count"] == 0
        assert stats["avg_step_count"] == 0.0
        assert stats["min_step_count"] == 0
        assert stats["max_step_count"] == 0

    def test_compute_dataset_stats(self):
        """Test compute_dataset_stats with sample traces."""
        traces = [
            {
                "prompt": "What is 2+2?",
                "final_answer": "4",
                "steps": [
                    {"i": 0, "type": "compute", "content": "Add 2 and 2"},
                ],
            },
            {
                "prompt": "Explain photosynthesis",
                "final_answer": "Plants convert light to energy",
                "steps": [
                    {"i": 0, "type": "define", "content": "Process explanation"},
                    {"i": 1, "type": "detail", "content": "Light absorption"},
                    {"i": 2, "type": "detail", "content": "Chemical conversion"},
                ],
            },
        ]

        stats = compute_dataset_stats(traces)
        assert stats["trace_count"] == 2
        assert stats["avg_step_count"] == 2.0  # (1 + 3) / 2
        assert stats["min_step_count"] == 1
        assert stats["max_step_count"] == 3
        assert stats["avg_total_chars"] > 0


class TestDatasetBuildEndpoint:
    """Test /api/datasets/build endpoint."""

    @pytest.mark.asyncio
    async def test_build_dataset_latest_strategy(self, client: AsyncClient):
        """Test building a dataset with 'latest' strategy."""
        # Create some test traces
        trace1 = await client.post(
            "/api/traces",
            json={
                "trace_version": "1.0",
                "prompt": "Test prompt 1",
                "final_answer": "Answer 1",
                "steps": [{"i": 0, "type": "test", "content": "Step 1"}],
                "meta": {"source": "test"},
            },
        )
        assert trace1.status_code == 201

        trace2 = await client.post(
            "/api/traces",
            json={
                "trace_version": "1.0",
                "prompt": "Test prompt 2",
                "final_answer": "Answer 2",
                "steps": [{"i": 0, "type": "test", "content": "Step 2"}],
                "meta": {"source": "test"},
            },
        )
        assert trace2.status_code == 201

        # Build dataset
        response = await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "test_dataset",
                "dataset_version": "v1",
                "filters": {"source": "test"},
                "limit": 10,
                "selection_strategy": "latest",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["dataset_key"] == "test_dataset-v1"
        assert "build_id" in data
        assert data["trace_count"] == 2
        assert "manifest_path" in data

    @pytest.mark.asyncio
    async def test_build_dataset_random_strategy(self, client: AsyncClient):
        """Test building a dataset with 'random' strategy."""
        # Create test traces
        for i in range(5):
            response = await client.post(
                "/api/traces",
                json={
                    "trace_version": "1.0",
                    "prompt": f"Prompt {i}",
                    "final_answer": f"Answer {i}",
                    "steps": [{"i": 0, "type": "test", "content": f"Step {i}"}],
                    "meta": {"source": "random_test"},
                },
            )
            assert response.status_code == 201

        # Build dataset with random strategy
        response = await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "random_dataset",
                "dataset_version": "v1",
                "filters": {"source": "random_test"},
                "limit": 3,
                "selection_strategy": "random",
                "seed": 42,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["dataset_key"] == "random_dataset-v1"
        assert data["trace_count"] == 3

        # Build again with same seed - should get same traces
        response2 = await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "random_dataset",
                "dataset_version": "v2",
                "filters": {"source": "random_test"},
                "limit": 3,
                "selection_strategy": "random",
                "seed": 42,
            },
        )

        assert response2.status_code == 201
        # Note: Same seed should produce same selection, but we'd need to
        # compare trace IDs from manifests to verify determinism

    @pytest.mark.asyncio
    async def test_build_dataset_random_requires_seed(self, client: AsyncClient):
        """Test that random strategy requires a seed."""
        response = await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "test",
                "dataset_version": "v1",
                "selection_strategy": "random",
                # Missing seed
            },
        )

        assert response.status_code == 422
        assert "seed" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_build_dataset_with_optional_fields(self, client: AsyncClient):
        """Test building dataset with optional multi-session fields."""
        # Create a test trace
        await client.post(
            "/api/traces",
            json={
                "trace_version": "1.0",
                "prompt": "Test",
                "final_answer": "Answer",
                "steps": [{"i": 0, "type": "test", "content": "Step"}],
                "meta": {"source": "session_test"},
            },
        )

        # Build dataset with optional fields
        response = await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "session_dataset",
                "dataset_version": "v1",
                "filters": {"source": "session_test"},
                "session_id": "session_123",
                "parent_dataset_id": "parent-v1",
                "training_run_id": "run_456",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["dataset_key"] == "session_dataset-v1"


class TestDatasetExportEndpoint:
    """Test /api/datasets/{dataset_key}/export.jsonl endpoint."""

    @pytest.mark.asyncio
    async def test_export_dataset_success(self, client: AsyncClient):
        """Test exporting a dataset as JSONL."""
        # Create test traces
        trace_ids = []
        for i in range(3):
            response = await client.post(
                "/api/traces",
                json={
                    "trace_version": "1.0",
                    "prompt": f"Prompt {i}",
                    "final_answer": f"Answer {i}",
                    "steps": [{"i": 0, "type": "test", "content": f"Step {i}"}],
                    "meta": {"source": "export_test"},
                },
            )
            assert response.status_code == 201
            trace_ids.append(response.json()["id"])

        # Build dataset
        build_response = await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "export_dataset",
                "dataset_version": "v1",
                "filters": {"source": "export_test"},
                "limit": 10,
            },
        )
        assert build_response.status_code == 201

        # Export dataset
        export_response = await client.get("/api/datasets/export_dataset-v1/export.jsonl")
        assert export_response.status_code == 200
        assert export_response.headers["content-type"] == "application/x-ndjson"

        # Parse JSONL
        lines = export_response.text.strip().split("\n")
        assert len(lines) == 3

        # Verify each line is valid JSON with expected fields
        for line in lines:
            record = json.loads(line)
            assert "id" in record
            assert "prompts" in record
            assert "trace_steps" in record
            assert "final_answer" in record
            assert "metadata" in record
            assert record["metadata"]["source"] == "export_test"

    @pytest.mark.asyncio
    async def test_export_dataset_not_found(self, client: AsyncClient):
        """Test exporting a non-existent dataset returns 404."""
        response = await client.get("/api/datasets/nonexistent-v1/export.jsonl")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_export_dataset_maintains_order(self, client: AsyncClient):
        """Test that export maintains the order from the manifest."""
        # Create test traces
        expected_order = []
        for i in range(3):
            response = await client.post(
                "/api/traces",
                json={
                    "trace_version": "1.0",
                    "prompt": f"Ordered prompt {i}",
                    "final_answer": f"Answer {i}",
                    "steps": [{"i": 0, "type": "test", "content": f"Content {i}"}],
                    "meta": {"source": "order_test", "index": i},
                },
            )
            assert response.status_code == 201
            expected_order.append(response.json()["id"])

        # Build dataset
        await client.post(
            "/api/datasets/build",
            json={
                "dataset_name": "ordered_dataset",
                "dataset_version": "v1",
                "filters": {"source": "order_test"},
                "limit": 10,
                "selection_strategy": "latest",
            },
        )

        # Export and verify order
        export_response = await client.get("/api/datasets/ordered_dataset-v1/export.jsonl")
        assert export_response.status_code == 200

        lines = export_response.text.strip().split("\n")
        exported_ids = [json.loads(line)["id"] for line in lines]

        # Latest strategy returns newest first, so reverse the expected order
        assert exported_ids == list(reversed(expected_order))

