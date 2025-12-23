"""Tests for Tunix integration endpoints and services.

M12: Mock-first Tunix integration tests
- Default tests: Work without Tunix installed (no runtime dependency)
- Service tests: Validate JSONL export and manifest generation
"""

import json
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
import yaml
from fastapi import status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db
from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas import ReasoningTrace, TraceStep

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

    # Override get_db dependency
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    # Clear overrides
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db(test_db: AsyncSession) -> AsyncSession:
    """Alias for test_db to match test expectations.

    Args:
        test_db: Test database session

    Returns:
        AsyncSession for testing
    """
    return test_db


# ================================================================================
# TEST FIXTURES
# ================================================================================


@pytest_asyncio.fixture
async def sample_traces(db: AsyncSession) -> list[uuid.UUID]:
    """Create sample traces for testing export.

    Returns:
        List of trace IDs
    """
    trace_ids = []

    for i in range(3):
        trace_data = ReasoningTrace(
            trace_version="1.0",
            prompt=f"Test prompt {i}",
            final_answer=f"Answer {i}",
            steps=[
                TraceStep(i=0, type="step", content=f"Step 0 for trace {i}"),
                TraceStep(i=1, type="step", content=f"Step 1 for trace {i}"),
            ],
            meta={"source": "test", "index": i},
        )

        db_trace = Trace(
            trace_version=trace_data.trace_version,
            payload=trace_data.model_dump(),
        )
        db.add(db_trace)
        await db.commit()
        await db.refresh(db_trace)
        trace_ids.append(db_trace.id)

    return trace_ids


# ================================================================================
# AVAILABILITY TESTS (No runtime dependency)
# ================================================================================


def test_tunix_available_returns_false():
    """Test that tunix_available returns False (M12 mock-first)."""
    from tunix_rt_backend.integrations.tunix.availability import tunix_available

    assert tunix_available() is False


def test_tunix_version_returns_none():
    """Test that tunix_version returns None (M12 mock-first)."""
    from tunix_rt_backend.integrations.tunix.availability import tunix_version

    assert tunix_version() is None


def test_tunix_availability_m13():
    """Test that tunix_available checks actual Tunix installation (M13)."""
    from tunix_rt_backend.integrations.tunix.availability import tunix_available

    # M13: Should check real Tunix availability (both package and CLI)
    # Without Tunix installed, returns False
    assert tunix_available() is False


# ================================================================================
# STATUS ENDPOINT TESTS
# ================================================================================


@pytest.mark.asyncio
async def test_tunix_status_endpoint(client: AsyncClient):
    """Test /api/tunix/status endpoint returns correct status.

    M13: Should show available=False (no Tunix installed), runtime_required=True
    """
    response = await client.get("/api/tunix/status")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["available"] is False
    assert data["version"] is None
    assert data["runtime_required"] is True  # M13: True when not available
    assert "dry-run" in data["message"].lower() or "install" in data["message"].lower()


# ================================================================================
# EXPORT ENDPOINT TESTS
# ================================================================================


@pytest.mark.asyncio
async def test_tunix_export_requires_dataset_or_traces(client: AsyncClient):
    """Test export endpoint requires either dataset_key or trace_ids."""
    response = await client.post(
        "/api/tunix/sft/export",
        json={
            # No dataset_key or trace_ids
            "limit": 10
        },
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "dataset_key or trace_ids" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_tunix_export_with_trace_ids(client: AsyncClient, sample_traces: list[uuid.UUID]):
    """Test export endpoint with specific trace IDs."""
    # Export 2 of 3 traces
    trace_ids = [str(sample_traces[0]), str(sample_traces[1])]

    response = await client.post(
        "/api/tunix/sft/export",
        json={
            "trace_ids": trace_ids,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "application/x-ndjson"

    # Parse JSONL
    lines = response.text.strip().split("\n")
    assert len(lines) == 2

    # Validate each line is valid JSON with tunix_sft format
    for line in lines:
        record = json.loads(line)
        assert "id" in record
        assert "prompts" in record  # tunix_sft uses 'prompts'
        assert "final_answer" in record
        assert "metadata" in record
        # tunix_sft format should have Gemma chat template
        assert "<start_of_turn>" in record["prompts"]
        assert "<end_of_turn>" in record["prompts"]


@pytest.mark.asyncio
async def test_tunix_export_with_nonexistent_dataset(client: AsyncClient):
    """Test export endpoint with non-existent dataset."""
    response = await client.post(
        "/api/tunix/sft/export",
        json={
            "dataset_key": "nonexistent-v1",
        },
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


# ================================================================================
# MANIFEST ENDPOINT TESTS
# ================================================================================


@pytest.mark.asyncio
async def test_tunix_manifest_generation(client: AsyncClient, sample_dataset: str):
    """Test manifest generation endpoint.

    Args:
        sample_dataset: Fixture that creates a dataset manifest
    """
    response = await client.post(
        "/api/tunix/sft/manifest",
        json={
            "dataset_key": sample_dataset,
            "model_id": "google/gemma-2b-it",
            "output_dir": "./output/test_run",
            "learning_rate": 1e-4,
            "num_epochs": 5,
            "batch_size": 16,
            "max_seq_length": 1024,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    assert data["dataset_key"] == sample_dataset
    assert data["model_id"] == "google/gemma-2b-it"
    assert data["format"] == "tunix_sft"
    assert "manifest" in data["message"].lower()

    # Validate YAML content
    manifest_yaml = data["manifest_yaml"]
    manifest_dict = yaml.safe_load(manifest_yaml)

    assert manifest_dict["version"] == "1.0"
    assert manifest_dict["runner"] == "tunix"
    assert manifest_dict["mode"] == "sft"
    assert manifest_dict["model"]["model_id"] == "google/gemma-2b-it"
    assert manifest_dict["dataset"]["format"] == "tunix_sft"
    assert sample_dataset in manifest_dict["dataset"]["path"]
    assert manifest_dict["training"]["learning_rate"] == 1e-4
    assert manifest_dict["training"]["num_epochs"] == 5
    assert manifest_dict["training"]["batch_size"] == 16
    assert manifest_dict["training"]["max_seq_length"] == 1024
    assert manifest_dict["output"]["output_dir"] == "./output/test_run"


@pytest.mark.asyncio
async def test_tunix_manifest_with_defaults(client: AsyncClient, sample_dataset: str):
    """Test manifest generation uses default hyperparameters."""
    response = await client.post(
        "/api/tunix/sft/manifest",
        json={
            "dataset_key": sample_dataset,
            "model_id": "google/gemma-2b-it",
            "output_dir": "./output/test_run",
            # Omit all hyperparameters - should use defaults
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    manifest_dict = yaml.safe_load(data["manifest_yaml"])

    # Verify defaults from M12_answers.md
    assert manifest_dict["training"]["learning_rate"] == 2e-5
    assert manifest_dict["training"]["num_epochs"] == 3
    assert manifest_dict["training"]["batch_size"] == 8
    assert manifest_dict["training"]["max_seq_length"] == 2048


@pytest.mark.asyncio
async def test_tunix_manifest_with_nonexistent_dataset(client: AsyncClient):
    """Test manifest generation fails for non-existent dataset."""
    response = await client.post(
        "/api/tunix/sft/manifest",
        json={
            "dataset_key": "nonexistent-v1",
            "model_id": "google/gemma-2b-it",
            "output_dir": "./output/test_run",
        },
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


# ================================================================================
# SERVICE LAYER TESTS
# ================================================================================


@pytest.mark.asyncio
async def test_export_service_with_trace_ids(db: AsyncSession, sample_traces: list[uuid.UUID]):
    """Test tunix_export service with trace IDs."""
    from tunix_rt_backend.schemas import TunixExportRequest
    from tunix_rt_backend.services.tunix_export import export_tunix_sft_jsonl

    request = TunixExportRequest(
        trace_ids=[str(sample_traces[0])],
    )

    jsonl_content = await export_tunix_sft_jsonl(request, db)

    # Validate output
    assert jsonl_content
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["id"] == str(sample_traces[0])
    assert "<start_of_turn>" in record["prompts"]


@pytest.mark.asyncio
async def test_export_service_requires_dataset_or_traces(db: AsyncSession):
    """Test export service fails without dataset_key or trace_ids."""
    from tunix_rt_backend.schemas import TunixExportRequest
    from tunix_rt_backend.services.tunix_export import export_tunix_sft_jsonl

    request = TunixExportRequest(
        # No dataset_key or trace_ids
    )

    with pytest.raises(ValueError, match="dataset_key or trace_ids"):
        await export_tunix_sft_jsonl(request, db)


def test_manifest_builder_generates_valid_yaml():
    """Test manifest builder produces valid YAML."""
    from tunix_rt_backend.integrations.tunix.manifest import build_sft_manifest
    from tunix_rt_backend.schemas import TunixManifestRequest

    request = TunixManifestRequest(
        dataset_key="test-v1",
        model_id="google/gemma-2b-it",
        output_dir="./output/test",
        learning_rate=1e-4,
        num_epochs=10,
        batch_size=32,
        max_seq_length=4096,
    )

    yaml_content = build_sft_manifest(request, "./datasets/test-v1.jsonl")

    # Parse YAML
    manifest = yaml.safe_load(yaml_content)

    # Validate structure
    assert manifest["version"] == "1.0"
    assert manifest["runner"] == "tunix"
    assert manifest["mode"] == "sft"
    assert manifest["model"]["model_id"] == "google/gemma-2b-it"
    assert manifest["dataset"]["format"] == "tunix_sft"
    assert manifest["dataset"]["path"] == "./datasets/test-v1.jsonl"
    assert manifest["training"]["learning_rate"] == 1e-4
    assert manifest["training"]["num_epochs"] == 10
    assert manifest["training"]["batch_size"] == 32
    assert manifest["training"]["max_seq_length"] == 4096
    assert manifest["output"]["output_dir"] == "./output/test"


# ================================================================================
# INTEGRATION TESTS (End-to-End)
# ================================================================================


@pytest.mark.asyncio
async def test_tunix_workflow_end_to_end(client: AsyncClient, sample_dataset: str):
    """Test complete Tunix workflow: status → export → manifest.

    Args:
        sample_dataset: Fixture that creates a dataset with traces
    """
    # 1. Check status
    status_response = await client.get("/api/tunix/status")
    assert status_response.status_code == status.HTTP_200_OK
    # M13: runtime_required is True when Tunix is not installed
    assert status_response.json()["runtime_required"] is True

    # 2. Export dataset
    export_response = await client.post(
        "/api/tunix/sft/export",
        json={"dataset_key": sample_dataset},
    )
    assert export_response.status_code == status.HTTP_200_OK
    assert export_response.headers["content-type"] == "application/x-ndjson"

    # 3. Generate manifest
    manifest_response = await client.post(
        "/api/tunix/sft/manifest",
        json={
            "dataset_key": sample_dataset,
            "model_id": "google/gemma-2b-it",
            "output_dir": "./output/e2e_test",
        },
    )
    assert manifest_response.status_code == status.HTTP_201_CREATED

    data = manifest_response.json()
    manifest = yaml.safe_load(data["manifest_yaml"])

    # Verify manifest references correct dataset
    assert sample_dataset in manifest["dataset"]["path"]
    assert manifest["dataset"]["format"] == "tunix_sft"


# ================================================================================
# FIXTURE: Sample Dataset
# ================================================================================


@pytest_asyncio.fixture
async def sample_dataset(db: AsyncSession, sample_traces: list[uuid.UUID]) -> str:
    """Create a sample dataset manifest for testing.

    Args:
        db: Database session
        sample_traces: List of trace IDs

    Returns:
        Dataset key (name-version)
    """
    from datetime import UTC, datetime

    from tunix_rt_backend.helpers.datasets import save_manifest
    from tunix_rt_backend.schemas import DatasetManifest

    dataset_key = "test_tunix-v1"
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        build_id=uuid.uuid4(),
        dataset_name="test_tunix",
        dataset_version="v1",
        created_at=datetime.now(UTC),
        selection_strategy="latest",
        trace_ids=sample_traces,
        trace_count=len(sample_traces),
    )

    save_manifest(manifest)
    return dataset_key
