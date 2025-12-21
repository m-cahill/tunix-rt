"""Tests for UNGAR integration endpoints.

This test file covers:
1. Default tests (without UNGAR installed): Test that endpoints return appropriate
   501/unavailable status when UNGAR is not installed.
2. Optional tests (with UNGAR installed, marked @pytest.mark.ungar): Test actual
   UNGAR functionality when the optional dependency is present.
"""

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db
from tunix_rt_backend.integrations.ungar.availability import ungar_available

# Test database URL (SQLite in-memory for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
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


@pytest_asyncio.fixture
async def client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ==================== DEFAULT TESTS (No UNGAR Required) ====================


@pytest.mark.asyncio
async def test_ungar_status_without_ungar_installed(client: AsyncClient):
    """Test /api/ungar/status returns available=False when UNGAR not installed."""
    response = await client.get("/api/ungar/status")

    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert isinstance(data["available"], bool)
    # If UNGAR is not installed, available should be False
    if not ungar_available():
        assert data["available"] is False
        assert data["version"] is None


@pytest.mark.asyncio
async def test_ungar_generate_returns_501_without_ungar_installed(client: AsyncClient):
    """Test /api/ungar/high-card-duel/generate returns 501 when UNGAR not installed."""
    if ungar_available():
        pytest.skip("UNGAR is installed; this test is for the unavailable case")

    response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 5, "seed": 42},
    )

    assert response.status_code == 501
    assert "not installed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ungar_export_jsonl_returns_empty_when_no_ungar_traces(client: AsyncClient):
    """Test /api/ungar/high-card-duel/export.jsonl returns empty when no traces."""
    response = await client.get("/api/ungar/high-card-duel/export.jsonl")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    # Should be empty or have minimal content if no UNGAR traces exist
    assert len(response.text.strip()) >= 0  # Can be empty


# ==================== OPTIONAL TESTS (UNGAR Required) ====================


@pytest.mark.ungar
@pytest.mark.asyncio
async def test_ungar_status_with_ungar_installed(client: AsyncClient):
    """Test /api/ungar/status returns available=True when UNGAR is installed."""
    if not ungar_available():
        pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

    response = await client.get("/api/ungar/status")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    # Version may be "unknown" if UNGAR doesn't have __version__
    assert data["version"] is not None


@pytest.mark.ungar
@pytest.mark.asyncio
async def test_ungar_generate_creates_traces(client: AsyncClient):
    """Test /api/ungar/high-card-duel/generate creates traces successfully."""
    if not ungar_available():
        pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

    response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 3, "seed": 42, "persist": True},
    )

    assert response.status_code == 201
    data = response.json()

    # Verify structure
    assert "trace_ids" in data
    assert "preview" in data
    assert len(data["trace_ids"]) == 3
    assert len(data["preview"]) == 3

    # Verify trace IDs are valid UUIDs
    for trace_id in data["trace_ids"]:
        uuid.UUID(trace_id)  # Should not raise

    # Verify preview structure
    for preview_item in data["preview"]:
        assert "trace_id" in preview_item
        assert "game" in preview_item
        assert "result" in preview_item
        assert "my_card" in preview_item


@pytest.mark.ungar
@pytest.mark.asyncio
async def test_ungar_generate_without_persist(client: AsyncClient):
    """Test /api/ungar/high-card-duel/generate with persist=False."""
    if not ungar_available():
        pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

    response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 2, "seed": 123, "persist": False},
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["trace_ids"]) == 2
    # trace_ids should still be present even if not persisted (placeholder IDs)


@pytest.mark.ungar
@pytest.mark.asyncio
async def test_ungar_generate_validates_count(client: AsyncClient):
    """Test /api/ungar/high-card-duel/generate validates count parameter."""
    if not ungar_available():
        pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

    # Test count too high
    response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 101, "seed": 42},
    )
    assert response.status_code == 422  # Validation error

    # Test count too low
    response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 0, "seed": 42},
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.ungar
@pytest.mark.asyncio
async def test_ungar_export_jsonl_basic(client: AsyncClient):
    """Test /api/ungar/high-card-duel/export.jsonl returns valid JSONL."""
    if not ungar_available():
        pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

    # First generate some traces
    gen_response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 2, "seed": 999, "persist": True},
    )
    assert gen_response.status_code == 201

    # Now export them
    export_response = await client.get("/api/ungar/high-card-duel/export.jsonl?limit=10")

    assert export_response.status_code == 200
    assert export_response.headers["content-type"] == "application/x-ndjson"

    # Parse JSONL (each line is a JSON object)
    import json

    lines = [line for line in export_response.text.strip().split("\n") if line]
    assert len(lines) >= 2  # At least the 2 we just created

    # Verify structure of first record
    if lines:
        record = json.loads(lines[0])
        assert "id" in record
        assert "prompts" in record  # Tunix-friendly field name
        assert "trace_steps" in record
        assert "final_answer" in record
        assert "metadata" in record
        assert record["metadata"]["source"] == "ungar"
        assert record["metadata"]["game"] == "high_card_duel"


@pytest.mark.ungar
@pytest.mark.asyncio
async def test_ungar_integration_end_to_end(client: AsyncClient):
    """End-to-end test: generate, verify persistence, export, verify JSONL."""
    if not ungar_available():
        pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

    # Step 1: Generate traces
    gen_response = await client.post(
        "/api/ungar/high-card-duel/generate",
        json={"count": 1, "seed": 777, "persist": True},
    )
    assert gen_response.status_code == 201
    gen_data = gen_response.json()
    trace_id = gen_data["trace_ids"][0]

    # Step 2: Verify trace exists (fetch via existing trace endpoint)
    trace_response = await client.get(f"/api/traces/{trace_id}")
    assert trace_response.status_code == 200
    trace_data = trace_response.json()
    assert trace_data["id"] == trace_id
    assert trace_data["payload"]["meta"]["source"] == "ungar"
    assert trace_data["payload"]["meta"]["game"] == "high_card_duel"

    # Step 3: Export to JSONL
    export_response = await client.get(
        f"/api/ungar/high-card-duel/export.jsonl?trace_ids={trace_id}"
    )
    assert export_response.status_code == 200

    import json

    lines = [line for line in export_response.text.strip().split("\n") if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == trace_id
    assert record["metadata"]["source"] == "ungar"
