"""Tests for trace endpoints."""

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db

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


@pytest.fixture
def example_trace() -> dict:
    """Example valid trace payload.

    Returns:
        dict with valid ReasoningTrace data
    """
    return {
        "trace_version": "1.0",
        "prompt": "What is 27 × 19?",
        "final_answer": "513",
        "steps": [
            {"i": 0, "type": "parse", "content": "Parse the multiplication task"},
            {"i": 1, "type": "compute", "content": "Break down: 27 × 19 = 27 × (20 - 1)"},
            {"i": 2, "type": "compute", "content": "Calculate: 27 × 20 = 540"},
            {"i": 3, "type": "compute", "content": "Calculate: 27 × 1 = 27"},
            {"i": 4, "type": "result", "content": "Final: 540 - 27 = 513"},
        ],
        "meta": {"source": "test", "tags": ["math", "multiplication"]},
    }


@pytest.mark.asyncio
async def test_create_trace_success(client: AsyncClient, example_trace: dict) -> None:
    """Test creating a trace successfully."""
    response = await client.post("/api/traces", json=example_trace)

    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert "created_at" in data
    assert data["trace_version"] == "1.0"

    # Verify UUID format
    trace_id = uuid.UUID(data["id"])
    assert isinstance(trace_id, uuid.UUID)


@pytest.mark.asyncio
async def test_create_and_get_trace(client: AsyncClient, example_trace: dict) -> None:
    """Test creating a trace and retrieving it."""
    # Create trace
    create_response = await client.post("/api/traces", json=example_trace)
    assert create_response.status_code == 201
    trace_id = create_response.json()["id"]

    # Get trace
    get_response = await client.get(f"/api/traces/{trace_id}")
    assert get_response.status_code == 200

    data = get_response.json()
    assert data["id"] == trace_id
    assert data["trace_version"] == "1.0"
    assert "payload" in data
    assert data["payload"]["prompt"] == "What is 27 × 19?"
    assert data["payload"]["final_answer"] == "513"
    assert len(data["payload"]["steps"]) == 5


@pytest.mark.asyncio
async def test_get_trace_not_found(client: AsyncClient) -> None:
    """Test getting a trace that doesn't exist."""
    random_id = uuid.uuid4()
    response = await client.get(f"/api/traces/{random_id}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_trace_invalid_schema(client: AsyncClient) -> None:
    """Test creating a trace with invalid schema."""
    invalid_trace = {
        "trace_version": "1.0",
        "prompt": "Test",
        # Missing final_answer and steps
    }

    response = await client.post("/api/traces", json=invalid_trace)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_trace_empty_steps(client: AsyncClient) -> None:
    """Test creating a trace with empty steps list."""
    invalid_trace = {
        "trace_version": "1.0",
        "prompt": "Test",
        "final_answer": "Answer",
        "steps": [],  # Empty - violates min_length=1
    }

    response = await client.post("/api/traces", json=invalid_trace)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_trace_duplicate_step_indices(client: AsyncClient) -> None:
    """Test creating a trace with duplicate step indices."""
    invalid_trace = {
        "trace_version": "1.0",
        "prompt": "Test",
        "final_answer": "Answer",
        "steps": [
            {"i": 0, "type": "step1", "content": "Content 1"},
            {"i": 0, "type": "step2", "content": "Content 2"},  # Duplicate index
        ],
    }

    response = await client.post("/api/traces", json=invalid_trace)
    assert response.status_code == 422
    assert "unique" in response.json()["detail"][0]["msg"].lower()


@pytest.mark.asyncio
async def test_create_trace_oversized_payload(client: AsyncClient) -> None:
    """Test creating a trace with oversized payload."""
    # Create a trace with very long content to exceed TRACE_MAX_BYTES
    large_content = "x" * (2 * 1024 * 1024)  # 2MB content
    oversized_trace = {
        "trace_version": "1.0",
        "prompt": large_content,
        "final_answer": "Answer",
        "steps": [{"i": 0, "type": "step", "content": "test"}],
    }

    response = await client.post("/api/traces", json=oversized_trace)
    assert response.status_code == 413
    assert "exceeds maximum" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_traces_empty(client: AsyncClient) -> None:
    """Test listing traces when database is empty."""
    response = await client.get("/api/traces")

    assert response.status_code == 200
    data = response.json()

    assert data["data"] == []
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["next_offset"] is None


@pytest.mark.asyncio
async def test_list_traces_with_data(client: AsyncClient, example_trace: dict) -> None:
    """Test listing traces with data."""
    # Create 3 traces
    for _ in range(3):
        await client.post("/api/traces", json=example_trace)

    # List traces
    response = await client.get("/api/traces")
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 3
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["next_offset"] is None


@pytest.mark.asyncio
async def test_list_traces_pagination(client: AsyncClient, example_trace: dict) -> None:
    """Test trace pagination."""
    # Create 25 traces
    for _ in range(25):
        await client.post("/api/traces", json=example_trace)

    # Get first page (limit=10)
    response = await client.get("/api/traces?limit=10&offset=0")
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 10
    assert data["pagination"]["limit"] == 10
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["next_offset"] == 10

    # Get second page
    response = await client.get("/api/traces?limit=10&offset=10")
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 10
    assert data["pagination"]["next_offset"] == 20


@pytest.mark.asyncio
async def test_list_traces_invalid_limit(client: AsyncClient) -> None:
    """Test listing traces with invalid limit."""
    response = await client.get("/api/traces?limit=101")
    assert response.status_code == 422
    assert "limit" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_traces_negative_offset(client: AsyncClient) -> None:
    """Test listing traces with negative offset."""
    response = await client.get("/api/traces?offset=-1")
    assert response.status_code == 422
    assert "offset" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_trace_list_excludes_payload(client: AsyncClient, example_trace: dict) -> None:
    """Test that list endpoint doesn't include full payload."""
    # Create a trace
    await client.post("/api/traces", json=example_trace)

    # List traces
    response = await client.get("/api/traces")
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 1

    # Verify list item doesn't have payload
    list_item = data["data"][0]
    assert "id" in list_item
    assert "created_at" in list_item
    assert "trace_version" in list_item
    assert "payload" not in list_item


@pytest.mark.asyncio
async def test_get_trace_with_invalid_uuid_format(client: AsyncClient) -> None:
    """Test getting a trace with malformed UUID."""
    response = await client.get("/api/traces/not-a-valid-uuid")
    # FastAPI validation should return 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_traces_with_zero_limit(client: AsyncClient) -> None:
    """Test listing traces with limit=0."""
    response = await client.get("/api/traces?limit=0")
    assert response.status_code == 422
    assert "limit" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_trace_success_path_explicit(client: AsyncClient, example_trace: dict) -> None:
    """Explicit test for get_trace success branch (db_trace is NOT None).

    This test ensures the ELSE branch of 'if db_trace is None' is counted as covered.
    """
    # Create trace
    create_response = await client.post("/api/traces", json=example_trace)
    trace_id = create_response.json()["id"]

    # Fetch it (hits the SUCCESS branch: db_trace is NOT None)
    get_response = await client.get(f"/api/traces/{trace_id}")

    # Explicitly verify we took the success branch
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == trace_id
    assert data["payload"] is not None
    assert data["payload"]["prompt"] == "What is 27 × 19?"
    assert data["payload"]["final_answer"] == "513"


@pytest.mark.asyncio
async def test_list_traces_valid_pagination(client: AsyncClient, example_trace: dict) -> None:
    """Explicit test for list_traces validation success branches.

    Ensures the ELSE paths of limit/offset validation are covered:
    - if limit < 1 or limit > 100 → FALSE (valid limit)
    - if offset < 0 → FALSE (valid offset)
    """
    # Create a trace
    await client.post("/api/traces", json=example_trace)

    # Valid pagination (hits success branches)
    response = await client.get("/api/traces?limit=20&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "pagination" in data
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
