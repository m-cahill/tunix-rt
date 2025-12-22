"""Tests for batch trace operations."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db():
    """Create a test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def client(test_db: AsyncSession):
    """Create test client with database override."""

    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


class TestBatchTraceCreation:
    """Tests for batch trace creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_traces_batch_success(self, client: AsyncClient):
        """Test creating multiple traces in a batch."""
        traces = [
            {
                "trace_version": "1.0",
                "prompt": f"Question {i}",
                "final_answer": f"Answer {i}",
                "steps": [{"i": 0, "type": "test", "content": f"Step {i}"}],
            }
            for i in range(5)
        ]

        response = await client.post("/api/traces/batch", json=traces)
        assert response.status_code == 201

        data = response.json()
        assert data["created_count"] == 5
        assert len(data["traces"]) == 5

        # Verify each trace has required fields
        for trace in data["traces"]:
            assert "id" in trace
            assert "created_at" in trace
            assert "trace_version" in trace
            assert trace["trace_version"] == "1.0"

    @pytest.mark.asyncio
    async def test_create_traces_batch_single(self, client: AsyncClient):
        """Test batch with single trace."""
        traces = [
            {
                "trace_version": "1.0",
                "prompt": "Single question",
                "final_answer": "Single answer",
                "steps": [{"i": 0, "type": "test", "content": "Single step"}],
            }
        ]

        response = await client.post("/api/traces/batch", json=traces)
        assert response.status_code == 201

        data = response.json()
        assert data["created_count"] == 1
        assert len(data["traces"]) == 1

    @pytest.mark.asyncio
    async def test_create_traces_batch_empty(self, client: AsyncClient):
        """Test that empty batch is rejected."""
        response = await client.post("/api/traces/batch", json=[])
        assert response.status_code == 400
        assert "at least one" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_traces_batch_exceeds_max_size(self, client: AsyncClient):
        """Test that batch exceeding max size is rejected."""
        # Create a batch with 1001 traces (exceeds limit of 1000)
        traces = [
            {
                "trace_version": "1.0",
                "prompt": f"Q{i}",
                "final_answer": f"A{i}",
                "steps": [{"i": 0, "type": "t", "content": "c"}],
            }
            for i in range(1001)
        ]

        response = await client.post("/api/traces/batch", json=traces)
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_traces_batch_invalid_trace(self, client: AsyncClient):
        """Test that batch with invalid trace is rejected."""
        traces = [
            {
                "trace_version": "1.0",
                "prompt": "Valid",
                "final_answer": "Valid",
                "steps": [{"i": 0, "type": "test", "content": "Valid"}],
            },
            {
                # Invalid: missing required field 'final_answer'
                "trace_version": "1.0",
                "prompt": "Invalid",
                "steps": [{"i": 0, "type": "test", "content": "Invalid"}],
            },
        ]

        response = await client.post("/api/traces/batch", json=traces)
        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_create_traces_batch_with_metadata(self, client: AsyncClient):
        """Test batch creation with trace metadata."""
        traces = [
            {
                "trace_version": "1.0",
                "prompt": f"Q{i}",
                "final_answer": f"A{i}",
                "steps": [{"i": 0, "type": "test", "content": f"S{i}"}],
                "meta": {"source": "eval", "run_id": "test-run"},
            }
            for i in range(3)
        ]

        response = await client.post("/api/traces/batch", json=traces)
        assert response.status_code == 201

        data = response.json()
        assert data["created_count"] == 3

        # Verify traces were actually created by fetching one
        first_trace_id = data["traces"][0]["id"]
        get_response = await client.get(f"/api/traces/{first_trace_id}")
        assert get_response.status_code == 200

        trace_data = get_response.json()
        assert trace_data["payload"]["meta"]["source"] == "eval"
        assert trace_data["payload"]["meta"]["run_id"] == "test-run"

    @pytest.mark.asyncio
    async def test_create_traces_batch_transaction_isolation(self, client: AsyncClient):
        """Test that batch creation is transactional (all-or-nothing)."""
        # This test creates a batch with one invalid trace in the middle
        # The entire batch should be rejected
        traces = [
            {
                "trace_version": "1.0",
                "prompt": "Valid 1",
                "final_answer": "Answer 1",
                "steps": [{"i": 0, "type": "test", "content": "Step 1"}],
            },
            {
                # Invalid: empty prompt
                "trace_version": "1.0",
                "prompt": "",
                "final_answer": "Answer 2",
                "steps": [{"i": 0, "type": "test", "content": "Step 2"}],
            },
            {
                "trace_version": "1.0",
                "prompt": "Valid 3",
                "final_answer": "Answer 3",
                "steps": [{"i": 0, "type": "test", "content": "Step 3"}],
            },
        ]

        response = await client.post("/api/traces/batch", json=traces)
        assert response.status_code == 422  # Validation fails

        # Verify no traces were created
        list_response = await client.get("/api/traces")
        assert list_response.status_code == 200
        assert len(list_response.json()["data"]) == 0
