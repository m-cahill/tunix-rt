"""Tests for trace scoring endpoints and logic."""

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.app import app
from tunix_rt_backend.db.base import Base, get_db
from tunix_rt_backend.schemas import ReasoningTrace, TraceStep
from tunix_rt_backend.scoring import baseline_score

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
async def async_client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
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


class TestBaselineScorer:
    """Tests for baseline scoring logic."""

    def test_baseline_score_minimal_trace(self):
        """Test baseline scorer with minimal valid trace (1 step, short content)."""
        trace = ReasoningTrace(
            trace_version="1.0",
            prompt="Test prompt",
            final_answer="Test answer",
            steps=[
                TraceStep(i=0, type="test", content="Short step"),
            ],
        )

        score, details = baseline_score(trace)

        # 1 step: 1/10 * 50 = 5.0
        # 10 chars: 10/500 * 50 = 1.0
        # Total: 6.0
        assert score == pytest.approx(6.0, abs=0.1)
        assert details.step_count == 1
        assert details.avg_step_length == 10.0
        assert details.total_chars == 10
        assert details.step_score == pytest.approx(5.0, abs=0.1)
        assert details.length_score == pytest.approx(1.0, abs=0.1)
        assert details.criteria == "baseline"

    def test_baseline_score_ideal_trace(self):
        """Test baseline scorer with ideal trace (10 steps, 500 chars each)."""
        steps = [TraceStep(i=i, type="step", content="a" * 500) for i in range(10)]
        trace = ReasoningTrace(
            trace_version="1.0",
            prompt="Test prompt",
            final_answer="Test answer",
            steps=steps,
        )

        score, details = baseline_score(trace)

        # 10 steps: 10/10 * 50 = 50.0
        # 500 chars: 500/500 * 50 = 50.0
        # Total: 100.0
        assert score == 100.0
        assert details.step_count == 10
        assert details.avg_step_length == 500.0
        assert details.total_chars == 5000
        assert details.step_score == 50.0
        assert details.length_score == 50.0

    def test_baseline_score_capped_values(self):
        """Test that scores cap at 50 for each component."""
        # More than 10 steps, more than 500 chars average
        steps = [TraceStep(i=i, type="step", content="a" * 1000) for i in range(20)]
        trace = ReasoningTrace(
            trace_version="1.0",
            prompt="Test prompt",
            final_answer="Test answer",
            steps=steps,
        )

        score, details = baseline_score(trace)

        # Both components should cap at 50
        assert score == 100.0
        assert details.step_score == 50.0
        assert details.length_score == 50.0

    def test_baseline_score_varied_step_lengths(self):
        """Test scorer with varied step lengths."""
        trace = ReasoningTrace(
            trace_version="1.0",
            prompt="Test prompt",
            final_answer="Test answer",
            steps=[
                TraceStep(i=0, type="short", content="Short"),
                TraceStep(i=1, type="medium", content="a" * 100),
                TraceStep(i=2, type="long", content="a" * 400),
            ],
        )

        score, details = baseline_score(trace)

        # 3 steps: 3/10 * 50 = 15.0
        # Avg length: (5 + 100 + 400) / 3 = 168.33
        # Length score: 168.33/500 * 50 = 16.83
        # Total: ~31.83
        assert details.step_count == 3
        assert details.avg_step_length == pytest.approx(168.33, abs=0.1)
        assert score == pytest.approx(31.83, abs=0.1)


class TestScoreEndpoint:
    """Tests for POST /api/traces/{id}/score endpoint."""

    @pytest.mark.asyncio
    async def test_score_trace_success(self, async_client: AsyncClient):
        """Test successfully scoring a trace."""
        # Create a trace first
        trace_data = {
            "trace_version": "1.0",
            "prompt": "What is 27 × 19?",
            "final_answer": "513",
            "steps": [
                {"i": 0, "type": "parse", "content": "Parse the multiplication task"},
                {"i": 1, "type": "compute", "content": "Break down: 27 × 19 = 27 × (20 - 1)"},
                {"i": 2, "type": "result", "content": "Final: 513"},
            ],
            "meta": {"source": "test"},
        }
        create_response = await async_client.post("/api/traces", json=trace_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        trace_id = create_response.json()["id"]

        # Score the trace
        score_response = await async_client.post(
            f"/api/traces/{trace_id}/score",
            json={"criteria": "baseline"},
        )

        assert score_response.status_code == status.HTTP_201_CREATED
        score_data = score_response.json()
        assert score_data["trace_id"] == trace_id
        assert "score" in score_data
        assert 0 <= score_data["score"] <= 100
        assert "details" in score_data
        assert score_data["details"]["step_count"] == 3
        assert score_data["details"]["criteria"] == "baseline"

    @pytest.mark.asyncio
    async def test_score_trace_not_found(self, async_client: AsyncClient):
        """Test scoring a non-existent trace."""
        fake_id = str(uuid.uuid4())
        response = await async_client.post(
            f"/api/traces/{fake_id}/score",
            json={"criteria": "baseline"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_score_trace_default_criteria(self, async_client: AsyncClient):
        """Test that baseline is the default criteria."""
        # Create a trace first
        trace_data = {
            "trace_version": "1.0",
            "prompt": "Test",
            "final_answer": "Answer",
            "steps": [
                {"i": 0, "type": "test", "content": "Content"},
            ],
        }
        create_response = await async_client.post("/api/traces", json=trace_data)
        trace_id = create_response.json()["id"]

        # Score without specifying criteria (should default to baseline)
        score_response = await async_client.post(
            f"/api/traces/{trace_id}/score",
            json={},
        )

        assert score_response.status_code == status.HTTP_201_CREATED
        score_data = score_response.json()
        assert score_data["details"]["criteria"] == "baseline"


class TestCompareEndpoint:
    """Tests for GET /api/traces/compare endpoint."""

    @pytest.mark.asyncio
    async def test_compare_traces_success(self, async_client: AsyncClient):
        """Test successfully comparing two traces."""
        # Create two distinct traces
        trace1_data = {
            "trace_version": "1.0",
            "prompt": "Simple task",
            "final_answer": "Simple answer",
            "steps": [
                {"i": 0, "type": "think", "content": "Short reasoning"},
            ],
        }
        trace2_data = {
            "trace_version": "1.0",
            "prompt": "Complex task requiring detailed reasoning",
            "final_answer": "Detailed answer with extensive explanation",
            "steps": [
                {"i": 0, "type": "analyze", "content": "a" * 200},
                {"i": 1, "type": "compute", "content": "a" * 300},
                {"i": 2, "type": "verify", "content": "a" * 250},
                {"i": 3, "type": "conclude", "content": "a" * 150},
            ],
        }

        response1 = await async_client.post("/api/traces", json=trace1_data)
        response2 = await async_client.post("/api/traces", json=trace2_data)

        trace1_id = response1.json()["id"]
        trace2_id = response2.json()["id"]

        # Compare the traces
        compare_response = await async_client.get(
            f"/api/traces/compare?base={trace1_id}&other={trace2_id}"
        )

        assert compare_response.status_code == status.HTTP_200_OK
        compare_data = compare_response.json()

        # Validate structure
        assert "base" in compare_data
        assert "other" in compare_data

        # Validate base trace
        assert compare_data["base"]["id"] == trace1_id
        assert "score" in compare_data["base"]
        assert "payload" in compare_data["base"]
        assert compare_data["base"]["payload"]["prompt"] == "Simple task"

        # Validate other trace
        assert compare_data["other"]["id"] == trace2_id
        assert "score" in compare_data["other"]
        assert "payload" in compare_data["other"]
        prompt = "Complex task requiring detailed reasoning"
        assert compare_data["other"]["payload"]["prompt"] == prompt

        # Trace 2 should have higher score (more steps, longer content)
        assert compare_data["other"]["score"] > compare_data["base"]["score"]

    @pytest.mark.asyncio
    async def test_compare_base_not_found(self, async_client: AsyncClient):
        """Test comparison when base trace doesn't exist."""
        # Create one valid trace
        trace_data = {
            "trace_version": "1.0",
            "prompt": "Test",
            "final_answer": "Answer",
            "steps": [{"i": 0, "type": "test", "content": "Content"}],
        }
        response = await async_client.post("/api/traces", json=trace_data)
        valid_id = response.json()["id"]
        fake_id = str(uuid.uuid4())

        compare_response = await async_client.get(
            f"/api/traces/compare?base={fake_id}&other={valid_id}"
        )

        assert compare_response.status_code == status.HTTP_404_NOT_FOUND
        assert "Base trace" in compare_response.json()["detail"]

    @pytest.mark.asyncio
    async def test_compare_other_not_found(self, async_client: AsyncClient):
        """Test comparison when other trace doesn't exist."""
        # Create one valid trace
        trace_data = {
            "trace_version": "1.0",
            "prompt": "Test",
            "final_answer": "Answer",
            "steps": [{"i": 0, "type": "test", "content": "Content"}],
        }
        response = await async_client.post("/api/traces", json=trace_data)
        valid_id = response.json()["id"]
        fake_id = str(uuid.uuid4())

        compare_response = await async_client.get(
            f"/api/traces/compare?base={valid_id}&other={fake_id}"
        )

        assert compare_response.status_code == status.HTTP_404_NOT_FOUND
        assert "Other trace" in compare_response.json()["detail"]

    @pytest.mark.asyncio
    async def test_compare_both_not_found(self, async_client: AsyncClient):
        """Test comparison when both traces don't exist."""
        fake_id1 = str(uuid.uuid4())
        fake_id2 = str(uuid.uuid4())

        compare_response = await async_client.get(
            f"/api/traces/compare?base={fake_id1}&other={fake_id2}"
        )

        assert compare_response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_compare_same_trace_twice(self, async_client: AsyncClient):
        """Test comparing a trace with itself (should work)."""
        trace_data = {
            "trace_version": "1.0",
            "prompt": "Test",
            "final_answer": "Answer",
            "steps": [{"i": 0, "type": "test", "content": "Content"}],
        }
        response = await async_client.post("/api/traces", json=trace_data)
        trace_id = response.json()["id"]

        compare_response = await async_client.get(
            f"/api/traces/compare?base={trace_id}&other={trace_id}"
        )

        assert compare_response.status_code == status.HTTP_200_OK
        compare_data = compare_response.json()
        assert compare_data["base"]["id"] == trace_id
        assert compare_data["other"]["id"] == trace_id
        assert compare_data["base"]["score"] == compare_data["other"]["score"]
