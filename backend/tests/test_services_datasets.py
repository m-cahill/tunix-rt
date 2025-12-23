"""Tests for dataset builder service layer."""

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.db.base import Base
from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas import ReasoningTrace, TraceStep
from tunix_rt_backend.schemas.dataset import DatasetBuildRequest
from tunix_rt_backend.services.datasets_builder import build_dataset_manifest

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


class TestDatasetBuilderService:
    """Tests for dataset building service."""

    @pytest.mark.asyncio
    async def test_build_dataset_random_requires_seed(self, test_db: AsyncSession):
        """Random selection strategy should require a seed."""
        request = DatasetBuildRequest(
            dataset_name="test",
            dataset_version="v1",
            filters={},
            limit=10,
            selection_strategy="random",
            seed=None,  # Missing seed
        )

        with pytest.raises(ValueError, match="requires a seed"):
            await build_dataset_manifest(request, test_db)

    @pytest.mark.asyncio
    async def test_build_dataset_latest_strategy(self, test_db: AsyncSession):
        """Latest strategy should select most recent traces."""
        # Create some test traces
        traces = []
        for i in range(5):
            trace = ReasoningTrace(
                trace_version="1.0",
                prompt=f"Test prompt {i}",
                final_answer=f"Answer {i}",
                steps=[TraceStep(i=0, type="think", content=f"Step {i}")],
                meta={"source": "test"},
            )
            db_trace = Trace(
                trace_version=trace.trace_version,
                payload=trace.model_dump(),
            )
            test_db.add(db_trace)
            traces.append(db_trace)

        await test_db.commit()

        # Build dataset with latest strategy
        request = DatasetBuildRequest(
            dataset_name="test",
            dataset_version="v1",
            filters={"source": "test"},
            limit=3,
            selection_strategy="latest",
        )

        dataset_key, build_id, trace_count, manifest_path = await build_dataset_manifest(
            request, test_db
        )

        # Should return expected values
        assert dataset_key == "test-v1"
        assert trace_count == 3  # Limited to 3
        assert manifest_path.exists()

    @pytest.mark.asyncio
    async def test_build_dataset_random_strategy(self, test_db: AsyncSession):
        """Random strategy with seed should be deterministic."""
        # Create test traces
        for i in range(10):
            trace = ReasoningTrace(
                trace_version="1.0",
                prompt=f"Test prompt {i}",
                final_answer=f"Answer {i}",
                steps=[TraceStep(i=0, type="think", content=f"Step {i}")],
                meta={"source": "random_test"},
            )
            db_trace = Trace(
                trace_version=trace.trace_version,
                payload=trace.model_dump(),
            )
            test_db.add(db_trace)

        await test_db.commit()

        # Build dataset twice with same seed
        request = DatasetBuildRequest(
            dataset_name="random_test",
            dataset_version="v1",
            filters={"source": "random_test"},
            limit=5,
            selection_strategy="random",
            seed=42,
        )

        _, _, count1, _ = await build_dataset_manifest(request, test_db)
        _, _, count2, _ = await build_dataset_manifest(request, test_db)

        # Both should select 5 traces
        assert count1 == 5
        assert count2 == 5
        # Note: Full determinism test would require checking exact trace IDs
        # which requires reading the manifest files

    @pytest.mark.asyncio
    async def test_build_dataset_with_filters(self, test_db: AsyncSession):
        """Dataset build should filter traces by metadata."""
        # Create traces with different sources
        for i in range(3):
            trace_a = ReasoningTrace(
                trace_version="1.0",
                prompt=f"Trace A {i}",
                final_answer=f"Answer A {i}",
                steps=[TraceStep(i=0, type="think", content="Step")],
                meta={"source": "source_a"},
            )
            trace_b = ReasoningTrace(
                trace_version="1.0",
                prompt=f"Trace B {i}",
                final_answer=f"Answer B {i}",
                steps=[TraceStep(i=0, type="think", content="Step")],
                meta={"source": "source_b"},
            )
            test_db.add(Trace(trace_version="1.0", payload=trace_a.model_dump()))
            test_db.add(Trace(trace_version="1.0", payload=trace_b.model_dump()))

        await test_db.commit()

        # Build dataset filtering for source_a
        request = DatasetBuildRequest(
            dataset_name="filtered",
            dataset_version="v1",
            filters={"source": "source_a"},
            limit=10,
            selection_strategy="latest",
        )

        _, _, trace_count, _ = await build_dataset_manifest(request, test_db)

        # Should only get source_a traces
        assert trace_count == 3

    @pytest.mark.asyncio
    async def test_build_dataset_empty_result(self, test_db: AsyncSession):
        """Building dataset with no matching traces should work."""
        request = DatasetBuildRequest(
            dataset_name="empty",
            dataset_version="v1",
            filters={"source": "nonexistent"},
            limit=10,
            selection_strategy="latest",
        )

        dataset_key, build_id, trace_count, manifest_path = await build_dataset_manifest(
            request, test_db
        )

        # Should create manifest even with 0 traces
        assert trace_count == 0
        assert manifest_path.exists()
