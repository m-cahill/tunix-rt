"""Tests for helper functions."""

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.db.base import Base
from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.helpers.traces import get_trace_or_404

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


@pytest.mark.asyncio
async def test_get_trace_or_404_success(test_db: AsyncSession) -> None:
    """Test get_trace_or_404 returns trace when it exists."""
    # Create a test trace
    trace = Trace(
        trace_version="1.0",
        payload={"prompt": "test", "final_answer": "answer", "steps": []},
    )
    test_db.add(trace)
    await test_db.commit()
    await test_db.refresh(trace)

    # Fetch it using helper
    result = await get_trace_or_404(test_db, trace.id)

    assert result.id == trace.id
    assert result.trace_version == "1.0"
    assert result.payload["prompt"] == "test"


@pytest.mark.asyncio
async def test_get_trace_or_404_not_found(test_db: AsyncSession) -> None:
    """Test get_trace_or_404 raises 404 when trace doesn't exist."""
    random_id = uuid.uuid4()

    with pytest.raises(HTTPException) as exc_info:
        await get_trace_or_404(test_db, random_id)

    assert exc_info.value.status_code == 404
    assert f"Trace {random_id} not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_trace_or_404_with_label(test_db: AsyncSession) -> None:
    """Test get_trace_or_404 uses label in error message when provided."""
    random_id = uuid.uuid4()

    with pytest.raises(HTTPException) as exc_info:
        await get_trace_or_404(test_db, random_id, label="Base")

    assert exc_info.value.status_code == 404
    assert f"Base trace {random_id} not found" in exc_info.value.detail
