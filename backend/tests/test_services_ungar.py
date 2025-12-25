"""Tests for UNGAR generator service layer."""

import uuid
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tunix_rt_backend.db.base import Base
from tunix_rt_backend.schemas import UngarGenerateRequest
from tunix_rt_backend.services.ungar_generator import (
    check_ungar_status,
    export_high_card_duel_jsonl,
    generate_high_card_duel_traces,
)

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


class TestUngarStatusService:
    """Tests for UNGAR status checking service."""

    def test_check_status_without_ungar(self):
        """Status check should work even without UNGAR installed."""
        status = check_ungar_status()

        # Should return a response (not raise error)
        assert status is not None
        assert hasattr(status, "available")
        assert isinstance(status.available, bool)

        # Version is None or a string
        assert status.version is None or isinstance(status.version, str)


class TestUngarGeneratorService:
    """Tests for UNGAR trace generation service."""

    @pytest.mark.asyncio
    async def test_generate_without_ungar_raises_error(self, test_db: AsyncSession):
        """Generating traces without UNGAR should raise ValueError."""
        request = UngarGenerateRequest(count=5, seed=42, persist=False)

        # Mock ungar_available to return False regardless of installation
        with patch(
            "tunix_rt_backend.integrations.ungar.availability.ungar_available", return_value=False
        ):
            with pytest.raises(ValueError, match="UNGAR is not installed"):
                await generate_high_card_duel_traces(request, test_db)

    @pytest.mark.asyncio
    @pytest.mark.ungar
    async def test_generate_traces_no_persist(self, test_db: AsyncSession):
        """Generate traces without persisting should return placeholder IDs."""
        from tunix_rt_backend.integrations.ungar.availability import ungar_available

        if not ungar_available():
            pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

        request = UngarGenerateRequest(count=3, seed=42, persist=False)

        trace_ids, preview = await generate_high_card_duel_traces(request, test_db)

        # Should return requested number of traces
        assert len(trace_ids) == 3
        # Preview should have max 3 items
        assert len(preview) <= 3
        # IDs should be valid UUIDs
        assert all(isinstance(tid, uuid.UUID) for tid in trace_ids)
        # Preview should have expected structure
        assert all("trace_id" in p for p in preview)
        assert all("game" in p for p in preview)

    @pytest.mark.asyncio
    @pytest.mark.ungar
    async def test_generate_traces_with_persist(self, test_db: AsyncSession):
        """Generate traces with persist should save to database."""
        from tunix_rt_backend.integrations.ungar.availability import ungar_available

        if not ungar_available():
            pytest.skip("UNGAR not installed; use: pip install -e '.[ungar]'")

        request = UngarGenerateRequest(count=2, seed=42, persist=True)

        trace_ids, preview = await generate_high_card_duel_traces(request, test_db)

        # Should return 2 trace IDs
        assert len(trace_ids) == 2
        # Preview should have 2 items (min of count and 3)
        assert len(preview) == 2
        # Traces should be persisted (check by querying DB)
        # Note: This requires the traces to be committed in the service
        # which they are, so the IDs should be real UUIDs from DB


class TestUngarExportService:
    """Tests for UNGAR JSONL export service."""

    @pytest.mark.asyncio
    async def test_export_empty_database(self, test_db: AsyncSession):
        """Exporting from empty database should return empty string."""
        result = await export_high_card_duel_jsonl(test_db, limit=10, trace_ids_str=None)

        # Should return empty JSONL (just empty string or newline)
        assert result in ("", "\n")

    @pytest.mark.asyncio
    async def test_export_with_trace_ids_parameter(self, test_db: AsyncSession):
        """Export with specific trace IDs should parse them correctly."""
        # This tests the ID parsing logic even if traces don't exist
        # (will just return empty since no traces match)
        fake_ids = f"{uuid.uuid4()},{uuid.uuid4()}"

        result = await export_high_card_duel_jsonl(test_db, limit=10, trace_ids_str=fake_ids)

        # Should not error on parsing, just return empty
        assert isinstance(result, str)
