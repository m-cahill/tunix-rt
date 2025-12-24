"""Tests for service layer functions.

These tests verify the business logic in the services layer.
Integration tests in test_traces_batch.py and test_datasets.py
provide comprehensive coverage of the endpoints that use these services.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas import ReasoningTrace
from tunix_rt_backend.schemas.dataset import DatasetManifest
from tunix_rt_backend.services.datasets_export import export_dataset_to_jsonl
from tunix_rt_backend.services.traces_batch import create_traces_batch

# test_db fixture moved to conftest.py


class TestTracesBatchService:
    """Tests for traces_batch service."""

    @pytest.mark.asyncio
    async def test_create_traces_batch_validates_empty_list(self, test_db: AsyncSession):
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="at least one trace"):
            await create_traces_batch([], test_db)

    @pytest.mark.asyncio
    async def test_create_traces_batch_validates_max_size(self, test_db: AsyncSession):
        """Test that oversized batch raises ValueError."""
        # Create 1001 minimal traces
        traces = [
            ReasoningTrace(
                trace_version="1.0",
                prompt=f"Q{i}",
                final_answer=f"A{i}",
                steps=[{"i": 0, "type": "test", "content": "step"}],
            )
            for i in range(1001)
        ]

        with pytest.raises(ValueError, match="exceeds maximum"):
            await create_traces_batch(traces, test_db)

    @pytest.mark.asyncio
    async def test_create_traces_batch_returns_correct_count(self, test_db: AsyncSession):
        """Test that service returns correct created_count."""
        traces = [
            ReasoningTrace(
                trace_version="1.0",
                prompt=f"Question {i}",
                final_answer=f"Answer {i}",
                steps=[{"i": 0, "type": "test", "content": f"Step {i}"}],
            )
            for i in range(5)
        ]

        result = await create_traces_batch(traces, test_db)

        assert result.created_count == 5
        assert len(result.traces) == 5
        # Verify all traces have UUIDs
        for trace_response in result.traces:
            assert isinstance(trace_response.id, uuid.UUID)
            assert trace_response.created_at is not None


class TestDatasetsExportService:
    """Tests for datasets_export service."""

    @pytest.mark.asyncio
    async def test_export_dataset_skips_deleted_traces(self, test_db: AsyncSession):
        """Test that export skips traces deleted after manifest creation."""
        # Create a trace
        db_trace = Trace(
            trace_version="1.0",
            payload={
                "trace_version": "1.0",
                "prompt": "Test",
                "final_answer": "Answer",
                "steps": [{"i": 0, "type": "test", "content": "step"}],
            },
        )
        test_db.add(db_trace)
        await test_db.commit()
        await test_db.refresh(db_trace)

        trace_id = db_trace.id
        fake_id = uuid.uuid4()  # ID that doesn't exist

        # Create manifest with both real and fake IDs
        manifest = DatasetManifest(
            dataset_key="test-v1",
            build_id=uuid.uuid4(),
            dataset_name="test",
            dataset_version="v1",
            created_at=db_trace.created_at,
            selection_strategy="latest",
            trace_ids=[trace_id, fake_id],  # Real ID first, then non-existent
            trace_count=2,
        )

        # Export should skip the missing trace
        content = await export_dataset_to_jsonl(manifest, test_db, "trace")

        # Should only have 1 line (not 2)
        lines = content.strip().split("\n")
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_export_dataset_maintains_manifest_order(self, test_db: AsyncSession):
        """Test that export maintains manifest order."""
        # Create 3 traces
        trace_ids = []
        for i in range(3):
            db_trace = Trace(
                trace_version="1.0",
                payload={
                    "trace_version": "1.0",
                    "prompt": f"Q{i}",
                    "final_answer": f"A{i}",
                    "steps": [{"i": 0, "type": "test", "content": f"step{i}"}],
                },
            )
            test_db.add(db_trace)
            await test_db.commit()
            await test_db.refresh(db_trace)
            trace_ids.append(db_trace.id)

        # Create manifest with reversed order
        manifest = DatasetManifest(
            dataset_key="order-test-v1",
            build_id=uuid.uuid4(),
            dataset_name="order-test",
            dataset_version="v1",
            created_at=db_trace.created_at,
            selection_strategy="latest",
            trace_ids=list(reversed(trace_ids)),  # Reverse order
            trace_count=3,
        )

        # Export
        content = await export_dataset_to_jsonl(manifest, test_db, "trace")
        lines = content.strip().split("\n")

        # Parse and verify order matches manifest (reversed)
        import json

        records = [json.loads(line) for line in lines]
        assert len(records) == 3

        # Verify order: should be A2, A1, A0 (reversed)
        assert records[0]["prompts"] == "Q2"
        assert records[1]["prompts"] == "Q1"
        assert records[2]["prompts"] == "Q0"
