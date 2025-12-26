"""Unit tests for datasets_ingest service.

Tests cover:
- Happy path: ingest valid JSONL with multiple traces
- Skip invalid lines: invalid JSON or schema violations are skipped
- File not found: raises FileNotFoundError
- Empty/invalid file: raises ValueError

M32 requirement: Coverage uplift for datasets_ingest.py (was 0%).
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tunix_rt_backend.services.datasets_ingest import ingest_jsonl_dataset


class TestIngestJsonlDataset:
    """Test suite for ingest_jsonl_dataset function."""

    @pytest.fixture
    def valid_trace_data(self) -> dict:
        """Return a valid trace in strict ReasoningTrace format."""
        return {
            "trace_version": "1.0",
            "prompt": "What is 2 + 2?",
            "final_answer": "4",
            "steps": [
                {"i": 0, "type": "calculation", "content": "Adding 2 and 2"},
                {"i": 1, "type": "result", "content": "Result: 4"},
            ],
            "meta": {"source": "test"},
        }

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def mock_batch_result(self) -> MagicMock:
        """Create a mock batch create result."""
        result = MagicMock()
        result.created_count = 2
        result.traces = [
            MagicMock(id=uuid.uuid4()),
            MagicMock(id=uuid.uuid4()),
        ]
        return result

    @pytest.mark.asyncio
    async def test_ingest_valid_jsonl_happy_path(
        self,
        valid_trace_data: dict,
        mock_db: AsyncMock,
        mock_batch_result: MagicMock,
    ) -> None:
        """Test ingesting a valid JSONL file with multiple traces."""
        # Create temp JSONL file with 2 valid traces
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(valid_trace_data) + "\n")
            f.write(json.dumps(valid_trace_data) + "\n")
            temp_path = f.name

        try:
            with patch(
                "tunix_rt_backend.services.datasets_ingest.create_traces_batch",
                new_callable=AsyncMock,
            ) as mock_create:
                mock_create.return_value = mock_batch_result

                count, trace_ids = await ingest_jsonl_dataset(
                    temp_path,
                    "test_source",
                    mock_db,
                )

                # Verify results
                assert count == 2
                assert len(trace_ids) == 2

                # Verify create_traces_batch was called with correct args
                mock_create.assert_called_once()
                call_args = mock_create.call_args
                traces_arg = call_args[0][0]
                db_arg = call_args[0][1]

                assert len(traces_arg) == 2
                assert db_arg == mock_db

                # Verify source metadata was added
                for trace in traces_arg:
                    assert trace.meta["source"] == "test_source"
                    assert "ingest_source_file" in trace.meta

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_ingest_skips_invalid_json_lines(
        self,
        valid_trace_data: dict,
        mock_db: AsyncMock,
        mock_batch_result: MagicMock,
    ) -> None:
        """Test that invalid JSON lines are skipped with warning."""
        # Adjust mock to return 1 trace (only valid one)
        mock_batch_result.created_count = 1
        mock_batch_result.traces = [MagicMock(id=uuid.uuid4())]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(valid_trace_data) + "\n")
            f.write("this is not valid json\n")  # Invalid line
            f.write('{"incomplete": true\n')  # Invalid JSON
            temp_path = f.name

        try:
            with patch(
                "tunix_rt_backend.services.datasets_ingest.create_traces_batch",
                new_callable=AsyncMock,
            ) as mock_create:
                mock_create.return_value = mock_batch_result

                count, trace_ids = await ingest_jsonl_dataset(
                    temp_path,
                    "test_source",
                    mock_db,
                )

                # Only 1 valid trace should be processed
                assert count == 1
                assert len(trace_ids) == 1

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_ingest_skips_invalid_trace_schema(
        self,
        valid_trace_data: dict,
        mock_db: AsyncMock,
        mock_batch_result: MagicMock,
    ) -> None:
        """Test that traces failing schema validation are skipped."""
        mock_batch_result.created_count = 1
        mock_batch_result.traces = [MagicMock(id=uuid.uuid4())]

        invalid_trace = {
            "trace_version": "1.0",
            "prompt": "test",
            # Missing required fields: final_answer, steps
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(valid_trace_data) + "\n")
            f.write(json.dumps(invalid_trace) + "\n")
            temp_path = f.name

        try:
            with patch(
                "tunix_rt_backend.services.datasets_ingest.create_traces_batch",
                new_callable=AsyncMock,
            ) as mock_create:
                mock_create.return_value = mock_batch_result

                count, trace_ids = await ingest_jsonl_dataset(
                    temp_path,
                    "test_source",
                    mock_db,
                )

                # Only 1 valid trace should be processed
                assert count == 1

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_ingest_file_not_found(self, mock_db: AsyncMock) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            await ingest_jsonl_dataset(
                "/nonexistent/path/to/file.jsonl",
                "test_source",
                mock_db,
            )

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ingest_path_is_directory(self, mock_db: AsyncMock) -> None:
        """Test that ValueError is raised when path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError) as exc_info:
                await ingest_jsonl_dataset(
                    temp_dir,
                    "test_source",
                    mock_db,
                )

            assert "not a file" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ingest_empty_file_raises_value_error(
        self,
        mock_db: AsyncMock,
    ) -> None:
        """Test that ValueError is raised for empty files."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            # Empty file
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                await ingest_jsonl_dataset(
                    temp_path,
                    "test_source",
                    mock_db,
                )

            assert "no valid traces" in str(exc_info.value).lower()

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_ingest_only_invalid_traces_raises_value_error(
        self,
        mock_db: AsyncMock,
    ) -> None:
        """Test that ValueError is raised when all traces are invalid."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("invalid json\n")
            f.write('{"not": "a trace"}\n')
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                await ingest_jsonl_dataset(
                    temp_path,
                    "test_source",
                    mock_db,
                )

            assert "no valid traces" in str(exc_info.value).lower()

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_ingest_skips_empty_lines(
        self,
        valid_trace_data: dict,
        mock_db: AsyncMock,
        mock_batch_result: MagicMock,
    ) -> None:
        """Test that empty lines are skipped without error."""
        mock_batch_result.created_count = 1
        mock_batch_result.traces = [MagicMock(id=uuid.uuid4())]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("\n")  # Empty line
            f.write(json.dumps(valid_trace_data) + "\n")
            f.write("   \n")  # Whitespace only
            f.write("\n")  # Another empty line
            temp_path = f.name

        try:
            with patch(
                "tunix_rt_backend.services.datasets_ingest.create_traces_batch",
                new_callable=AsyncMock,
            ) as mock_create:
                mock_create.return_value = mock_batch_result

                count, trace_ids = await ingest_jsonl_dataset(
                    temp_path,
                    "test_source",
                    mock_db,
                )

                # 1 valid trace should be processed
                assert count == 1

        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_ingest_adds_source_metadata(
        self,
        valid_trace_data: dict,
        mock_db: AsyncMock,
        mock_batch_result: MagicMock,
    ) -> None:
        """Test that source metadata is added to each trace."""
        mock_batch_result.created_count = 1
        mock_batch_result.traces = [MagicMock(id=uuid.uuid4())]

        # Trace without meta field
        trace_no_meta = {
            "trace_version": "1.0",
            "prompt": "What is 1 + 1?",
            "final_answer": "2",
            "steps": [{"i": 0, "type": "calc", "content": "1 + 1 = 2"}],
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(trace_no_meta) + "\n")
            temp_path = f.name

        try:
            with patch(
                "tunix_rt_backend.services.datasets_ingest.create_traces_batch",
                new_callable=AsyncMock,
            ) as mock_create:
                mock_create.return_value = mock_batch_result

                await ingest_jsonl_dataset(
                    temp_path,
                    "my_custom_source",
                    mock_db,
                )

                # Verify source metadata was added
                call_args = mock_create.call_args
                traces_arg = call_args[0][0]

                assert traces_arg[0].meta["source"] == "my_custom_source"
                assert Path(temp_path).name in traces_arg[0].meta["ingest_source_file"]

        finally:
            Path(temp_path).unlink()
