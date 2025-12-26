"""Dataset ingestion service.

This module provides business logic for ingesting traces from JSONL files.
"""

import json
import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.schemas import ReasoningTrace
from tunix_rt_backend.services.traces_batch import create_traces_batch

logger = logging.getLogger(__name__)


async def ingest_jsonl_dataset(
    file_path: str,
    source_name: str,
    db: AsyncSession,
) -> tuple[int, list[uuid.UUID]]:
    """Ingest traces from a JSONL file into the database.

    Args:
        file_path: Path to JSONL file (server-side path)
        source_name: Source name to tag traces with (e.g., 'external_import', 'competition_data')
        db: Database session

    Returns:
        Tuple of (ingested_count, trace_ids)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or contains invalid traces
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Read and parse JSONL
    traces = []
    line_num = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                trace_data = json.loads(line)

                # Add source metadata if not present
                if "meta" not in trace_data:
                    trace_data["meta"] = {}
                trace_data["meta"]["source"] = source_name
                trace_data["meta"]["ingest_source_file"] = str(path.name)

                # Validate with Pydantic
                trace = ReasoningTrace(**trace_data)
                traces.append(trace)

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Skipping invalid trace at line {line_num}: {e}")
                continue

    if not traces:
        raise ValueError(f"No valid traces found in {file_path}")

    # Batch create traces
    result = await create_traces_batch(traces, db)

    return result.created_count, [t.id for t in result.traces]
