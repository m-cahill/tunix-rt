"""Tunix dataset export service.

This module provides business logic for exporting traces in Tunix-compatible
formats. M12 reuses the existing tunix_sft export format from M09.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.helpers.datasets import load_manifest
from tunix_rt_backend.schemas import TunixExportRequest
from tunix_rt_backend.services.datasets_export import export_dataset_to_jsonl


async def export_tunix_sft_jsonl(
    request: TunixExportRequest,
    db: AsyncSession,
) -> str:
    """Export traces in Tunix SFT format (JSONL).

    This function reuses the existing tunix_sft export format from M09.
    It supports two modes:
    1. Export from a dataset key (uses dataset manifest)
    2. Export from specific trace IDs

    Args:
        request: Export request with dataset_key OR trace_ids
        db: Database session

    Returns:
        JSONL content string (one JSON object per line)

    Raises:
        ValueError: If neither dataset_key nor trace_ids provided
        FileNotFoundError: If dataset_key doesn't exist

    Note:
        The tunix_sft format uses Gemma chat templates with reasoning steps,
        as defined in M09. See docs/M09_DATASET_FORMAT.md for details.
    """
    # Mode 1: Export from dataset manifest
    if request.dataset_key:
        manifest = load_manifest(request.dataset_key)
        # Use existing dataset export service (delegates to datasets_export.py)
        jsonl_content = await export_dataset_to_jsonl(
            manifest=manifest,
            db=db,
            format="tunix_sft",  # Reuse M09 tunix_sft format
        )
        return jsonl_content

    # Mode 2: Export from specific trace IDs
    if request.trace_ids:
        # Parse trace IDs
        trace_uuids = [uuid.UUID(tid.strip()) for tid in request.trace_ids]

        # Create a minimal "manifest" structure for export_dataset_to_jsonl
        # This reuses the existing export logic without duplicating code
        from datetime import UTC, datetime

        from tunix_rt_backend.schemas import DatasetManifest

        # Build temporary manifest
        temp_manifest = DatasetManifest(
            dataset_key="temp_export",
            build_id=uuid.uuid4(),
            dataset_name="temp",
            dataset_version="export",
            created_at=datetime.now(UTC),
            selection_strategy="latest",
            trace_ids=trace_uuids,
            trace_count=len(trace_uuids),
        )

        # Export using existing service
        jsonl_content = await export_dataset_to_jsonl(
            manifest=temp_manifest,
            db=db,
            format="tunix_sft",
        )
        return jsonl_content

    # Neither dataset_key nor trace_ids provided
    raise ValueError("Either dataset_key or trace_ids must be provided")
