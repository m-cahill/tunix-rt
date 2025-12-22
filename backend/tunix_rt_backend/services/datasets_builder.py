"""Dataset building service.

This module provides business logic for creating dataset manifests from traces.
"""

import random
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.helpers.datasets import (
    compute_dataset_stats,
    create_dataset_key,
    save_manifest,
)
from tunix_rt_backend.schemas.dataset import DatasetBuildRequest, DatasetManifest


async def build_dataset_manifest(
    request: DatasetBuildRequest,
    db: AsyncSession,
) -> tuple[str, uuid.UUID, int, Path]:
    """Build a dataset manifest from traces in the database.
    
    Args:
        request: Dataset build parameters (filters, selection strategy, etc.)
        db: Database session
    
    Returns:
        Tuple of (dataset_key, build_id, trace_count, manifest_path)
    
    Raises:
        ValueError: If random strategy is used without a seed
    """
    # Validate random strategy requires seed
    if request.selection_strategy == "random" and request.seed is None:
        raise ValueError("Random selection strategy requires a seed for reproducibility")

    # Create dataset key
    dataset_key = create_dataset_key(request.dataset_name, request.dataset_version)

    # Build query based on filters
    query = select(Trace).order_by(Trace.created_at.desc())

    # Apply filters (e.g., source=ungar)
    # For now, filters are applied at Python level for DB compatibility
    # Future: Use DB-specific JSON queries for better performance
    result = await db.execute(query.limit(request.limit * 10))  # Fetch more for filtering
    all_traces = result.scalars().all()

    # Filter traces based on request filters
    filtered_traces = []
    for trace in all_traces:
        payload = trace.payload
        meta = payload.get("meta", {})

        # Check if trace matches all filters
        matches = True
        for key, value in request.filters.items():
            if meta.get(key) != value:
                matches = False
                break

        if matches:
            filtered_traces.append(trace)

    # Apply selection strategy
    if request.selection_strategy == "latest":
        # Already sorted by created_at desc, just take first N
        selected_traces = filtered_traces[: request.limit]
    elif request.selection_strategy == "random":
        # Random selection with seed
        random.seed(request.seed)
        selected_traces = random.sample(filtered_traces, min(len(filtered_traces), request.limit))
    else:
        # Should never happen due to Pydantic validation
        selected_traces = filtered_traces[: request.limit]

    # Extract trace IDs and payloads
    trace_ids = [trace.id for trace in selected_traces]
    trace_payloads = [trace.payload for trace in selected_traces]

    # Compute stats
    stats = compute_dataset_stats(trace_payloads)

    # Create manifest
    build_id = uuid.uuid4()
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        build_id=build_id,
        dataset_name=request.dataset_name,
        dataset_version=request.dataset_version,
        dataset_schema_version="1.0",
        created_at=datetime.now(UTC),
        filters=request.filters,
        selection_strategy=request.selection_strategy,
        seed=request.seed,
        trace_ids=trace_ids,
        trace_count=len(trace_ids),
        stats=stats,
        session_id=request.session_id,
        parent_dataset_id=request.parent_dataset_id,
        training_run_id=request.training_run_id,
    )

    # Save manifest to disk
    manifest_path = save_manifest(manifest)

    return dataset_key, build_id, len(trace_ids), manifest_path

