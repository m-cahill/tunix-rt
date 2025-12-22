"""Dataset export service.

This module handles dataset export formatting logic for different output formats.
"""

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.models import Trace
from tunix_rt_backend.schemas import ExportFormat, ReasoningTrace
from tunix_rt_backend.schemas.dataset import DatasetManifest


async def export_dataset_to_jsonl(
    manifest: DatasetManifest,
    db: AsyncSession,
    format: ExportFormat,
) -> str:
    """Export a dataset as JSONL content.

    This service function handles:
    - Fetching traces by manifest IDs
    - Format-specific record construction
    - JSONL serialization

    Args:
        manifest: Dataset manifest with trace_ids
        db: Database session
        format: Export format ('trace', 'tunix_sft', or 'training_example')

    Returns:
        NDJSON content string (one JSON object per line)

    Note:
        Output maintains manifest order (deterministic).
        Traces deleted after manifest creation are skipped.
    """
    # Fetch traces by IDs from manifest
    result = await db.execute(select(Trace).where(Trace.id.in_(manifest.trace_ids)))
    db_traces = result.scalars().all()

    # Create a mapping for fast lookup
    trace_map: dict[uuid.UUID, Trace] = {trace.id: trace for trace in db_traces}

    # Build JSONL output in manifest order
    lines = []
    for trace_id in manifest.trace_ids:
        trace = trace_map.get(trace_id)
        if not trace:
            # Trace was deleted after manifest was created; skip it
            continue

        trace_payload = ReasoningTrace(**trace.payload)

        # Build record based on format
        if format == "trace":
            record = _build_trace_record(trace, trace_payload)
        elif format == "tunix_sft":
            record = _build_tunix_sft_record(trace, trace_payload)
        else:  # format == "training_example"
            record = _build_training_example_record(trace, trace_payload)

        lines.append(json.dumps(record))

    # Return as NDJSON
    return "\n".join(lines) + "\n" if lines else ""


def _build_trace_record(trace: Trace, trace_payload: ReasoningTrace) -> dict[str, Any]:
    """Build trace format record (raw trace data).

    Args:
        trace: Database trace model
        trace_payload: Parsed ReasoningTrace payload

    Returns:
        Dict with trace data for JSONL serialization
    """
    return {
        "id": str(trace.id),
        "prompts": trace_payload.prompt,  # Use 'prompts' for Tunix compatibility
        "trace_steps": [step.content for step in trace_payload.steps],
        "final_answer": trace_payload.final_answer,
        "metadata": {
            "created_at": trace.created_at.isoformat(),
            "trace_version": trace.trace_version,
            **(trace_payload.meta or {}),
        },
    }


def _build_tunix_sft_record(trace: Trace, trace_payload: ReasoningTrace) -> dict[str, Any]:
    """Build Tunix SFT format record (with rendered prompts).

    Args:
        trace: Database trace model
        trace_payload: Parsed ReasoningTrace payload

    Returns:
        Dict with SFT-formatted data for JSONL serialization
    """
    from tunix_rt_backend.training.renderers import render_tunix_sft_prompt

    # Build SFT-formatted record with rendered prompt
    trace_data = {
        "prompt": trace_payload.prompt,
        "trace_steps": [step.content for step in trace_payload.steps],
        "final_answer": trace_payload.final_answer,
    }
    rendered_prompt = render_tunix_sft_prompt(trace_data)

    return {
        "id": str(trace.id),
        "prompts": rendered_prompt,  # Rendered Tunix SFT prompt
        "final_answer": trace_payload.final_answer,
        "metadata": {
            "created_at": trace.created_at.isoformat(),
            "trace_version": trace.trace_version,
            "format": "tunix_sft",
            **(trace_payload.meta or {}),
        },
    }


def _build_training_example_record(trace: Trace, trace_payload: ReasoningTrace) -> dict[str, Any]:
    """Build TrainingExample format record (prompt/response pairs).

    Args:
        trace: Database trace model
        trace_payload: Parsed ReasoningTrace payload

    Returns:
        Dict with TrainingExample data for JSONL serialization
    """
    from tunix_rt_backend.training.schema import TrainingExample

    # Prompt: original question with instruction
    prompt = f"{trace_payload.prompt}\n\nPlease show your reasoning steps."

    # Response: reasoning steps + final answer
    reasoning_steps = [step.content for step in trace_payload.steps]
    response_parts = []
    if reasoning_steps:
        response_parts.append("Reasoning:")
        for i, step in enumerate(reasoning_steps, 1):
            response_parts.append(f"{i}. {step}")
    response_parts.append(f"Answer: {trace_payload.final_answer}")
    response = "\n".join(response_parts)

    # Create TrainingExample
    example = TrainingExample(
        prompt=prompt,
        response=response,
        metadata={
            "source_trace_id": str(trace.id),
            "created_at": trace.created_at.isoformat(),
            "trace_version": trace.trace_version,
            **(trace_payload.meta or {}),
        },
    )

    return example.model_dump(mode="json")
