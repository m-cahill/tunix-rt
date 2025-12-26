"""Dataset management endpoints.

Domain: Dataset lifecycle (build, ingest, export)

Primary endpoints:
- POST /api/datasets/build: Create versioned dataset from filtered traces
- POST /api/datasets/ingest: Import traces from JSONL files with provenance
- GET /api/datasets/{key}/export.jsonl: Export dataset in various formats

Cross-cutting concerns:
- All datasets include provenance metadata for reproducibility
- Supports trace, tunix_sft, and training_example export formats
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.schemas import (
    DatasetBuildRequest,
    DatasetBuildResponse,
    DatasetIngestRequest,
    DatasetIngestResponse,
    ExportFormat,
)
from tunix_rt_backend.services.datasets_export import export_dataset_to_jsonl

router = APIRouter()


@router.post(
    "/api/datasets/build",
    response_model=DatasetBuildResponse,
    status_code=status.HTTP_201_CREATED,
)
async def build_dataset(
    request: DatasetBuildRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DatasetBuildResponse:
    """Build a new dataset from traces.

    Creates a dataset manifest and saves it to disk. The dataset can then be
    exported using the export endpoint.

    Args:
        request: Dataset build parameters
        db: Database session

    Returns:
        Dataset build response with dataset_key and build_id

    Raises:
        HTTPException: 422 if validation fails (e.g., random strategy without seed)
    """
    from tunix_rt_backend.services.datasets_builder import build_dataset_manifest

    # Delegate to service layer
    try:
        (
            dataset_key,
            build_id,
            trace_count,
            manifest_path,
        ) = await build_dataset_manifest(request, db)
    except ValueError as e:
        # Convert service-level ValueError to HTTP 422
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )

    return DatasetBuildResponse(
        dataset_key=dataset_key,
        build_id=build_id,
        trace_count=trace_count,
        manifest_path=str(manifest_path),
    )


@router.post(
    "/api/datasets/ingest",
    response_model=DatasetIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_dataset(
    request: DatasetIngestRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DatasetIngestResponse:
    """Ingest traces from a JSONL file into the database.

    This endpoint reads a JSONL file from the server filesystem, validates each trace,
    and imports them into the traces database. Useful for bulk importing external datasets.

    Args:
        request: Ingest request with file path and source name
        db: Database session

    Returns:
        DatasetIngestResponse with ingested count and trace IDs

    Raises:
        HTTPException: 400 if file not found or contains invalid data
    """
    from tunix_rt_backend.services.datasets_ingest import ingest_jsonl_dataset

    try:
        ingested_count, trace_ids = await ingest_jsonl_dataset(
            request.path,
            request.source_name,
            db,
        )
        return DatasetIngestResponse(
            ingested_count=ingested_count,
            trace_ids=trace_ids,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/api/datasets/{dataset_key}/export.jsonl")
async def export_dataset(
    dataset_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    format: ExportFormat = "trace",
) -> Response:
    """Export a dataset as JSONL.

    Loads the dataset manifest and exports all included traces in JSONL format.
    Supports three formats:
    - 'trace': Raw trace data (default)
    - 'tunix_sft': Formatted for Tunix SFT training with rendered prompts
    - 'training_example': TrainingExample objects (prompt/response pairs)

    Args:
        dataset_key: Dataset identifier (name-version)
        db: Database session
        format: Export format ('trace', 'tunix_sft', or 'training_example', default: 'trace')

    Returns:
        NDJSON response with one trace per line

    Raises:
        HTTPException: 404 if dataset not found
        HTTPException: 422 if format is invalid (FastAPI validates automatically)
    """
    from tunix_rt_backend.helpers.datasets import load_manifest

    # Load manifest
    try:
        manifest = load_manifest(dataset_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_key}",
        )

    # Delegate to service layer for export formatting
    content = await export_dataset_to_jsonl(manifest, db, format)

    # Return as NDJSON
    return Response(content=content, media_type="application/x-ndjson")
