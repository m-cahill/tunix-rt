"""Model registry endpoints.

Domain: Model artifact storage and version management

Primary endpoints:
- POST /api/models/artifacts: Create logical model family
- GET /api/models/artifacts: List all artifacts
- GET /api/models/artifacts/{id}: Artifact details with versions
- POST /api/models/artifacts/{id}/promote: Promote run to version
- GET /api/models/versions/{id}/download: Download model archive

Cross-cutting concerns:
- Content-addressed storage (SHA256 for deduplication)
- Provenance tracking (source run, dataset, config)
- Version immutability (once created, cannot be modified)
- Temporary file cleanup via background tasks
"""

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from tunix_rt_backend.db.base import get_db
from tunix_rt_backend.schemas.model_registry import (
    ModelArtifactCreate,
    ModelArtifactRead,
    ModelPromotionRequest,
    ModelVersionRead,
)
from tunix_rt_backend.services.model_registry import ModelRegistryService

router = APIRouter()


@router.post("/api/models", response_model=ModelArtifactRead, status_code=status.HTTP_201_CREATED)
async def create_model_artifact(
    request: ModelArtifactCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelArtifactRead:
    """Create a new model artifact family."""
    service = ModelRegistryService(db)
    try:
        artifact = await service.create_artifact(request)
        return ModelArtifactRead(
            id=artifact.id,
            name=artifact.name,
            description=artifact.description,
            task_type=artifact.task_type,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/api/models", response_model=list[ModelArtifactRead])
async def list_model_artifacts(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelArtifactRead]:
    """List all model artifacts."""
    service = ModelRegistryService(db)
    artifacts = await service.list_artifacts()
    return [
        ModelArtifactRead(
            id=a.id,
            name=a.name,
            description=a.description,
            task_type=a.task_type,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in artifacts
    ]


@router.get("/api/models/{artifact_id}", response_model=ModelArtifactRead)
async def get_model_artifact(
    artifact_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelArtifactRead:
    """Get details of a model artifact."""
    service = ModelRegistryService(db)
    artifact = await service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    return ModelArtifactRead(
        id=artifact.id,
        name=artifact.name,
        description=artifact.description,
        task_type=artifact.task_type,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


@router.post("/api/models/{artifact_id}/versions/promote", response_model=ModelVersionRead)
async def promote_run_to_version(
    artifact_id: uuid.UUID,
    request: ModelPromotionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelVersionRead:
    """Promote a TunixRun to a ModelVersion."""
    service = ModelRegistryService(db)
    try:
        version = await service.promote_run(artifact_id, request)
        return ModelVersionRead(
            id=version.id,
            artifact_id=version.artifact_id,
            version=version.version,
            source_run_id=version.source_run_id,
            status=version.status,
            metrics_json=version.metrics_json,
            config_json=version.config_json,
            provenance_json=version.provenance_json,
            storage_uri=version.storage_uri,
            sha256=version.sha256,
            size_bytes=version.size_bytes,
            created_at=version.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/api/models/versions/{version_id}", response_model=ModelVersionRead)
async def get_model_version(
    version_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelVersionRead:
    """Get details of a model version."""
    service = ModelRegistryService(db)
    version = await service.get_version(version_id)
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    return ModelVersionRead(
        id=version.id,
        artifact_id=version.artifact_id,
        version=version.version,
        source_run_id=version.source_run_id,
        status=version.status,
        metrics_json=version.metrics_json,
        config_json=version.config_json,
        provenance_json=version.provenance_json,
        storage_uri=version.storage_uri,
        sha256=version.sha256,
        size_bytes=version.size_bytes,
        created_at=version.created_at,
    )


@router.get("/api/models/versions/{version_id}/download")
async def download_model_version(
    version_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    """Download a model version (as zip stream)."""
    service = ModelRegistryService(db)
    version = await service.get_version(version_id)
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    # Locate directory
    path = service.storage.get(version.storage_uri)

    # Create temp file for zip
    temp_dir = tempfile.mkdtemp()
    base_name = str(Path(temp_dir) / f"{version.version}")

    try:
        # Run zip in thread to avoid blocking
        zip_path = await asyncio.to_thread(shutil.make_archive, base_name, "zip", root_dir=path)

        # Cleanup temp dir after response
        return FileResponse(
            zip_path,
            filename=f"{version.artifact_id}_{version.version}.zip",
            background=BackgroundTask(shutil.rmtree, temp_dir),
        )
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to zip artifact: {e}")
