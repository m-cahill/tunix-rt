"""Model Registry schemas (M20)."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelArtifactCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9._-]+$")
    description: str | None = None
    task_type: str | None = None


class ModelVersionRead(BaseModel):
    id: uuid.UUID
    artifact_id: uuid.UUID
    version: str
    source_run_id: uuid.UUID | None
    status: str
    metrics_json: dict[str, Any] | None
    config_json: dict[str, Any] | None
    provenance_json: dict[str, Any] | None
    storage_uri: str
    sha256: str
    size_bytes: int
    created_at: datetime


class ModelArtifactRead(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    task_type: str | None
    created_at: datetime
    updated_at: datetime
    latest_version: ModelVersionRead | None = None


class ModelPromotionRequest(BaseModel):
    source_run_id: uuid.UUID
    version_label: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    description: str | None = None
