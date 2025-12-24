"""Model Registry models (M20).

Stores metadata for Model Artifacts (families) and specific Model Versions.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tunix_rt_backend.db.base import Base


class ModelArtifact(Base):
    """Represents a logical model family (e.g., 'Gemma3-1B Tunix SFT')."""

    __tablename__ = "model_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="artifact", cascade="all, delete-orphan"
    )

    # Indexes/Constraints
    __table_args__ = (Index("ix_model_artifacts_name", "name", unique=True),)


class ModelVersion(Base):
    """Represents a specific versioned build of a ModelArtifact."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    # Optional link to source TunixRun
    source_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tunix_runs.run_id", ondelete="SET NULL"), nullable=True
    )

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    # Statuses: created, ready, failed

    # Metadata snapshots
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    provenance_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Storage info
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    artifact: Mapped["ModelArtifact"] = relationship(back_populates="versions")
    source_run = relationship("TunixRun")

    # Indexes/Constraints
    __table_args__ = (
        UniqueConstraint("artifact_id", "version", name="uq_model_versions_artifact_version"),
        Index("ix_model_versions_sha256", "sha256"),
        Index("ix_model_versions_created_at", "created_at"),
    )
