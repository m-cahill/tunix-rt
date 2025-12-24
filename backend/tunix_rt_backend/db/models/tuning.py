"""Tuning job models (M19)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tunix_rt_backend.db.base import Base


class TunixTuningJob(Base):
    """Represents a Ray Tune hyperparameter optimization job."""

    __tablename__ = "tunix_tuning_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="created")
    # Statuses: created, running, completed, failed, cancelled

    dataset_key: Mapped[str] = mapped_column(String(256), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(256), nullable=False)
    mode: Mapped[str] = mapped_column(String(64), nullable=False, default="local")

    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "score_mean"
    metric_mode: Mapped[str] = mapped_column(String(16), nullable=False)  # "max" or "min"

    num_samples: Mapped[int] = mapped_column(nullable=False, default=1)
    max_concurrent_trials: Mapped[int] = mapped_column(nullable=False, default=1)

    search_space_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    best_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tunix_runs.run_id"), nullable=True
    )
    best_params_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    ray_storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    trials: Mapped[list["TunixTuningTrial"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    best_run = relationship("TunixRun", foreign_keys=[best_run_id])


class TunixTuningTrial(Base):
    """Represents a single trial within a tuning job."""

    __tablename__ = "tunix_tuning_trials"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Ray trial ID
    tuning_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tunix_tuning_jobs.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tunix_runs.run_id"), nullable=True)

    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    # Statuses: PENDING, RUNNING, TERMINATED, ERROR

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    job: Mapped["TunixTuningJob"] = relationship(back_populates="trials")
    run = relationship("TunixRun")
