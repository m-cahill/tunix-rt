"""TunixRun model for storing Tunix training run metadata and results."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from tunix_rt_backend.db.base import Base


class TunixRun(Base):
    """SQLAlchemy model for storing Tunix training run metadata.

    This table persists all Tunix run executions (both dry-run and local modes)
    to provide an audit trail and enable run history tracking.

    Attributes:
        run_id: UUID primary key
        dataset_key: Dataset identifier (e.g., 'my_dataset-v1')
        model_id: Model identifier (e.g., 'google/gemma-2b-it')
        mode: Execution mode ('dry-run' or 'local')
        status: Execution status ('pending', 'running', 'completed', 'failed', 'timeout')
        exit_code: Process exit code (NULL for dry-run and timeout cases)
        started_at: Timestamp when execution started (timezone-aware UTC)
        completed_at: Timestamp when execution completed (NULL if never completed)
        duration_seconds: Execution duration in seconds (NULL if never completed)
        stdout: Standard output from execution (truncated to 10KB)
        stderr: Standard error from execution (truncated to 10KB)
        created_at: Record creation timestamp (timezone-aware UTC)

    Notes:
        - M14: Records are created immediately with status='running', then updated
        - M15+: 'pending' status will be used for async/background execution
        - exit_code is NULL for dry-run (validation only) and timeout cases
        - stdout/stderr are truncated at capture time to 10KB (M13 behavior)
        - completed_at and duration_seconds are NULL only if execution never finished
    """

    __tablename__ = "tunix_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_key: Mapped[str] = mapped_column(String(256), nullable=False)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)  # 'dry-run' or 'local'
    status: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # 'pending', 'running', 'completed', 'failed', 'timeout'
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    stdout: Mapped[str] = mapped_column(Text, nullable=False, default="")
    stderr: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_tunix_runs_dataset_key", "dataset_key"),
        Index("ix_tunix_runs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of TunixRun."""
        return (
            f"<TunixRun(run_id={self.run_id}, dataset_key={self.dataset_key}, "
            f"mode={self.mode}, status={self.status}, created_at={self.created_at})>"
        )


class TunixRunLogChunk(Base):
    """Log chunk for real-time streaming of Tunix run output."""

    __tablename__ = "tunix_run_log_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tunix_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    stream: Mapped[str] = mapped_column(String(16), nullable=False)  # 'stdout' or 'stderr'
    chunk: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (Index("ix_tunix_run_log_chunks_run_id_seq", "run_id", "seq"),)
