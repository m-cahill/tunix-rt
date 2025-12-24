"""Evaluation database models (M17)."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tunix_rt_backend.db.base import Base


class TunixRunEvaluation(Base):
    """Database model for run evaluations.

    Stores high-level metrics for leaderboard sorting/filtering.
    Full details are stored in the 'details' JSON column (mirroring the artifact).
    """

    __tablename__ = "tunix_run_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tunix_runs.run_id", ondelete="CASCADE"), index=True
    )

    # Core sortable metrics
    score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)  # pass, fail, etc.
    judge_name: Mapped[str] = mapped_column(String(64), nullable=False)
    judge_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # Full payload
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    run = relationship("TunixRun", backref="evaluations")
