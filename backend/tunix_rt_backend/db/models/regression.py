"""Regression baseline model for tracking quality gates."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tunix_rt_backend.db.base import Base
from tunix_rt_backend.db.models.tunix_run import TunixRun


class RegressionBaseline(Base):
    """Stores named baselines for regression testing.

    A baseline links a specific metric to a specific run, allowing
    future runs to be compared against this historical performance.
    """

    __tablename__ = "regression_baselines"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tunix_runs.run_id"), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    run: Mapped[TunixRun] = relationship()

    __table_args__ = (Index("ix_regression_baselines_name", "name", unique=True),)
