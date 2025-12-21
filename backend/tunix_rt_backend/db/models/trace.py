"""Trace model for storing reasoning traces."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tunix_rt_backend.db.base import Base

if TYPE_CHECKING:
    from tunix_rt_backend.db.models.score import Score


class Trace(Base):
    """SQLAlchemy model for storing reasoning traces.

    Attributes:
        id: UUID primary key
        created_at: Timestamp when trace was created (timezone-aware UTC)
        trace_version: Version string of the trace format
        payload: Full trace data as JSON/JSONB
        scores: Relationship to associated Score records
    """

    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    trace_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Relationship to scores
    scores: Mapped[list["Score"]] = relationship(
        "Score", back_populates="trace", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """String representation of Trace."""
        return (
            f"<Trace(id={self.id}, trace_version={self.trace_version}, "
            f"created_at={self.created_at})>"
        )
