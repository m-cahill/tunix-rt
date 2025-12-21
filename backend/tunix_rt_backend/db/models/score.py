"""Score model for storing trace evaluation scores."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tunix_rt_backend.db.base import Base

if TYPE_CHECKING:
    from tunix_rt_backend.db.models.trace import Trace


class Score(Base):
    """SQLAlchemy model for storing trace evaluation scores.

    Attributes:
        id: UUID primary key
        trace_id: Foreign key to traces table
        criteria: Scoring criteria identifier (e.g., 'baseline', 'llm_judge')
        score: Numeric score value (0-100 for baseline scorer)
        details: Additional scoring metadata as JSON
        created_at: Timestamp when score was created (timezone-aware UTC)
        trace: Relationship to parent Trace
    """

    __tablename__ = "scores"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"),
        nullable=False,
    )
    criteria: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship to Trace (back-populates will be added to Trace model)
    trace: Mapped["Trace"] = relationship("Trace", back_populates="scores")

    def __repr__(self) -> str:
        """String representation of Score."""
        return (
            f"<Score(id={self.id}, trace_id={self.trace_id}, criteria={self.criteria}, "
            f"score={self.score}, created_at={self.created_at})>"
        )
