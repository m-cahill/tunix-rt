"""add_scores_table

Revision ID: f3cc010ca8a6
Revises: f8f1393630e4
Create Date: 2025-12-21 10:43:55.134384

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3cc010ca8a6"
down_revision: Union[str, Sequence[str], None] = "f8f1393630e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "scores",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("trace_id", sa.UUID(), nullable=False),
        sa.Column("criteria", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["trace_id"], ["traces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # Index for querying scores by trace
    op.create_index("ix_scores_trace_id", "scores", ["trace_id"])
    # Index for querying scores by criteria
    op.create_index("ix_scores_criteria", "scores", ["criteria"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_scores_criteria", table_name="scores")
    op.drop_index("ix_scores_trace_id", table_name="scores")
    op.drop_table("scores")
