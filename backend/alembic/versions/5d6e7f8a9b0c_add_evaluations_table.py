"""add_evaluations_table

Revision ID: 5d6e7f8a9b0c
Revises: 3c4d5e6f7a8b
Create Date: 2025-12-23 13:00:00.000000

This migration adds the 'tunix_run_evaluations' table for M17.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5d6e7f8a9b0c"
down_revision: Union[str, Sequence[str], None] = "3c4d5e6f7a8b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create tunix_run_evaluations table
    op.create_table(
        "tunix_run_evaluations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("judge_name", sa.String(length=64), nullable=False),
        sa.Column("judge_version", sa.String(length=32), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["tunix_runs.run_id"], ondelete="CASCADE"),
    )

    # Create indexes
    op.create_index(
        "ix_tunix_run_evaluations_run_id",
        "tunix_run_evaluations",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_tunix_run_evaluations_score",
        "tunix_run_evaluations",
        ["score"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_tunix_run_evaluations_score", table_name="tunix_run_evaluations")
    op.drop_index("ix_tunix_run_evaluations_run_id", table_name="tunix_run_evaluations")
    op.drop_table("tunix_run_evaluations")
