"""add_regression_baselines_table

Revision ID: 6e7f8a9b0c1d
Revises: 5d6e7f8a9b0c
Create Date: 2025-12-23 15:00:00.000000

This migration adds the 'regression_baselines' table for M18.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6e7f8a9b0c1d"
down_revision: Union[str, Sequence[str], None] = "5d6e7f8a9b0c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create regression_baselines table
    op.create_table(
        "regression_baselines",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["tunix_runs.run_id"]),
    )

    # Create index
    op.create_index(
        "ix_regression_baselines_name",
        "regression_baselines",
        ["name"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_regression_baselines_name", table_name="regression_baselines")
    op.drop_table("regression_baselines")
