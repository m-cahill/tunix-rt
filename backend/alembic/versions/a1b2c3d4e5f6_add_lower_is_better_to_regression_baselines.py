"""add_lower_is_better_to_regression_baselines

Revision ID: a1b2c3d4e5f6
Revises: 9b0c1d2e3f4g
Create Date: 2025-12-25 12:00:00.000000

This migration adds the 'lower_is_better' column to regression_baselines for M29.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "9b0c1d2e3f4g"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add lower_is_better column to regression_baselines
    op.add_column(
        "regression_baselines",
        sa.Column("lower_is_better", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("regression_baselines", "lower_is_better")
