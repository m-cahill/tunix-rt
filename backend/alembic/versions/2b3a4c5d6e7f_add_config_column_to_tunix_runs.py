"""add_config_column_to_tunix_runs

Revision ID: 2b3a4c5d6e7f
Revises: 4bf76cdb97da
Create Date: 2025-12-23 10:00:00.000000

This migration adds the 'config' JSON column to the tunix_runs table.
This allows persisting full request configuration (hyperparameters, etc.)
needed for async worker execution (M15).
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2b3a4c5d6e7f"
down_revision: Union[str, Sequence[str], None] = "4bf76cdb97da"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tunix_runs", sa.Column("config", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("tunix_runs", "config")
