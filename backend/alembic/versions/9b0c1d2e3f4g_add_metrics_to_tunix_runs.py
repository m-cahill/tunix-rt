"""Add metrics to TunixRun

Revision ID: 9b0c1d2e3f4g
Revises: 8a9b0c1d2e3f
Create Date: 2025-12-24 18:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b0c1d2e3f4g"
down_revision: Union[str, None] = "8a9b0c1d2e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tunix_runs", sa.Column("metrics", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("tunix_runs", "metrics")
