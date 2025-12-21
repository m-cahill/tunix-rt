"""add traces created_at index

Revision ID: f8f1393630e4
Revises: 001
Create Date: 2025-12-21 07:47:02.314138

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f8f1393630e4"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add index on traces.created_at for list performance."""
    op.create_index(
        "ix_traces_created_at",  # Explicit index name for cross-DB consistency
        "traces",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove index on traces.created_at."""
    op.drop_index("ix_traces_created_at", table_name="traces")
