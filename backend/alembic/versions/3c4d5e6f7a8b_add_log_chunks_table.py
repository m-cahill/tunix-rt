"""add_log_chunks_table

Revision ID: 3c4d5e6f7a8b
Revises: 2b3a4c5d6e7f
Create Date: 2025-12-23 12:00:00.000000

This migration adds the 'tunix_run_log_chunks' table for real-time log streaming.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c4d5e6f7a8b"
down_revision: Union[str, Sequence[str], None] = "2b3a4c5d6e7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create tunix_run_log_chunks table
    op.create_table(
        "tunix_run_log_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("stream", sa.String(), nullable=False),
        sa.Column("chunk", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["tunix_runs.run_id"], ondelete="CASCADE"),
    )

    # Create index for fast retrieval by run and sequence
    op.create_index(
        "ix_tunix_run_log_chunks_run_id_seq",
        "tunix_run_log_chunks",
        ["run_id", "seq"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_tunix_run_log_chunks_run_id_seq", table_name="tunix_run_log_chunks")
    op.drop_table("tunix_run_log_chunks")
