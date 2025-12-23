"""add_tunix_runs_table

Revision ID: 4bf76cdb97da
Revises: f3cc010ca8a6
Create Date: 2025-12-22 20:16:23.153841

This migration adds the tunix_runs table for persisting Tunix training run metadata.

M14 Design:
- Stores all Tunix runs (dry-run + local modes) for audit trail
- Includes stdout/stderr (truncated to 10KB at capture time)
- Forward-compatible with M15 async execution (includes 'pending' status)
- No FK dependencies (dataset_key is string reference, not relational)

Nullable Fields:
- exit_code: NULL for dry-run and timeout cases
- completed_at: NULL only if execution never completed (system crash)
- duration_seconds: NULL only if execution never completed

Indexes:
- dataset_key: For filtering runs by dataset
- created_at: For pagination and recent-run queries
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4bf76cdb97da"
down_revision: Union[str, Sequence[str], None] = "f3cc010ca8a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - create tunix_runs table."""
    op.create_table(
        "tunix_runs",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("dataset_key", sa.String(length=256), nullable=False),
        sa.Column("model_id", sa.String(length=256), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("stdout", sa.Text(), nullable=False),
        sa.Column("stderr", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    # Index for filtering runs by dataset
    op.create_index("ix_tunix_runs_dataset_key", "tunix_runs", ["dataset_key"])
    # Index for pagination and recent-run queries
    op.create_index("ix_tunix_runs_created_at", "tunix_runs", ["created_at"])


def downgrade() -> None:
    """Downgrade schema - drop tunix_runs table."""
    op.drop_index("ix_tunix_runs_created_at", table_name="tunix_runs")
    op.drop_index("ix_tunix_runs_dataset_key", table_name="tunix_runs")
    op.drop_table("tunix_runs")
