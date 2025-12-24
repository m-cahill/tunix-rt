"""add_tuning_jobs_table

Revision ID: 7f8a9b0c1d2e
Revises: 6e7f8a9b0c1d
Create Date: 2025-12-23 15:30:00.000000

This migration adds the 'tunix_tuning_jobs' and 'tunix_tuning_trials' tables for M19.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7f8a9b0c1d2e"
down_revision: Union[str, Sequence[str], None] = "6e7f8a9b0c1d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create tunix_tuning_jobs table
    op.create_table(
        "tunix_tuning_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False, server_default="created"),
        sa.Column("dataset_key", sa.String(length=256), nullable=False),
        sa.Column("base_model_id", sa.String(length=256), nullable=False),
        sa.Column("mode", sa.String(length=64), nullable=False, server_default="local"),
        sa.Column("metric_name", sa.String(length=64), nullable=False),
        sa.Column("metric_mode", sa.String(length=16), nullable=False),
        sa.Column("num_samples", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_concurrent_trials", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("search_space_json", JSONB(), nullable=False),
        sa.Column("best_run_id", sa.UUID(), nullable=True),
        sa.Column("best_params_json", JSONB(), nullable=True),
        sa.Column("ray_storage_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["best_run_id"], ["tunix_runs.run_id"]),
    )

    # Create tunix_tuning_trials table
    op.create_table(
        "tunix_tuning_trials",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tuning_job_id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column("params_json", JSONB(), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["tuning_job_id"], ["tunix_tuning_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["tunix_runs.run_id"]),
    )

    # Indexes
    op.create_index(
        "ix_tunix_tuning_jobs_created_at",
        "tunix_tuning_jobs",
        ["created_at"],
    )
    op.create_index(
        "ix_tunix_tuning_trials_tuning_job_id",
        "tunix_tuning_trials",
        ["tuning_job_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_tunix_tuning_trials_tuning_job_id", table_name="tunix_tuning_trials")
    op.drop_table("tunix_tuning_trials")
    op.drop_index("ix_tunix_tuning_jobs_created_at", table_name="tunix_tuning_jobs")
    op.drop_table("tunix_tuning_jobs")
