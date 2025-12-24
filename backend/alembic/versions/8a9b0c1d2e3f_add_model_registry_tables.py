"""add_model_registry_tables

Revision ID: 8a9b0c1d2e3f
Revises: 7f8a9b0c1d2e
Create Date: 2025-12-23 21:40:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8a9b0c1d2e3f"
down_revision: Union[str, Sequence[str], None] = "7f8a9b0c1d2e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create model_artifacts table
    op.create_table(
        "model_artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task_type", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_artifacts_name", "model_artifacts", ["name"], unique=True)

    # Create model_versions table
    op.create_table(
        "model_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("artifact_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("source_run_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="created"),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("provenance_json", sa.JSON(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["artifact_id"], ["model_artifacts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_run_id"], ["tunix_runs.run_id"], ondelete="SET NULL"),
        sa.UniqueConstraint("artifact_id", "version", name="uq_model_versions_artifact_version"),
    )
    op.create_index("ix_model_versions_sha256", "model_versions", ["sha256"])
    op.create_index("ix_model_versions_created_at", "model_versions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_versions_created_at", table_name="model_versions")
    op.drop_index("ix_model_versions_sha256", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_model_artifacts_name", table_name="model_artifacts")
    op.drop_table("model_artifacts")
