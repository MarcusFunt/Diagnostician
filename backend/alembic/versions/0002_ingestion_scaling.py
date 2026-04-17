"""ingestion progress and case listing indexes

Revision ID: 0002_ingestion_scaling
Revises: 0001_initial_schema
Create Date: 2026-04-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_ingestion_scaling"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("current_offset", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("accepted_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("playable_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_ingestion_runs_source_path", "ingestion_runs", ["source_path"])
    op.create_index("ix_ingestion_runs_source_hash", "ingestion_runs", ["source_hash"])
    op.create_index("ix_ingestion_runs_status", "ingestion_runs", ["status"])
    op.create_index("ix_source_documents_path", "source_documents", ["path"])
    op.create_index("ix_cases_created_at", "cases", ["created_at"])
    op.create_index("ix_cases_difficulty", "cases", ["difficulty"])
    op.create_index(
        "ix_cases_review_specialty_difficulty_created",
        "cases",
        ["review_status", "specialty", "difficulty", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_cases_review_specialty_difficulty_created", table_name="cases")
    op.drop_index("ix_cases_difficulty", table_name="cases")
    op.drop_index("ix_cases_created_at", table_name="cases")
    op.drop_index("ix_source_documents_path", table_name="source_documents")
    op.drop_index("ix_ingestion_runs_status", table_name="ingestion_runs")
    op.drop_index("ix_ingestion_runs_source_hash", table_name="ingestion_runs")
    op.drop_index("ix_ingestion_runs_source_path", table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
