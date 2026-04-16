"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "source_documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("deidentified", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("citation", sa.Text()),
        sa.Column("raw_text", sa.Text()),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "cases",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("review_status", sa.String(32), nullable=False),
        sa.Column("difficulty", sa.String(64), nullable=False),
        sa.Column("specialty", sa.String(128)),
        sa.Column("final_diagnosis", sa.Text(), nullable=False),
        sa.Column("truth_payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_cases_review_status", "cases", ["review_status"])
    op.create_index("ix_cases_specialty", "cases", ["specialty"])

    op.create_table(
        "case_facts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("case_id", sa.Uuid(), sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("spoiler", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("initially_visible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_case_facts_case_id", "case_facts", ["case_id"])
    op.create_index("ix_case_facts_category", "case_facts", ["category"])
    op.create_index("ix_case_facts_spoiler", "case_facts", ["spoiler"])

    op.create_table(
        "case_fact_embeddings",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("fact_id", sa.Uuid(), sa.ForeignKey("case_facts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_case_fact_embeddings_fact_id", "case_fact_embeddings", ["fact_id"], unique=True)
    op.create_index(
        "ix_case_fact_embeddings_embedding",
        "case_fact_embeddings",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_with={"lists": 100},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("case_id", sa.Uuid(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("score", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_runs_case_id", "runs", ["case_id"])
    op.create_index("ix_runs_status", "runs", ["status"])

    op.create_table(
        "turns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(), nullable=False),
        sa.Column("response_payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_turns_run_id", "turns", ["run_id"])

    op.create_table(
        "validation_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("turn_id", sa.Uuid(), sa.ForeignKey("turns.id", ondelete="SET NULL")),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_validation_logs_run_id", "validation_logs", ["run_id"])

    op.create_table(
        "scores",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "reviews",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("reviews")
    op.drop_table("scores")
    op.drop_index("ix_validation_logs_run_id", table_name="validation_logs")
    op.drop_table("validation_logs")
    op.drop_index("ix_turns_run_id", table_name="turns")
    op.drop_table("turns")
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_case_id", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_case_fact_embeddings_embedding", table_name="case_fact_embeddings")
    op.drop_index("ix_case_fact_embeddings_fact_id", table_name="case_fact_embeddings")
    op.drop_table("case_fact_embeddings")
    op.drop_index("ix_case_facts_spoiler", table_name="case_facts")
    op.drop_index("ix_case_facts_category", table_name="case_facts")
    op.drop_index("ix_case_facts_case_id", table_name="case_facts")
    op.drop_table("case_facts")
    op.drop_index("ix_cases_specialty", table_name="cases")
    op.drop_index("ix_cases_review_status", table_name="cases")
    op.drop_table("cases")
    op.drop_table("source_documents")
