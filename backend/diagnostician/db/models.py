from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from diagnostician.core.windows_platform import disable_slow_wmi_platform_probe

disable_slow_wmi_platform_probe()

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from diagnostician.core.config import get_settings
from diagnostician.db.session import Base


def _uuid() -> UUID:
    return uuid4()


def json_type():
    return JSON().with_variant(JSONB, "postgresql")


class SourceDocumentRow(Base):
    __tablename__ = "source_documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    deidentified: Mapped[bool] = mapped_column(nullable=False, default=True)
    citation: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(json_type(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CaseRow(Base):
    __tablename__ = "cases"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String(64), nullable=False)
    specialty: Mapped[str | None] = mapped_column(String(128), index=True)
    final_diagnosis: Mapped[str] = mapped_column(Text, nullable=False)
    truth_payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    facts: Mapped[list[CaseFactRow]] = relationship(
        back_populates="case", cascade="all, delete-orphan", lazy="selectin"
    )


class CaseFactRow(Base):
    __tablename__ = "case_facts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    spoiler: Mapped[bool] = mapped_column(nullable=False, default=False, index=True)
    initially_visible: Mapped[bool] = mapped_column(nullable=False, default=False)
    payload: Mapped[dict] = mapped_column(json_type(), nullable=False)

    case: Mapped[CaseRow] = relationship(back_populates="facts")
    embedding: Mapped[CaseFactEmbeddingRow | None] = relationship(
        back_populates="fact", cascade="all, delete-orphan", uselist=False
    )


class CaseFactEmbeddingRow(Base):
    __tablename__ = "case_fact_embeddings"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    fact_id: Mapped[UUID] = mapped_column(
        ForeignKey("case_facts.id", ondelete="CASCADE"), unique=True, index=True
    )
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(get_settings().embedding_dimensions))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    fact: Mapped[CaseFactRow] = relationship(back_populates="embedding")


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    score: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    request_payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    response_payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ValidationLogRow(Base):
    __tablename__ = "validation_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    turn_id: Mapped[UUID | None] = mapped_column(ForeignKey("turns.id", ondelete="SET NULL"))
    payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScoreRow(Base):
    __tablename__ = "scores"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), unique=True)
    payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewRow(Base):
    __tablename__ = "reviews"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=_uuid)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), unique=True)
    payload: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
