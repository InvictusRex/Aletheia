"""SQLAlchemy ORM tables for Phase 1 provenance-first PDF ingestion.

Pydantic models in ``app.models`` own the vocabulary (enums); these ORM
rows store enums as plain strings.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for Aletheia ORM models."""


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String, nullable=False)
    ingestion_status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    pages: Mapped[list[PageRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    evidence_units: Mapped[list[EvidenceUnitRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PageRow(Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("document_id", "pdf_page_number"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    pdf_page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_page_number: Mapped[str | None] = mapped_column(String, nullable=True)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    suspicious_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    extraction_quality: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    has_images: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_verdict: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[DocumentRow] = relationship(back_populates="pages")


class EvidenceUnitRow(Base):
    __tablename__ = "evidence_units"
    __table_args__ = (
        Index("ix_evidence_units_document_page", "document_id", "pdf_page_number"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    pdf_page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_page_number: Mapped[str | None] = mapped_column(String, nullable=True)
    type: Mapped[str] = mapped_column("type", String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bbox: Mapped[list | None] = mapped_column(JSON, nullable=True)
    extraction_method: Mapped[str] = mapped_column(String, nullable=False)
    extraction_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    document: Mapped[DocumentRow] = relationship(back_populates="evidence_units")
