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


from datetime import date  # noqa: E402 -- appended for FactRow date hints; existing imports above untouched
from sqlalchemy import Date  # noqa: E402 -- appended for FactRow Date columns; existing import block untouched


class FactRow(Base):
    __tablename__ = "facts"
    __table_args__ = (
        Index("ix_facts_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    subject: Mapped[str] = mapped_column(String, nullable=False)
    canonical_subject: Mapped[str | None] = mapped_column(String, nullable=True)
    predicate: Mapped[str] = mapped_column(String, nullable=False)
    canonical_predicate: Mapped[str | None] = mapped_column(String, nullable=True)
    value_kind: Mapped[str] = mapped_column(String, nullable=False)
    value_text: Mapped[str] = mapped_column(Text, nullable=False)
    value_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    normalized_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    normalized_unit: Mapped[str | None] = mapped_column(String, nullable=True)
    time_text: Mapped[str | None] = mapped_column(String, nullable=True)
    time_kind: Mapped[str] = mapped_column(String, nullable=False, default="UNKNOWN")
    time_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    time_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    scope_text: Mapped[str | None] = mapped_column(String, nullable=True)
    estimate_status: Mapped[str] = mapped_column(
        String, nullable=False, default="UNKNOWN"
    )
    geography: Mapped[str | None] = mapped_column(String, nullable=True)
    context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    extraction_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ambiguity_flags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False)

    document: Mapped[DocumentRow] = relationship(back_populates="facts")


class FactEvidenceRow(Base):
    __tablename__ = "fact_evidence"

    fact_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("facts.id", ondelete="CASCADE"), primary_key=True
    )
    evidence_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("evidence_units.id", ondelete="CASCADE"), primary_key=True
    )


# Corresponding collection on DocumentRow. Assigned post-hoc so the existing
# DocumentRow definition above is preserved byte-for-byte.
DocumentRow.facts: Mapped[list["FactRow"]] = relationship(
    "FactRow",
    back_populates="document",
    cascade="all, delete-orphan",
    passive_deletes=True,
)


from pgvector.sqlalchemy import Vector  # noqa: E402 -- appended for R-DB embeddings; existing imports above untouched


class FactEmbeddingRow(Base):
    """One 384-d embedding per fact (R-DB matching persistence)."""

    __tablename__ = "fact_embeddings"

    fact_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("facts.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(384).with_variant(JSON, "sqlite"), nullable=False
    )


class RelationshipRow(Base):
    """Persisted fact-pair relationship (enums stored as plain strings)."""

    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint("fact_a_id", "fact_b_id"),
        Index("ix_relationships_fact_a_id", "fact_a_id"),
        Index("ix_relationships_fact_b_id", "fact_b_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    fact_a_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("facts.id", ondelete="CASCADE"), nullable=False
    )
    fact_b_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("facts.id", ondelete="CASCADE"), nullable=False
    )
    relationship_type: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning_metadata: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="RESOLVED")
