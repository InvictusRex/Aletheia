"""initial phase1 (documents/pages/evidence_units)

Revision ID: 7f3a9c1d2e4b
Revises:
Create Date: 2026-09-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "7f3a9c1d2e4b"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("file_sha256", sa.String(), nullable=False),
        sa.Column("ingestion_status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "pages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("pdf_page_number", sa.Integer(), nullable=False),
        sa.Column("source_page_number", sa.String(), nullable=True),
        sa.Column("width", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("suspicious_ratio", sa.Float(), nullable=False),
        sa.Column("extraction_quality", sa.Float(), nullable=False),
        sa.Column("has_images", sa.Boolean(), nullable=False),
        sa.Column("quality_verdict", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "pdf_page_number"),
    )
    op.create_table(
        "evidence_units",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("pdf_page_number", sa.Integer(), nullable=False),
        sa.Column("source_page_number", sa.String(), nullable=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("extraction_method", sa.String(), nullable=False),
        sa.Column("extraction_quality", sa.Float(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evidence_units_document_page",
        "evidence_units",
        ["document_id", "pdf_page_number"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_evidence_units_document_page", table_name="evidence_units")
    op.drop_table("evidence_units")
    op.drop_table("pages")
    op.drop_table("documents")
