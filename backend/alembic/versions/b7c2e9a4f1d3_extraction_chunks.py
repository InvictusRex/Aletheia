"""extraction_chunks (per-chunk extraction progress for resume)

Revision ID: b7c2e9a4f1d3
Revises: f42014770cfa
Create Date: 2026-09-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b7c2e9a4f1d3"
down_revision: Union[str, None] = "f42014770cfa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "extraction_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("pdf_page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("fact_ids", sa.JSON(), nullable=False),
        sa.Column("evidence_hash", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "pdf_page_number", "chunk_index"),
    )
    op.create_index(
        "ix_extraction_chunks_document_id",
        "extraction_chunks",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_extraction_chunks_document_id", table_name="extraction_chunks")
    op.drop_table("extraction_chunks")
