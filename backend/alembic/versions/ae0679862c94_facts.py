"""facts + fact_evidence (F-DB fact persistence)

Revision ID: ae0679862c94
Revises: 7f3a9c1d2e4b
Create Date: 2026-09-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "ae0679862c94"
down_revision: Union[str, None] = "7f3a9c1d2e4b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "facts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("canonical_subject", sa.String(), nullable=True),
        sa.Column("predicate", sa.String(), nullable=False),
        sa.Column("canonical_predicate", sa.String(), nullable=True),
        sa.Column("value_kind", sa.String(), nullable=False),
        sa.Column("value_text", sa.Text(), nullable=False),
        sa.Column("value_number", sa.Float(), nullable=True),
        sa.Column("unit", sa.String(), nullable=True),
        sa.Column("normalized_number", sa.Float(), nullable=True),
        sa.Column("normalized_unit", sa.String(), nullable=True),
        sa.Column("time_text", sa.String(), nullable=True),
        sa.Column("time_kind", sa.String(), nullable=False),
        sa.Column("time_start", sa.Date(), nullable=True),
        sa.Column("time_end", sa.Date(), nullable=True),
        sa.Column("scope_text", sa.String(), nullable=True),
        sa.Column("estimate_status", sa.String(), nullable=False),
        sa.Column("geography", sa.String(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("extraction_confidence", sa.Float(), nullable=False),
        sa.Column("ambiguity_flags", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_facts_document_id",
        "facts",
        ["document_id"],
        unique=False,
    )
    op.create_table(
        "fact_evidence",
        sa.Column("fact_id", sa.Uuid(), nullable=False),
        sa.Column("evidence_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["evidence_id"], ["evidence_units.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("fact_id", "evidence_id"),
    )


def downgrade() -> None:
    op.drop_table("fact_evidence")
    op.drop_index("ix_facts_document_id", table_name="facts")
    op.drop_table("facts")
