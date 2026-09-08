"""relationships + fact_embeddings (R-DB matching persistence)

Revision ID: f42014770cfa
Revises: ae0679862c94
Create Date: 2026-09-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "f42014770cfa"
down_revision: Union[str, None] = "ae0679862c94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "fact_embeddings",
        sa.Column("fact_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("fact_id"),
    )
    op.create_table(
        "relationships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("fact_a_id", sa.Uuid(), nullable=False),
        sa.Column("fact_b_id", sa.Uuid(), nullable=False),
        sa.Column("relationship_type", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("reasoning_metadata", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["fact_a_id"], ["facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["fact_b_id"], ["facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_a_id", "fact_b_id"),
    )
    op.create_index(
        "ix_relationships_fact_a_id",
        "relationships",
        ["fact_a_id"],
        unique=False,
    )
    op.create_index(
        "ix_relationships_fact_b_id",
        "relationships",
        ["fact_b_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_relationships_fact_b_id", table_name="relationships")
    op.drop_index("ix_relationships_fact_a_id", table_name="relationships")
    op.drop_table("relationships")
    op.drop_table("fact_embeddings")
    # NOTE: the ``vector`` extension is intentionally left installed on
    # downgrade — it is shared/cluster-scoped and other objects may depend
    # on it; dropping it could break unrelated schemas.
