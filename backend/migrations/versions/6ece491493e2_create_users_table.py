"""create users table

Revision ID: 6ece491493e2
Revises: 
Create Date: 2026-04-08 21:08:45.309596

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6ece491493e2'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=20), nullable=False),
        sa.Column("age", sa.SmallInteger(), nullable=False),
        sa.Column("gender", sa.String(length=1), nullable=False),
        sa.Column("city_tier", sa.String(length=10), nullable=False),
        sa.Column("device_os", sa.String(length=20), nullable=False),
        sa.Column("platform", sa.String(length=20), nullable=False),
        sa.Column("income_band", sa.String(length=10), nullable=False),
        sa.Column("activity_segment", sa.String(length=10), nullable=False),
        sa.Column("interest_1", sa.String(length=50), nullable=False),
        sa.Column("interest_2", sa.String(length=50), nullable=False),
        sa.Column("interest_3", sa.String(length=50), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("signup_days_ago", sa.Integer(), nullable=False),
        sa.Column("is_premium", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("users")
