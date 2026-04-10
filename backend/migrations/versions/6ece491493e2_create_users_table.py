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
    op.create_table(
        "banners",
        sa.Column("banner_id", sa.String(length=20), nullable=False),
        sa.Column("brand", sa.String(length=50), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("subcategory", sa.String(length=100), nullable=False),
        sa.Column("banner_format", sa.String(length=20), nullable=False),
        sa.Column("campaign_goal", sa.String(length=30), nullable=False),
        sa.Column("target_gender", sa.String(length=1), nullable=False),
        sa.Column("target_age_min", sa.SmallInteger(), nullable=False),
        sa.Column("target_age_max", sa.SmallInteger(), nullable=False),
        sa.Column("cpm_bid", sa.Numeric(10, 2), nullable=False),
        sa.Column("quality_score", sa.Numeric(4, 3), nullable=False),
        sa.Column("created_at", sa.Date(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("landing_page", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("banner_id"),
        sa.CheckConstraint("target_gender IN ('M', 'F', 'U')", name="ck_banners_target_gender"),
        sa.CheckConstraint(
            "banner_format IN ('static', 'native', 'animated', 'video')",
            name="ck_banners_banner_format",
        ),
        sa.CheckConstraint(
            "campaign_goal IN ('awareness', 'app_install', 'purchase', 'traffic', 'lead_gen')",
            name="ck_banners_campaign_goal",
        ),
        sa.CheckConstraint(
            "target_age_min >= 0 AND target_age_max >= target_age_min",
            name="ck_banners_age_range",
        ),
        sa.CheckConstraint(
            "quality_score BETWEEN 0 AND 1",
            name="ck_banners_quality_score",
        ),
    )


def downgrade() -> None:
    """
    Downgrade schema.
    Удаление таблиц должно быть в обраьной последовательности.
    """
    op.drop_table("banners")
    op.drop_table("users")
