from datetime import date
from decimal import Decimal

from sqlalchemy import Boolean, Date, Numeric, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.src.core.db import Base


class Banner(Base):
    __tablename__ = "banners"

    banner_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    brand: Mapped[str] = mapped_column(String(50))
    category: Mapped[str] = mapped_column(String(50))
    subcategory: Mapped[str] = mapped_column(String(100))
    banner_format: Mapped[str] = mapped_column(String(20), index=True)
    campaign_goal: Mapped[str] = mapped_column(String(30), index=True)
    target_gender: Mapped[str] = mapped_column(String(1), index=True)
    target_age_min: Mapped[int] = mapped_column(SmallInteger)
    target_age_max: Mapped[int] = mapped_column(SmallInteger)
    cpm_bid: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    quality_score: Mapped[Decimal] = mapped_column(Numeric(4, 3))
    created_at: Mapped[date] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean)
    landing_page: Mapped[str] = mapped_column(Text)
