from sqlalchemy import Boolean, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.src.core.db import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    age: Mapped[int] = mapped_column(SmallInteger)
    gender: Mapped[str] = mapped_column(String(1), index=True)
    city_tier: Mapped[str] = mapped_column(String(10))
    device_os: Mapped[str] = mapped_column(String(20))
    platform: Mapped[str] = mapped_column(String(20))
    income_band: Mapped[str] = mapped_column(String(10))
    activity_segment: Mapped[str] = mapped_column(String(10))
    interest_1: Mapped[str] = mapped_column(String(50))
    interest_2: Mapped[str] = mapped_column(String(50))
    interest_3: Mapped[str] = mapped_column(String(50))
    country: Mapped[str] = mapped_column(String(2))
    signup_days_ago: Mapped[int] = mapped_column(Integer)
    is_premium: Mapped[bool] = mapped_column(Boolean)
