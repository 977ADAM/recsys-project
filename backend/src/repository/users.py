from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.src.repository.models.users import User


def get_by_user_id(db: Session, user_id: str) -> User | None:
    return db.execute(
        select(User).where(User.user_id == user_id)
    ).scalar_one_or_none()


def create_user(
    db: Session,
    *,
    user_id: str,
    age: int,
    gender: str,
    city_tier: str,
    device_os: str,
    platform: str,
    income_band: str,
    activity_segment: str,
    interest_1: str,
    interest_2: str,
    interest_3: str,
    country: str,
    signup_days_ago: int,
    is_premium: bool,
) -> User:
    user = User(
        user_id=user_id,
        age=age,
        gender=gender,
        city_tier=city_tier,
        device_os=device_os,
        platform=platform,
        income_band=income_band,
        activity_segment=activity_segment,
        interest_1=interest_1,
        interest_2=interest_2,
        interest_3=interest_3,
        country=country,
        signup_days_ago=signup_days_ago,
        is_premium=is_premium,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.user_id)).all())


def get_user(db: Session, user_id: str) -> User | None:
    return db.execute(
        select(User).where(User.user_id == user_id)
    ).scalar_one_or_none()
