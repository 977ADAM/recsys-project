from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.src.repository.models.banners import Banner


def get_banner_by_id(db: Session, banner_id: str) -> Banner | None:
    return db.execute(
        select(Banner).where(Banner.banner_id == banner_id)
    ).scalar_one_or_none()


def create_banner(
    db: Session,
    *,
    banner_id: str,
    brand: str,
    category: str,
    subcategory: str,
    banner_format: str,
    campaign_goal: str,
    target_gender: str,
    target_age_min: int,
    target_age_max: int,
    cpm_bid: Decimal,
    quality_score: Decimal,
    created_at: date,
    is_active: bool,
    landing_page: str,
) -> Banner:
    banner = Banner(
        banner_id=banner_id,
        brand=brand,
        category=category,
        subcategory=subcategory,
        banner_format=banner_format,
        campaign_goal=campaign_goal,
        target_gender=target_gender,
        target_age_min=target_age_min,
        target_age_max=target_age_max,
        cpm_bid=cpm_bid,
        quality_score=quality_score,
        created_at=created_at,
        is_active=is_active,
        landing_page=landing_page,
    )
    db.add(banner)
    db.commit()
    db.refresh(banner)
    return banner


def get_banners(db: Session) -> list[Banner]:
    return list(db.scalars(select(Banner).order_by(Banner.banner_id)).all())


def get_banner(db: Session, banner_id: str) -> Banner | None:
    return db.execute(
        select(Banner).where(Banner.banner_id == banner_id)
    ).scalar_one_or_none()


def delete_banner(db: Session, banner_id: str) -> Banner | None:
    banner = db.scalar(select(Banner).where(Banner.banner_id == banner_id))

    if banner:
        db.delete(banner)
        db.commit()

    return banner


def patch_banner(db: Session, banner_id: str, **fields) -> Banner | None:
    banner = db.scalar(select(Banner).where(Banner.banner_id == banner_id))
    if banner is None:
        return None

    for field_name, value in fields.items():
        setattr(banner, field_name, value)

    db.commit()
    db.refresh(banner)
    return banner
