from sqlalchemy.orm import Session

from backend.src.repository.models.banners import Banner
from backend.src.repository.models.users import User
from backend.src.repository import banners as banners_repo
from backend.src.repository import users as users_repo


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_user_id(self, user_id: str) -> User | None:
        return users_repo.get_by_user_id(self.db, user_id)

    def create_user(
        self,
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
        return users_repo.create_user(
            db=self.db,
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

    def get_users(self) -> list[User]:
        return users_repo.get_users(self.db)

    def get_user(self, user_id: str) -> User | None:
        return users_repo.get_user(self.db, user_id)

    def delete_user(self, user_id: str) -> User | None:
        return users_repo.delete_user(self.db, user_id)

    def patch_user(self, user_id: str, **fields) -> User | None:
        return users_repo.patch_user(self.db, user_id, **fields)


class BannerRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_banner_by_id(self, banner_id: str) -> Banner | None:
        return banners_repo.get_banner_by_id(self.db, banner_id)

    def create_banner(
        self,
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
        cpm_bid,
        quality_score,
        created_at,
        is_active: bool,
        landing_page: str,
    ) -> Banner:
        return banners_repo.create_banner(
            db=self.db,
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

    def get_banners(self) -> list[Banner]:
        return banners_repo.get_banners(self.db)

    def get_banner(self, banner_id: str) -> Banner | None:
        return banners_repo.get_banner(self.db, banner_id)

    def delete_banner(self, banner_id: str) -> Banner | None:
        return banners_repo.delete_banner(self.db, banner_id)

    def patch_banner(self, banner_id: str, **fields) -> Banner | None:
        return banners_repo.patch_banner(self.db, banner_id, **fields)
