from datetime import date
from decimal import Decimal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from backend.src.core.schemas.schema import BannerFormat, CampaignGoal, Gender


class BannerCreate(BaseModel):
    banner_id: str = Field(min_length=1, max_length=20)
    brand: str = Field(min_length=1, max_length=50)
    category: str = Field(min_length=1, max_length=50)
    subcategory: str = Field(min_length=1, max_length=100)
    banner_format: BannerFormat
    campaign_goal: CampaignGoal
    target_gender: Gender
    target_age_min: int = Field(ge=0, le=32767)
    target_age_max: int = Field(ge=0, le=32767)
    cpm_bid: Decimal = Field(ge=0)
    quality_score: Decimal = Field(ge=0, le=1)
    created_at: date
    is_active: bool
    landing_page: AnyHttpUrl

    @model_validator(mode="after")
    def validate_age_range(self) -> "BannerCreate":
        if self.target_age_max < self.target_age_min:
            raise ValueError("target_age_max must be greater than or equal to target_age_min")
        return self


class BannerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    banner_id: str
    brand: str
    category: str
    subcategory: str
    banner_format: BannerFormat
    campaign_goal: CampaignGoal
    target_gender: Gender
    target_age_min: int
    target_age_max: int
    cpm_bid: Decimal
    quality_score: Decimal
    created_at: date
    is_active: bool
    landing_page: AnyHttpUrl


class BannersResponse(BaseModel):
    banners: list[BannerResponse]
