from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from backend.src.schemas.schema import BannerFormat, CampaignGoal


ScoreMode = Literal["ctr", "value"]


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=20)
    top_k: int = Field(default=10, ge=1, le=100)
    score_mode: ScoreMode = "value"
    only_active: bool = False
    exclude_seen: bool = False
    retrieval_artifacts_dir: str | None = None
    retrieval_top_n: int = Field(default=100, ge=1, le=1000)
    as_of_date: date | None = None
    interactions_csv: str | None = None
    users_csv: str | None = None
    banners_csv: str | None = None
    artifacts_dir: str | None = None


class RecommendationItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    banner_id: str
    brand: str
    category: str
    subcategory: str
    banner_format: BannerFormat
    campaign_goal: CampaignGoal
    pred_ctr: float
    final_score: float
    cpm_bid: Decimal
    quality_score: Decimal
    age_match: int
    gender_match: int
    interest_match_any: int
    interest_match_count: int
    banner_ctr_prior: float
    user_ctr_prior: float
    user_subcategory_ctr_prior: float
    user_banner_ctr_prior: float
    served_impressions_total: int
    served_clicks_total: int
    is_active: bool
    landing_page: AnyHttpUrl | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None


class RecommendationResponse(BaseModel):
    user_id: str
    as_of_date: date
    score_mode: ScoreMode
    candidate_mode: Literal["all banners", "retrieval + ranking"]
    model_type: str
    items: list[RecommendationItem]
